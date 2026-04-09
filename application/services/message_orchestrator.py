"""
Nutribot Backend — MessageOrchestratorService
Orquesta el flujo principal de los mensajes entrantes, coordinando
los servicios de Onboarding, Survey, Extracción y el LLM.
"""
import logging
from typing import Optional
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState, NormalizedMessage, User
from domain.value_objects import OnboardingStatus, OnboardingStep
from domain.utils import get_now_peru
from application.services.onboarding_service import OnboardingService
from application.services.survey_service import SurveyService
from application.services.profile_extraction_service import ProfileExtractionService
from domain.ports import LLMService

logger = logging.getLogger(__name__)

class MessageOrchestratorService:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        openai_model: str,
        onboarding_service: OnboardingService,
        survey_service: SurveyService,
        profile_extractor: ProfileExtractionService,
        llm_service: LLMService,
        system_instructions: str
    ):
        self._openai_client = openai_client
        self._openai_model = openai_model
        self._onboarding_service = onboarding_service
        self._survey_service = survey_service
        self._profile_extractor = profile_extractor
        self._llm_service = llm_service
        self._system_instructions = system_instructions

    def _fmt(self, val, unit=""):
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return "⚠️ Pendiente"
        if isinstance(val, str) and val.upper() == "NINGUNA":
            return "Ninguna ℹ️"
        return f"{val}{unit}"

    def _human_fmt(self, val, unit=""):
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return "No registrado"
        if isinstance(val, str) and val.upper() == "NINGUNA":
            return "Ninguna"
        return f"{val}{unit}"

    async def build_profile_context(self, session: AsyncSession, uid: int) -> tuple[str, str, dict]:
        res = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        p = res.mappings().fetchone()
        if p:
            parts = [
                f"Edad: {self._fmt(p.get('edad'), ' años')}",
                f"Peso: {self._fmt(p.get('peso_kg'), 'kg')}",
                f"Talla: {self._fmt(p.get('altura_cm'), 'cm')}",
                f"Tipo de dieta: {self._fmt(p.get('tipo_dieta'))}",
                f"Alergias: {self._fmt(p.get('alergias'))}",
                f"Enfermedades: {self._fmt(p.get('enfermedades'))}",
                f"Restricciones: {self._fmt(p.get('restricciones_alimentarias'))}",
                f"Objetivo: {self._fmt(p.get('objetivo_nutricional'))}",
                f"Ubicación: {self._fmt(p.get('distrito') or p.get('provincia') or p.get('region'))}"
            ]
            profile_text = "\n[DATOS ACTUALES DEL PERFIL DEL USUARIO]\n- " + "\n- ".join(parts)
            summary = f"• Edad: {self._human_fmt(p.get('edad'), ' años')}\n• Peso: {self._human_fmt(p.get('peso_kg'), 'kg')}\n• Talla: {self._human_fmt(p.get('altura_cm'), 'cm')}\n• Alergias: {self._human_fmt(p.get('alergias'))}\n• Enfermedades: {self._human_fmt(p.get('enfermedades'))}\n• Objetivo: {self._human_fmt(p.get('objetivo_nutricional'))}"
            return profile_text, summary, dict(p)
        else:
            parts = [f"{label}: ⚠️ Pendiente" for label in ["Edad", "Peso", "Talla", "Tipo de dieta", "Alergias", "Enfermedades", "Restricciones", "Objetivo", "Región", "Provincia", "Distrito"]]
            return "\n[DATOS ACTUALES DEL PERFIL DEL USUARIO]\n- " + "\n- ".join(parts), "Aún no tengo datos registrados sobre ti.", {}

    async def process_turn(
        self,
        session: AsyncSession,
        state: ConversationState,
        state_snapshot: ConversationState,
        user: User,
        normalized: NormalizedMessage,
        rag_text: Optional[str],
        factory
    ) -> tuple[str, Optional[str]]:
        
        profile_text, summary, p_map = await self.build_profile_context(session, user.id)
        
        reply = None
        v_text = normalized.text.lower().strip()
        is_asking_for_recommendation = any(w in v_text for w in ["menu", "menú", "receta", "dieta", "qué como", "que como", "comida saludable", "recomienda", "recomendación", "almuerzo", "cena", "desayuno", "coman", "nutricional", "comer"])
        is_short_greeting = len(v_text) < 25 and any(w in v_text for w in ["hola", "buenas", "buenos", "empezar", "arrancar", "nutribot", "que tal", "holis"])
        
        is_requesting_personalization = any(w in v_text for w in ["personalizar", "completar mi perfil", "mis datos", "cambiar mi peso", "actualizar perfil", "personaliza"])
        if not is_requesting_personalization and len(v_text) > 10:
            pers_prompt = f"¿El usuario está expresando deseo de completar su perfil, cambiar sus datos o personalizar más sus respuestas? Responde SOLO 'YES' o 'NO'.\n\nUSUARIO: '{normalized.text}'"
            pers_resp = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "user", "content": pers_prompt}],
                max_tokens=5,
                temperature=0
            )
            is_requesting_personalization = "YES" in pers_resp.choices[0].message.content.strip().upper()

        onboarding_interception_happened = False
        extracted_data = {}
        is_profile_relevant_message = not is_short_greeting

        if is_profile_relevant_message:
            async with factory() as independent_session:
                async with independent_session.begin():
                    extracted_data = await self._profile_extractor.extract_and_save(
                        normalized.text, user.id, independent_session, current_step=state.onboarding_step
                    )
            if extracted_data:
                profile_text, summary, p_map = await self.build_profile_context(session, user.id)

        is_annoyed = any(w in v_text for w in ["ya te dije", "deja de preguntar", "no me preguntes", "qué molesto", "que molesto", "responde", "dame el menú", "dame el menu", "solo quiero"])

        if not is_annoyed and state.onboarding_status in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            is_long_interruption = len(v_text) > 60 and ("?" in v_text or "como" in v_text or "que" in v_text or "ayuda" in v_text)
            if not is_long_interruption:
                onboarding_interception_happened = True
                # Pass extracted_data to avoid double LLM calls
                reply = await self._onboarding_service.advance_flow(
                    normalized.text, state, session, treat_ninguna_as_missing=False, pre_extracted_data=extracted_data
                )
                if reply is None:
                    onboarding_interception_happened = False


        if not onboarding_interception_happened and is_requesting_personalization:
            next_step = await self._onboarding_service._find_next_missing_step(session, user.id, ignore_skips=True, treat_ninguna_as_missing=False)
            if next_step:
                intro = "¡Claro! 🥗 Vamos a chequear tu perfil para que mis consejos sean 100% precisos."
                reply = f"{intro}\n\nEsto es lo que tengo registrado:\n{summary}\n\n¿Deseas corregir algún dato o prefieres que completemos lo pendiente? Empecemos por confirmar tu **{next_step}**... 😊"
                state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
                state.onboarding_step = next_step
                onboarding_interception_happened = True
            else:
                reply = f"¡Ya tengo tu perfil completo! 😊\n\n{summary}\n\nSi quieres cambiar algún dato específico (como tu peso o talla), solo dímelo directamente en cualquier momento."

        if not is_annoyed and not onboarding_interception_happened and state.onboarding_status != OnboardingStatus.COMPLETED.value and (is_short_greeting or is_asking_for_recommendation):
            skipped = p_map.get("skipped_fields", {}) if isinstance(p_map.get("skipped_fields"), dict) else {}
            missing_essential = []
            if not p_map.get("edad") and not skipped.get("edad"): missing_essential.append("edad")
            if not p_map.get("peso_kg") and not skipped.get("peso_kg"): missing_essential.append("peso_kg")
            if not p_map.get("altura_cm") and not skipped.get("altura_cm"): missing_essential.append("altura_cm")

            if is_asking_for_recommendation:
                if not missing_essential:
                    onboarding_interception_happened = False
                else:
                    intro = "¡Claro! 🥗 Me encantaría darte una recomendación a tu medida."
                    known_parts = []
                    if p_map.get("edad"): known_parts.append(f"Edad: {p_map['edad']} años")
                    if p_map.get("peso_kg"): known_parts.append(f"Peso: {p_map['peso_kg']}kg")
                    if p_map.get("altura_cm"): known_parts.append(f"Talla: {p_map['altura_cm']}cm")
                    if known_parts:
                        intro += f" 😊 Veo que ya tengo algunos datos registrados: **{', '.join(known_parts)}**."
                    
                    missing_step = await self._onboarding_service._find_next_missing_step(session, user.id)
                    if missing_step:
                        step_name = "talla (estatura)" if missing_step == "altura_cm" else ("peso" if missing_step == "peso_kg" else missing_step)
                        if "peso_kg" in missing_essential or "altura_cm" in missing_essential:
                            reply = f"{intro}\n\nPero para que mi sugerencia sea 100% precisa y calcular tu IMC, solo me faltaría completar un par de datos más. ¿Te parece si empezamos por tu **{step_name}**? 🙏"
                        else:
                            reply = f"{intro}\n\nSolo me faltaría completar un pequeño detalle para ser más preciso. ¿Te parece si confirmamos tu **{step_name}**? 😊"
                        
                        state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
                        state.onboarding_step = missing_step
                        state.onboarding_last_invited_at = get_now_peru()
                        state.version += 1
                        onboarding_interception_happened = True
                    else:
                        onboarding_interception_happened = False
            elif is_short_greeting:
                if state.onboarding_status == OnboardingStatus.NOT_STARTED.value:
                    reply = "¡Hola! Soy NutriBot 🍏. Para empezar con el pie derecho, ¿te gustaría que personalice mis sugerencias en base a tu perfil nutricional? Es súper rápido. 😊"
                else:
                    reply = "¡Hola de nuevo! 😊 Qué bueno verte. Por cierto, aún nos faltan algunos datos para que mis consejos de hoy sean 100% precisos para ti. ¿Te gustaría completarlos ahora? Es un ratito."
                state.onboarding_status = OnboardingStatus.INVITED.value
                state.onboarding_step = OnboardingStep.INVITACION.value
                state.onboarding_last_invited_at = get_now_peru()
                state.version += 1
                onboarding_interception_happened = True

        new_response_id = state_snapshot.last_openai_response_id
        if not onboarding_interception_happened and reply is None:
            extra_instr = ""
            if extracted_data:
                confirm_list = []
                for k, v in extracted_data.items():
                    c_name = "peso" if k == "peso_kg" else ("talla" if k == "altura_cm" else ("restricciones" if k == "restricciones_alimentarias" else ("objetivo" if k == "objetivo_nutricional" else k)))
                    confirm_list.append(f"{c_name} a '{v}'")
                extra_instr = f"\n\n[INSTRUCCIÓN CRÍTICA: El sistema acaba de actualizar estos datos del perfil: {', '.join(confirm_list)}. DEBES empezar tu respuesta confirmando de forma breve y natural que ya guardaste esta información (ej: '¡Listo! Ya registré tu nuevo peso...'). NO ignores esta instrucción.]"

            final_instructions = self._system_instructions + extra_instr
            final_profile_context = None
            if profile_text:
                final_profile_context = profile_text
                if is_asking_for_recommendation:
                    prefix_ack = "¡Genial! Ya actualicé tu perfil. " if extracted_data else ""
                    citation = f"{prefix_ack}Considerando"
                    if p_map:
                        citation += f" que tienes {p_map.get('edad')} años, pesas {p_map.get('peso_kg')}kg y mides {p_map.get('altura_cm')}cm"
                        if p_map.get('alergias') and p_map.get('alergias').upper() != 'NINGUNA':
                            citation += f", tienes alergia a {p_map.get('alergias')}"
                        if p_map.get('objetivo_nutricional') and p_map.get('objetivo_nutricional').upper() != 'NINGUNA':
                            citation += f" y tu objetivo es {p_map.get('objetivo_nutricional')}"
                    else:
                        citation += " tus datos actuales"
                    citation += ":"

                    final_profile_context = f"""[INSTRUCCIÓN CRÍTICA DE FORMATO]
Tu respuesta DEBE comenzar OBLIGATORIAMENTE con el siguiente texto exacto (no agregues 'Hola' antes de esto):
"{citation}"

[DATOS DE PERFIL PARA TU ANÁLISIS INTERNO]
{profile_text}"""

            reply, new_response_id = await self._llm_service.generate_reply(
                state=state_snapshot,
                normalized=normalized,
                instructions=final_instructions,
                rag_context=rag_text,
                profile_context=final_profile_context,
            )

            if not is_short_greeting:
                state.meaningful_interactions_count += 1

            if state.onboarding_status in [OnboardingStatus.NOT_STARTED.value, OnboardingStatus.SKIPPED.value, OnboardingStatus.PAUSED.value]:
                # Check threshold
                if state.meaningful_interactions_count >= 5:
                    now = get_now_peru()
                    is_eligible = (state.onboarding_status == OnboardingStatus.NOT_STARTED.value) or (state.onboarding_next_eligible_at and now >= state.onboarding_next_eligible_at)
                    is_urgent = len(v_text) < 15 or any(w in v_text for w in ["ayuda", "urgente", "duele", "dolor", "mal", "vomito", "diarrea"])
                    
                    if is_eligible and not is_urgent:
                        # Media suggestions logic
                        res_m = await session.execute(text("SELECT uso_audio, uso_imagen FROM formulario_en_progreso WHERE usuario_id = :uid"), {"uid": user.id})
                        m_row = res_m.fetchone()
                        media_tips = []
                        if m_row:
                            if not m_row.uso_audio: media_tips.append("🎙️ ¿Sabías que también puedes hablarme por **audio**? Es muy cómodo.")
                            if not m_row.uso_imagen: media_tips.append("📸 ¡Prueba enviarme **fotos** de tus platos! Me ayuda a ser más visual.")
                        
                        tip_text = "\n\n" + "\n".join(media_tips) if media_tips else ""

                        if state.onboarding_status == OnboardingStatus.PAUSED.value:
                            reply += f"\n\nPD: Aún nos faltan algunos datos para completar tu perfil personalizado. ¿Te gustaría continuar donde nos quedamos? (Sí/No) 😊{tip_text}"
                        else:
                            reply += f"\n\nPD: Si quieres, también puedo personalizar mejor mis recomendaciones con un perfil nutricional rápido. ¿Te gustaría configurarlo ahora? (Sí/No) 😊{tip_text}"
                        
                        # Set to INVITED and RESET COUNTER
                        state.onboarding_status = OnboardingStatus.INVITED.value
                        state.onboarding_step = OnboardingStep.INVITACION.value
                        state.onboarding_last_invited_at = get_now_peru()
                        state.meaningful_interactions_count = 0 
                        state.version += 1


        if normalized.used_audio:
            await session.execute(text("UPDATE formulario_en_progreso SET uso_audio = TRUE WHERE usuario_id = :uid"), {"uid": user.id})
        if normalized.image_base64:
            await session.execute(text("UPDATE formulario_en_progreso SET uso_imagen = TRUE WHERE usuario_id = :uid"), {"uid": user.id})

        await session.execute(text("INSERT INTO extraction_jobs (usuario_id, raw_text) VALUES (:uid, :txt)"), {"uid": user.id, "txt": normalized.text})

        if onboarding_interception_happened:
            final_reply = reply
        else:
            original_mode = state.mode
            addon = await self._survey_service.process(session, state, normalized.text)
            
            if addon:
                if original_mode in ("collecting_usability", "collecting_profile") or state.mode in ("collecting_usability", "collecting_profile"):
                    final_reply = addon
                else:
                    final_reply = f"{reply}\n\n{addon}"
            else:
                final_reply = reply

        return final_reply, new_response_id
