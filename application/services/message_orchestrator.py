"""
Nutribot Backend — MessageOrchestratorService
Orquesta el flujo principal de los mensajes entrantes, coordinando
los servicios de Onboarding, Survey, Extracción y el LLM.
"""
import logging
from typing import Optional
from openai import AsyncOpenAI
from sqlalchemy import text, JSON
from sqlalchemy.ext.asyncio import AsyncSession
import json

from domain.entities import ConversationState, NormalizedMessage, User
from domain.value_objects import OnboardingStatus, OnboardingStep, SessionMode
from domain.utils import get_now_peru
from application.services.onboarding_service import OnboardingService, ONBOARDING_QUESTIONS
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

    async def _get_recent_history(self, session: AsyncSession, uid: int) -> list[dict]:
        """Recupera los últimos 12 mensajes del historial en formato JSONB."""
        try:
            res = await session.execute(
                text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid"),
                {"uid": uid}
            )
            val = res.scalar()
            return val if isinstance(val, list) else []
        except Exception as e:
            logger.error("Error recuperando historial para usuario %s: %s", uid, e)
            return []

    async def _append_to_chat_memory(self, session: AsyncSession, uid: int, user_text: str, assistant_reply: str):
        """Añade la interacción actual al historial y mantiene un máximo de 20 mensajes."""
        try:
            # ON CONFLICT garante que el registro exista, pero UserRepo ya debería cubrirlo
            sql = """
                INSERT INTO memoria_chat (usuario_id, historial_mensajes, actualizado_en)
                VALUES (:uid, :init, NOW())
                ON CONFLICT (usuario_id) DO UPDATE 
                SET historial_mensajes = (
                    CASE 
                        WHEN jsonb_array_length(memoria_chat.historial_mensajes) >= 20 
                        THEN (memoria_chat.historial_mensajes - 0 - 1) 
                        ELSE memoria_chat.historial_mensajes 
                    END
                ) || :new_pair::jsonb,
                actualizado_en = NOW();
            """
            # Simplificamos la lógica de truncado en Python para mayor control si es necesario, 
            # pero por ahora lo hacemos atómico en SQL.
            # Nota: -0 -1 quita los dos primeros (par viejo) si superamos el límite.
            new_pair = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_reply}
            ]
            
            # Recuperamos actual, añadimos, truncamos y guardamos (más seguro que lógica compleja SQL)
            res = await session.execute(text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid"), {"uid": uid})
            hist = res.scalar() or []
            hist.extend(new_pair)
            hist = hist[-20:] # Mantener 10 pares máximo
            
            await session.execute(
                text("UPDATE memoria_chat SET historial_mensajes = :hist, actualizado_en = NOW() WHERE usuario_id = :uid"),
                {"uid": uid, "hist": json.dumps(hist)}
            )
        except Exception as e:
            logger.error("Error actualizando memoria_chat para usuario %s: %s", uid, e)

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
        history = await self._get_recent_history(session, user.id)
        
        reply = None
        v_text = normalized.text.lower().strip()
        mode_before_survey = state.mode
        
        # 0. COMANDOS DE SISTEMA (Global)
        if v_text == "/reset":
            await self._onboarding_service._handle_system_reset(user.id, session)
            state.onboarding_status = OnboardingStatus.INVITED.value
            state.onboarding_step = OnboardingStep.INVITACION.value
            state.mode = SessionMode.ACTIVE_CHAT.value
            state.version += 1
            return "¡Entendido! He borrado tus datos de perfil para que podamos empezar de cero cuando gustes. 🔄\n\n¿Quieres que empecemos ahora?", None

        is_asking_for_recommendation = any(w in v_text for w in ["menu", "menú", "receta", "dieta", "qué como", "que como", "comida saludable", "recomienda", "recomendación", "almuerzo", "cena", "desayuno", "coman", "nutricional", "comer", "imc", "calorías", "grasa", "proteína", "keto", "ayuno", "carbohidratos"])
        is_short_greeting = len(v_text) < 25 and any(w in v_text for w in ["hola", "buenas", "buenos", "empezar", "arrancar", "nutribot", "que tal", "holis"])
        
        is_requesting_personalization = any(w in v_text for w in ["personalizar", "completar mi perfil", "mis datos", "cambiar mi peso", "actualizar perfil", "personaliza", "mejorar", "ayudarte a mejorar", "encuesta", "formulario", "llenar datos", "mis objetivos"])
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
        
        # 1. TRAMO DE REGISTRO (ONBOARDING)
        # Si el usuario está en proceso de registro, le damos prioridad absoluta
        # para manejar saludos, dudas o datos del paso actual.
        # 1. TRAMO DE REGISTRO (ONBOARDING)
        # El Switchboard (Cerebro) es ahora la autoridad única.
        if state.onboarding_status in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            onboarding_interception_happened = True
            reply = await self._onboarding_service.advance_flow(
                normalized.text, state, session, 
                treat_ninguna_as_missing=False, 
                pre_extracted_data=None,
                history=history
            )
            if reply is None:
                onboarding_interception_happened = False

        # 2. EXTRACCIÓN GENERAL (Eliminado Legacy - Ahora todo pasa por Switchboard)

        if not onboarding_interception_happened and is_requesting_personalization:
            next_step = await self._onboarding_service._find_next_missing_step(session, user.id, ignore_skips=True, treat_ninguna_as_missing=False)
            if next_step:
                reply = f"¡Claro! 🥗 Me encantaría que personalicemos tus recomendaciones. Vamos a completar tu perfil para darte consejos exactos.\n\nEsto es lo que tengo registrado:\n{summary}\n\n¿Empezamos por confirmar tu **{next_step}**? 😊"
                state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
                state.onboarding_step = next_step
                onboarding_interception_happened = True
            else:
                reply = f"¡Ya tengo tu perfil completo! 😊\n\n{summary}\n\nSi quieres cambiar algún dato específico (como tu peso o talla), solo dímelo directamente en cualquier momento."

        if not onboarding_interception_happened and state.onboarding_status != OnboardingStatus.COMPLETED.value and (is_short_greeting or is_asking_for_recommendation):
            skipped = p_map.get("skipped_fields", {}) if isinstance(p_map.get("skipped_fields"), dict) else {}
            missing_essential = []
            if not p_map.get("edad") and not skipped.get("edad"): missing_essential.append("edad")
            if not p_map.get("peso_kg") and not skipped.get("peso_kg"): missing_essential.append("peso_kg")
            if not p_map.get("altura_cm") and not skipped.get("altura_cm"): missing_essential.append("altura_cm")
            if not p_map.get("alergias") and not skipped.get("alergias"): missing_essential.append("alergias")

            if is_asking_for_recommendation:
                if not missing_essential:
                    onboarding_interception_happened = False
                else:
                    intro = "¡Claro! 🥗 Me encantaría darte una recomendación a tu medida."
                    known_parts = []
                    if p_map.get("edad"): known_parts.append(f"Edad: {p_map['edad']} años")
                    if p_map.get("peso_kg"): known_parts.append(f"Peso: {p_map['peso_kg']}kg")
                    if p_map.get("altura_cm"): known_parts.append(f"Talla: {p_map['altura_cm']}cm")
                    if p_map.get("alergias"): known_parts.append(f"Alergias: {p_map['alergias']}")
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
                    citation = "Considerando"
                    if p_map:
                        citation += f" que tienes {p_map.get('edad')} años, pesas {p_map.get('peso_kg')}kg y mides {p_map.get('altura_cm')}cm"
                        if p_map.get('alergias') and p_map.get('alergias').upper() != 'NINGUNA':
                            citation += f", tienes alergia a {p_map.get('alergias')}"
                        if p_map.get('objetivo_nutricional') and p_map.get('objetivo_nutricional').upper() != 'NINGUNA':
                            citation += f" y tu objetivo es {p_map.get('objetivo_nutricional')}"
                    else:
                        citation += " tus datos actuales"
                    citation += ":"

                    extra_instr += """
                    [REGLA DE PERSONALIZACIÓN]
                    PRIORIDAD DE PETICIÓN: Si el usuario pide explícitamente algo que está en sus restricciones o alergias (ej: pide receta de pescado teniendo restricción de pescado), CUMPLE con la petición pero adviértele brevemente sobre su restricción registrada. Su deseo actual manda sobre su perfil previo.
                    """

                    final_profile_context = f"""[INSTRUCCIÓN CRÍTICA DE FORMATO]
Tu respuesta DEBE comenzar OBLIGATORIAMENTE con el siguiente texto exacto (no agregues 'Hola' antes de esto):
"{citation}"

[DATOS DE PERFIL PARA TU ANÁLISIS INTERNO]
{profile_text}"""

            # Recalcular por si se agregaron reglas a extra_instr dentro del bloque de recomendación.
            final_instructions = self._system_instructions + extra_instr
            reply, new_response_id = await self._llm_service.generate_reply(
                state=state_snapshot,
                normalized=normalized,
                instructions=final_instructions,
                rag_context=rag_text,
                profile_context=final_profile_context,
                history=history
            )

        # 1. CONTADOR UNIVERSAL DE INTERACCIONES
        # Incrementamos si el bot respondió algo fuera del registro puro.
        base_should_count_interaction = bool(not onboarding_interception_happened and reply)
        should_count_before_survey = bool(
            base_should_count_interaction
            and mode_before_survey not in (SessionMode.COLLECTING_USABILITY.value, SessionMode.COLLECTING_PROFILE.value)
        )
        projected_interactions_count = state.meaningful_interactions_count + (1 if should_count_before_survey else 0)

        # 2. DISPARADOR DE INVITACIÓN 
        # Universal Nag Nuclear: Sale SI O SI si el contador es >= 5
        if False and state.meaningful_interactions_count >= 5:
            inv_text = ""
            try:
                # PRIORIDAD 1: Encuesta de Usabilidad (si falta)
                if state.usability_completion_pct < 100:
                    res_m = await session.execute(text("SELECT uso_audio, uso_imagen, estado_actual FROM formulario_en_progreso WHERE usuario_id = :uid"), {"uid": user.id})
                    prog = res_m.fetchone()
                    media_tips = []
                    if prog:
                        if not prog.uso_audio: media_tips.append("🎙️ ¿Sabías que también puedes hablarme por **audio**? Es muy cómodo.")
                        if not prog.uso_imagen: media_tips.append("📸 ¡Prueba enviarme **fotos** de tus platos! Me ayuda a ser más visual.")
                    
                    tip_text = ("\n" + "\n".join(media_tips)) if media_tips else ""
                    
                    if prog and prog.estado_actual not in ["completado", "esperando_correo"]:
                        inv_text = f"PD: ¡Hola de nuevo! 😊 Nos quedaron unas preguntitas pendientes para mejorar NutriBot, ¿te animas a completarlas ahora? 🙏{tip_text}"
                    else:
                        inv_text = f"PD: ¡Muchas gracias por tus consultas! 😊 Ayúdame a mejorar respondiendo algunas preguntas y así empiezas oficialmente tu camino con NutriBot. 🙏{tip_text}"
                
                # PRIORIDAD 2: Completar Perfil (si no se hizo lo anterior y hay perfil pendiente)
                elif state.onboarding_status != OnboardingStatus.COMPLETED.value:
                    if state.onboarding_status == OnboardingStatus.IN_PROGRESS.value:
                        inv_text = "PD: Por cierto, veo que tienes tu perfil a medias. 😊 ¿Te gustaría terminar de configurarlo ahora para que mis recetas sean exactas?"
                    elif state.onboarding_status == OnboardingStatus.PAUSED.value:
                        inv_text = "PD: Aún nos faltan algunos datos para completar tu perfil personalizado. ¿Te gustaría continuar donde nos quedamos? (Sí/No) 😊"
            except Exception as e:
                # Fallback genérico si la DB falla
                inv_text = "PD: ¡Muchas gracias por tus consultas! 😊 Ayúdame a mejorar respondiendo algunas preguntas breves sobre tu experiencia. 🙏"
            
            if inv_text:
                # Aplicar separador visual para simular "segundo mensaje"
                reply += f"\n\n---\n\n{inv_text}"
                
                # Resetear contador para permitir recurrencia de la invitación
                state.meaningful_interactions_count = 0
                state.onboarding_last_invited_at = get_now_peru()
                
                if state.onboarding_status not in [OnboardingStatus.IN_PROGRESS.value, OnboardingStatus.COMPLETED.value]:
                    state.onboarding_status = OnboardingStatus.INVITED.value
                    state.onboarding_step = OnboardingStep.INVITACION.value
                state.version += 1


        if normalized.used_audio:
            await session.execute(text("UPDATE formulario_en_progreso SET uso_audio = TRUE WHERE usuario_id = :uid"), {"uid": user.id})
        if normalized.image_base64:
            await session.execute(text("UPDATE formulario_en_progreso SET uso_imagen = TRUE WHERE usuario_id = :uid"), {"uid": user.id})

        # --- ANCLA DE CONTINUIDAD (FINAL) ---
        if reply and not onboarding_interception_happened:
            # Añadir recordatorio educativo de personalización si no es el PD de encuesta
            if "PD:" not in reply:
                reply += "\n\n💡 *Tip NutriBot:* Recuerda que puedes decirme tus alergias, restricciones (ej. 'no como carne') o enfermedades en cualquier momento para ser 100% exacto para ti. ¡Tú tienes el control! 🍏"

            if state.onboarding_status == OnboardingStatus.IN_PROGRESS.value and state.onboarding_step:
                q_text = ONBOARDING_QUESTIONS.get(state.onboarding_step, "")
                if q_text:
                    if not any(q_text[:20] in reply for q_text in ONBOARDING_QUESTIONS.values()): # Evitar duplicados
                         reply = f"{reply}\n\nPor cierto, sigamos con tu perfil: **{q_text}**"

        await session.execute(text("INSERT INTO extraction_jobs (usuario_id, raw_text) VALUES (:uid, :txt)"), {"uid": user.id, "txt": normalized.text})

        if onboarding_interception_happened:
            final_reply = reply
        else:
            original_mode = state.mode
            addon = await self._survey_service.process(
                session,
                state,
                normalized.text,
                projected_interactions_count=projected_interactions_count,
            )
            
            if addon:
                if original_mode in ("collecting_usability", "collecting_profile"):
                    final_reply = addon
                else:
                    final_reply = f"{reply}\n\n{addon}"
            else:
                final_reply = reply

            survey_was_interrupted = bool(
                original_mode in (SessionMode.COLLECTING_USABILITY.value, SessionMode.COLLECTING_PROFILE.value)
                and state.mode == SessionMode.ACTIVE_CHAT.value
                and addon is None
            )

            if (should_count_before_survey or survey_was_interrupted) and state.mode == SessionMode.ACTIVE_CHAT.value:
                if survey_was_interrupted:
                    state.meaningful_interactions_count += 1
                else:
                    state.meaningful_interactions_count = projected_interactions_count
                logger.info("Universal interaction counter for user %s: %s", user.id, state.meaningful_interactions_count)

        # Persistir en memoria_chat
        await self._append_to_chat_memory(session, user.id, normalized.text, final_reply)

        return final_reply, new_response_id
