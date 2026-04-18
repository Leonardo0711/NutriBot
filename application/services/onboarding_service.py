"""
Nutribot Backend — OnboardingService
Gestiona la secuencia inicial opt-in para recolectar el perfil del usuario utilizando OOP.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState
from domain.value_objects import OnboardingStatus, OnboardingStep, ONBOARDING_STEPS_ORDER, ONBOARDING_PHASE_1, ONBOARDING_PHASE_2
from domain.utils import get_now_peru
from domain.parsers import parse_weight, parse_height, standardize_text_list
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_read_service import ProfileReadService
from application.services.nutrition_assessment_service import NutritionAssessmentService
from application.services.conversation_state_service import ConversationStateService

logger = logging.getLogger(__name__)

ONBOARDING_QUESTIONS: dict[str, str] = {
    OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂",
    OnboardingStep.PESO.value: "¿Y cuál es tu peso aproximado en kilos? ⚖️",
    OnboardingStep.ALTURA.value: "¿Y cuál es tu estatura aproximada? Puedes usar centímetros o metros (Ej. 1.70m, 170cm) 📐",
    OnboardingStep.TIPO_DIETA.value: "¿Sigues algún **tipo de dieta** especial? 🥗 (Ej: *Omnívora* si comes de todo, *Vegetariana*, *Vegana*, *Keto*, *Sin gluten*, o ninguna en particular)",
    OnboardingStep.ALERGIAS.value: "¿Tienes alguna **alergia** o intolerancia alimentaria? (Ej. Maní, mariscos, lactosa...) 🍎",
    OnboardingStep.ENFERMEDADES.value: "¿Padeces alguna condición de salud relevante? 🏥 (Ej: *Diabetes*, *Hipertensión*, *Hipotiroidismo* o ninguna)",
    OnboardingStep.RESTRICCIONES.value: "¿Tienes alguna **restricción alimentaria** por religión, ética o gusto personal? 🚫 (Ej: No como cerdo, no como carnes rojas, no me gusta el brócoli...)",
    OnboardingStep.OBJETIVO.value: "¿Cuál es tu **objetivo principal** al usar Nutribot? 🎯 (Ej. *Bajar de peso*, *Ganar masa muscular*, *Controlar mi glucemia*, o simplemente *Comer más sano*)",
    OnboardingStep.PROVINCIA.value: "¿En qué **provincia** de Perú te encuentras actualmente? 😊",
    OnboardingStep.DISTRITO.value: "¿Y en qué **distrito** vives específicamente? 🏠 (Para darte recomendaciones locales)"
}

class OnboardingService:
    HEALTH_FALLBACK_SKIP_MARKERS = (
        "saltar",
        "paso",
        "omitir",
        "luego",
        "siguiente",
        "prefiero no",
        "por ahora no",
        "despues",
        "después",
        "otro tema",
    )

    HEALTH_FALLBACK_INVALID_VALUES = {
        "",
        "NO SE",
        "NO SÉ",
        "NO SABE",
        "N/A",
        "NA",
        "X",
        "POR AHORA",
        "LUEGO",
        "DESPUES",
        "DESPUÉS",
        "NO QUIERO DECIR",
        "NO ENTIENDO",
    }

    FIELD_LABELS = {
        OnboardingStep.EDAD.value: "tu **edad**",
        OnboardingStep.PESO.value: "tu **peso**",
        OnboardingStep.ALTURA.value: "tu **talla (estatura)**",
        OnboardingStep.ALERGIAS.value: "si tienes alguna **alergia o restriccion**",
        OnboardingStep.TIPO_DIETA.value: "si sigues algun **tipo de dieta**",
        OnboardingStep.ENFERMEDADES.value: "si padeces alguna **condicion de salud**",
        OnboardingStep.RESTRICCIONES.value: "si tienes alguna **restriccion alimentaria**",
        OnboardingStep.OBJETIVO.value: "tu **objetivo nutricional**",
        OnboardingStep.PROVINCIA.value: "la **provincia** donde te encuentras",
        OnboardingStep.DISTRITO.value: "tu **distrito**",
    }

    @staticmethod
    def _clean_health_fallback_text(user_text: str) -> str:
        text = (user_text or "").strip().lower()
        if not text:
            return ""
        patterns = [
            r"^ya\s+te\s+dije\s+que\s+",
            r"^te\s+dije\s+que\s+",
            r"^que\s+no\s+entiendes\s+(de\s+)?",
            r"^(yo\s+)?(tengo|padezco|sufro\s+de|presento|me\s+diagnosticaron)\s+",
            r"^mi\s+enfermedad\s+(es|son)\s+",
            r"^(es|son)\s+",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()
        return cleaned.strip(" .,!?:;")

    SWITCHBOARD_SYSTEM_PROMPT = """Eres el Cerebro de Nutribot (Switchboard), encargado de clasificar la intención del usuario durante el registro de perfil.

REGLAS DE INTENCIÓN:
1. ANSWER: El usuario provee el dato solicitado (ej: '30 años', '80kg', 'no tengo alergias').
2. DOUBT: El usuario está confundido o hace una petición de chat/nutrición EN LUGAR de responder (ej: 'dame un menú', '¿para qué sirve?').
3. SKIP: El usuario pide saltar la pregunta.
4. GREETING: Saludo inicial.
5. RESET/STOP: Comandos de sistema.

REGLAS DE ORO (CRÍTICAS):
- PETICIONES DE COMIDA = DOUBT: Si el usuario pide un menú, receta o consejo (ej: 'Dame un menú marino', 'Dame dieta para bajar peso') MIENTRAS estás en un paso de perfil, clasifica SIEMPRE como DOUBT. NUNCA lo extraigas como 'Ninguna' o como dato de perfil.
- BLINDAJE DE ALERGIAS: Si el paso es ALERGIAS y el usuario pide comida, NUNCA devuelvas 'Ninguna'. Es preferible clasificar como DOUBT y pedir aclaración.

- COHERENCIA MÉDICA Y BIOLÓGICA (NUEVO):
  * RECHAZA datos absurdos (ej: 'alergia al aire', 'enfermedad de los marcianos').
  * RECHAZA métricas imposibles (ej: un adulto de 300cm, un bebé de 200kg, o una persona de 2 metros que pese 20kg).
  * Si el dato no tiene sentido biológico o médico, clasifica como DOUBT y pide aclaración en 'explanation'.

- Si tienes dudas entre ANSWER y DOUBT por una petición de comida o incoherencia, elige DOUBT.

EXAMPLES:
- Paso: PESO | Usuario: 'Peso 500 kilos' -> Intent: DOUBT, Data: {}, Explanation: '¡Wow! 😮 Quizás hubo un error al escribir. ¿Me confirmas tu peso real para calcular bien tu plan?'
- Paso: PESO | Usuario: '80kg y dame un menú' -> Intent: DOUBT, Data: {}, Explanation: '¡Perfecto! 📝 Ya casi llegamos al menú, solo confírmame primero el peso para que sea exacto.'
- Paso: PROVINCIA | Usuario: 'no deseo' -> Intent: SKIP, Data: {}, Explanation: 'Ningún problema, podemos seguir sin eso.'

REGLAS DE TONO Y FLEXIBILIDAD:
- EVITA FRASES BLOQUEANTES: Nunca digas "necesito esto para continuar" o "es obligatorio". Usa "Me ayudaría mucho a..." o "Para ser más preciso...".
- DETECCIÓN DE RECHAZO: Si el usuario dice "no quiero decirte", "ya me aburrí", "muchos datos", "no deseo", clasifica como SKIP.
- PETICIONES DE COMIDA DURANTE UBICACIÓN: Si estamos en PROVINCIA o DISTRITO y el usuario pide COMIDA/MENÚ, clasifica como SKIP para el campo actual para no bloquear.
- Si el usuario muestra FRUSTRACIÓN o ABURRIMIENTO, genera una 'explanation' muy breve y empática, y sugiere que pueden seguir con otra cosa.

FORMATO DE SALIDA (JSON):
{
  "intent": "ANSWER|DOUBT|SKIP|GREETING|RESET|STOP",
  "data": {"campo": "valor"} o {},
  "explanation": "Frase empática y breve (máx 12 palabras). Evita sonar repetitivo.",
  "confidence": 0.0-1.0
}
"""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        openai_model: str,
        profile_extractor: ProfileExtractionService,
        profile_reader: Optional[ProfileReadService] = None,
        nutrition_assessment: Optional[NutritionAssessmentService] = None,
        state_service: Optional[ConversationStateService] = None,
    ):
        self._openai_client = openai_client
        self._openai_model = openai_model
        self._profile_extractor = profile_extractor
        self._profile_reader = profile_reader or ProfileReadService()
        self._nutrition_assessment = nutrition_assessment or NutritionAssessmentService()
        self._state_service = state_service or ConversationStateService()

    def _validate_onboarding_field(self, step: str, raw_value: str) -> tuple[bool, Optional[str], Optional[str]]:
        v = raw_value.strip()
        vl = v.lower()
        skip_words = ["no", "saltar", "ninguno", "ninguna", "paso", "no se", "omitir", "despues"]
        is_skip = any(vl == w or vl.startswith(w + " ") for w in skip_words)

        if step == OnboardingStep.EDAD.value:
            try:
                age = int(re.sub(r"\D", "", v))
                if 5 <= age <= 120:
                    return True, str(age), None
            except ValueError:
                pass
            return False, None, "¿Podrías darme tu edad en números enteros? (Ej. 30)"
        
        elif step == OnboardingStep.PESO.value:
            if is_skip: return True, None, None
            w = parse_weight(v)
            if w:
                return True, str(w), None
            return False, None, "No logré captar el peso. ¿Podrías decirlo en kilos o libras? (O escribe 'saltar' si prefieres no decirlo aún)"
        
        elif step == OnboardingStep.ALTURA.value:
            if is_skip: return True, None, None
            h = parse_height(v)
            if h:
                return True, str(h), None
            return False, None, "No logré captar la estatura. ¿Podrías decirlo en centímetros o metros? (O escribe 'saltar' si prefieres no decirlo aún)"
        else:
            if len(v) > 200:
                return False, None, "¡Uy, es un poco largo! ¿Podrías resumirlo un poquito más, por favor?"
            return True, v, None

    def _set_onboarding_state(self, state: ConversationState, status: OnboardingStatus, step: Optional[str], **kwargs):
        old_status = state.onboarding_status
        if status == OnboardingStatus.INVITED:
            self._state_service.set_onboarding_invited(state)
        elif status == OnboardingStatus.IN_PROGRESS:
            if step:
                self._state_service.set_onboarding_in_progress(state, step)
        elif status == OnboardingStatus.COMPLETED:
            self._state_service.set_onboarding_completed(state)
        elif status == OnboardingStatus.SKIPPED:
            self._state_service.set_onboarding_skipped(state, days_until_retry=14)
            if "skip_count" in kwargs:
                state.onboarding_skip_count = kwargs["skip_count"]
        elif status == OnboardingStatus.PAUSED:
            self._state_service.set_onboarding_paused(state, days_until_retry=3)
        else:
            state.onboarding_status = status.value
            state.onboarding_step = step
            state.onboarding_updated_at = get_now_peru()
        
        logger.info(
            "Onboarding state change: user=%s, status=%s -> %s, step=%s",
            state.usuario_id, old_status, status.value, step
        )

    async def _get_profile_flat(self, session: AsyncSession, uid: int) -> dict:
        """Proyeccion compatible construida desde el modelo normalizado V3."""
        return await self._profile_reader.fetch_projection(session, uid)

    async def advance_flow(
        self,
        user_text: str,
        state: ConversationState,
        session: AsyncSession,
        treat_ninguna_as_missing: bool = False,
        pre_extracted_data: Optional[dict] = None,
        history: Optional[list[dict]] = None
    ) -> Optional[str]:
        if state.onboarding_status not in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            return None

        vl = user_text.lower().strip()
        current_step = state.onboarding_step

        # --- NEW Switchboard Logic (The Unified Brain) ---
        analysis = await self._analyze_turn(user_text, current_step, history)
        intent = analysis["intent"]
        
        if intent == "RESET" or user_text.strip() == "/reset":
            await self._handle_system_reset(state.usuario_id, session)
            self._set_onboarding_state(state, OnboardingStatus.INVITED, OnboardingStep.INVITACION.value)
            return "¡Entendido! He borrado tus datos de perfil para que podamos empezar de cero cuando gustes. 🔄\n\n¿Quieres que empecemos ahora?"

        if current_step and current_step != OnboardingStep.INVITACION.value:
            # --- MODO OBSTINADO (Prioritarios = Phase 1 fields) ---
            PRIORITARY_STEPS = [s.value for s in ONBOARDING_PHASE_1 if s != OnboardingStep.INVITACION]
            is_food_request = any(w in vl for w in ["menu", "menú", "receta", "dieta", "comida", "desayuno", "almuerzo", "cena"])
            
            # REGLA DE ORO PARA UBICACIÓN (Provincia/Distrito):
            # Si en esta etapa cambia de tema o no desea compartir ubicación, NO bloquear el chat.
            if current_step in (OnboardingStep.PROVINCIA.value, OnboardingStep.DISTRITO.value) and (
                intent in ("DOUBT", "SKIP") or is_food_request
            ):
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
                if current_step == OnboardingStep.PROVINCIA.value:
                    await self._mark_field_as_skipped(session, state.usuario_id, OnboardingStep.DISTRITO.value)

                next_step = await self._find_next_missing_step(session, state.usuario_id)
                if next_step and next_step not in (OnboardingStep.PROVINCIA.value, OnboardingStep.DISTRITO.value):
                    self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
                else:
                    self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
                return None
            
            # REGLA DE OBSTINACIÓN (Nutrición requiere Perfil):
            if intent == "DOUBT" and current_step in PRIORITARY_STEPS:
                missing_label = self.FIELD_LABELS.get(current_step, current_step)
                return (
                    f"Entiendo perfectamente tu duda y me encantaría ayudarte con eso ahora mismo. 🍏 "
                    f"Sin embargo, para poder darte una recomendación que sea **asertiva, segura y 100% a tu medida**, "
                    f"todavía necesito completar tu **{missing_label}**.\n\n"
                    f"¿Me lo podrías confirmar para seguir? 🙏\n\n"
                    f"**{ONBOARDING_QUESTIONS.get(current_step, '')}**"
                )

            if intent == "DOUBT":
                explanation = analysis.get("explanation")
                
                if current_step in (OnboardingStep.PROVINCIA.value, OnboardingStep.DISTRITO.value):
                    if self._check_frustration(history, current_step) or any(w in vl for w in ["aburr", "harto", "no quiero", "no deseo", "basta", "dame", "nada"]):
                        await self._mark_field_as_skipped(session, state.usuario_id, current_step)
                        next_step = await self._find_next_missing_step(session, state.usuario_id)
                        if not next_step:
                            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
                            return "¡Entendido! No te preocupes por eso. 😊 ¿En qué más puedo ayudarte hoy?"
                        
                        self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
                        if is_food_request or "dame" in vl:
                             return None
                        return f"No hay problema, podemos saltarlo. 😊 Sigamos con otro detalle: **{ONBOARDING_QUESTIONS[next_step]}**"

                if explanation:
                    return f"{explanation}\n\n¿Seguimos con tu perfil? **{ONBOARDING_QUESTIONS.get(current_step, '')}**"

                if is_food_request:
                    p = await self._get_profile_flat(session, state.usuario_id)
                    known_parts = []
                    if p.get("edad"):
                        known_parts.append(f"Edad: {p['edad']} anos")
                    if p.get("peso_kg"):
                        known_parts.append(f"Peso: {p['peso_kg']}kg")
                    if p.get("altura_cm"):
                        h = float(p["altura_cm"])
                        h_str = f"{h/100:.2f}m" if h > 10 else f"{h:.2f}m"
                        known_parts.append(f"Talla: {h_str}")
                    if p.get("alergias"):
                        known_parts.append(f"Alergias: {p['alergias']}")

                    known_line = f"Tengo registrado: {', '.join(known_parts)}. " if known_parts else ""
                    campo_lindo = self.FIELD_LABELS.get(current_step, f"tu **{current_step}**")
                    question = ONBOARDING_QUESTIONS.get(current_step, "")
                    return (
                        f"Vamos bien. {known_line}Para darte una recomendacion 100% personalizada, "
                        f"solo me falta confirmar {campo_lindo}. {question}"
                    ).strip()

                return None
            
            if intent == "SKIP":
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
            
            if intent == "GREETING":
                return None

            if intent == "STOP":
                self._set_onboarding_state(state, OnboardingStatus.PAUSED, current_step)
                return "De acuerdo, pausamos aquí. Si quieres seguir más tarde, solo dime 'continuar'. 👋"

        # --- INVITACIÓN ---
        if current_step == OnboardingStep.INVITACION.value:
            prompt = f"""Analiza la respuesta del usuario a una invitación de 'Personalizar perfil nutricional'.
            Responde SOLO con una palabra: ACCEPT, REJECT o OTHER.
            USUARIO: "{user_text}"
            """
            resp = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "system", "content": "Eres un clasificador de intenciones binarias."},
                          {"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            intent = resp.choices[0].message.content.strip().upper()
            
            if "REJECT" in intent:
                if len(user_text) > 15:
                    self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
                    return None
                
                self._set_onboarding_state(state, OnboardingStatus.SKIPPED, None, skip_count=state.onboarding_skip_count + 1)
                return "¡Entendido! Seguimos conversando libremente. Si alguna vez quieres personalizar tu perfil, solo dímelo. 😊 ¿En qué más puedo ayudarte hoy?"
            elif "OTHER" in intent:
                self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
                return None
            else:
                next_step = await self._find_next_missing_step(session, state.usuario_id)
                if next_step is None:
                    self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
                    return "¡Veo que ya tengo tu perfil nutricional completo! 😊 ¿En qué puedo ayudarte hoy?"

                self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
                intro_profile = (
                    "¡Genial! 😊 Solo necesito 5 datos rápidos "
                    "(edad, peso, talla, alergias y objetivo) para darte orientación personalizada. ¡Es un ratito! 🚀"
                )
                p = await self._get_profile_flat(session, state.usuario_id)
                known_parts = []
                if p.get("edad"): known_parts.append(f"Edad: {p['edad']} años")
                if p.get("peso_kg"): known_parts.append(f"Peso: {p['peso_kg']}kg")
                if p.get("altura_cm"):
                    h = float(p["altura_cm"])
                    h_str = f"{h/100:.2f}m" if h > 10 else f"{h:.2f}m"
                    known_parts.append(f"Talla: {h_str}")
                
                if known_parts:
                    return (
                        f"{intro_profile}\n\n"
                        f"Ya tengo registrado: **{', '.join(known_parts)}**. "
                        f"Ahora necesito completar unos datos más.\n\n{ONBOARDING_QUESTIONS[next_step]}"
                    )
                return f"{intro_profile}\n\n{ONBOARDING_QUESTIONS[next_step]}"

        # --- PROCESAMIENTO DE RESPUESTA ---
        current_step = state.onboarding_step
        if not current_step:
            return None

        meta_flags = {}
        if intent == "ANSWER" and analysis.get("data"):
            ext_result = await self._profile_extractor.apply_cleaning_and_save(
                raw_extractions=analysis["data"],
                user_text=user_text,
                usuario_id=state.usuario_id,
                session=session,
                current_step=current_step
            )
            extracted = ext_result.clean_data
            meta_flags = ext_result.meta_flags
        elif intent == "SKIP":
            extracted = {}
        else:
            extracted = {}

        # Fallback inteligente para enfermedades
        if (
            intent == "ANSWER"
            and not extracted
            and current_step == OnboardingStep.ENFERMEDADES.value
        ):
            has_question_shape = "?" in vl
            looks_like_skip = any(marker in vl for marker in self.HEALTH_FALLBACK_SKIP_MARKERS)
            if not has_question_shape and not looks_like_skip:
                candidate_text = self._clean_health_fallback_text(user_text)
                candidate = standardize_text_list(candidate_text)
                candidate_upper = candidate.strip().upper() if candidate else ""
                if (
                    candidate_upper not in self.HEALTH_FALLBACK_INVALID_VALUES
                    and not self._profile_extractor.contains_absurd_claim(candidate)
                ):
                    extracted = {"enfermedades": candidate}
                    await self._profile_extractor.save_clean_data(state.usuario_id, extracted, session)
                    logger.info("Onboarding fallback: user=%s, enfermedades=%s", state.usuario_id, candidate)

        if not extracted:
            if intent == "ANSWER":
                if self._check_frustration(history, current_step):
                    campo_lindo = self.FIELD_LABELS.get(current_step, current_step)
                    return f"Veo que este punto es algo confuso. 😅 Si prefieres, podemos **saltarlo** por ahora y seguir con lo demás para no estancarnos. ¿Te parece?\n\nO si gustas, dime tu **{campo_lindo}** para continuar."

                return f"No logré captar ese detalle para tu perfil. 😅 ¿Podrías decírmelo de forma más simple?\n\n**{ONBOARDING_QUESTIONS.get(current_step, '')}**"
            elif intent == "SKIP":
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
            else:
                pass

        if not extracted and intent == "SKIP":
             await self._mark_field_as_skipped(session, state.usuario_id, current_step)

        # Si hubo una extracción y requiere aclaración clínica, atajamos el flujo
        if extracted and meta_flags.get("needs_health_clarification"):
            state.onboarding_step = current_step
            return meta_flags.get("clarification_prompt", "¿Te importaría aclararlo un poco más para ser más precisos?")

        updated_cols = list(extracted.keys()) if extracted else []
        current_idx = -1
        for i, s in enumerate(ONBOARDING_STEPS_ORDER):
            if s.value == current_step:
                current_idx = i
                break

        was_current_answered = any(col in updated_cols for col in [current_step, "peso_kg" if current_step=="peso" else current_step])
        was_current_skipped = not extracted and any(w in vl.split() or vl.startswith(w) for w in ["saltar", "paso", "omitir", "luego", "siguiente"])
        
        search_start_idx = current_idx + 1 if (was_current_answered or was_current_skipped) else current_idx

        # PHASE 1: buscar solo en pasos de Phase 1 durante onboarding activo
        next_step = await self._find_next_missing_step(
            session,
            state.usuario_id,
            treat_ninguna_as_missing=treat_ninguna_as_missing,
            start_from_idx=search_start_idx,
            ignore_cols=updated_cols,
            phase=ONBOARDING_PHASE_1,
        )
        if next_step:
            self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
            if updated_cols:
                transition = "¡Perfecto! Ya anoté esos detalles. ✍️"
                if len(updated_cols) == 1 and updated_cols[0] == "region":
                    transition = "¡Qué bueno! Me encanta esa zona. 📍"
                return f"{transition} Ahora, para seguir personalizando tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
            
            if next_step == current_step:
                return f"Para poder ayudarte mejor, necesito este dato: **{ONBOARDING_QUESTIONS[current_step]}**"

            return f"Siguiendo con tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
        else:
            # Phase 1 completada → dar valor inmediato y pasar a active_chat
            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
            completion_msg = "Listo 😊 ya tengo lo básico de tu perfil."
            try:
                p_final_data = await self._get_profile_flat(session, state.usuario_id)
                bmi_msg = self._nutrition_assessment.build_referential_message_from_flat(p_final_data)
                if bmi_msg:
                    completion_msg += f"\n\n{bmi_msg}"
            except Exception as e:
                logger.warning("No se pudo calcular IMC al completar Phase 1: %s", e)
            completion_msg += (
                "\n\nSi quieres, luego completamos más datos poquito a poco para personalizar aún más tus orientaciones 🍏."
            )
            return completion_msg

    async def _find_next_missing_step(
        self,
        session: AsyncSession,
        uid: int,
        ignore_skips: bool = False,
        treat_ninguna_as_missing: bool = False,
        skip_step: Optional[str] = None,
        start_from_idx: Optional[int] = None,
        ignore_cols: Optional[list[str]] = None,
        phase: Optional[list] = None,
    ) -> Optional[str]:
        p = await self._get_profile_flat(session, uid)
        
        if not p:
            return OnboardingStep.EDAD.value

        skipped = p.get("skipped_fields", {})
        if not isinstance(skipped, dict):
            skipped = {}

        col_map = {
            OnboardingStep.EDAD.value: "edad",
            OnboardingStep.ALERGIAS.value: "alergias",
            OnboardingStep.ENFERMEDADES.value: "enfermedades",
            OnboardingStep.RESTRICCIONES.value: "restricciones_alimentarias",
            OnboardingStep.TIPO_DIETA.value: "tipo_dieta",
            OnboardingStep.OBJETIVO.value: "objetivo_nutricional",
            OnboardingStep.PESO.value: "peso_kg",
            OnboardingStep.ALTURA.value: "altura_cm",
            OnboardingStep.REGION.value: "region",
            OnboardingStep.PROVINCIA.value: "provincia",
            OnboardingStep.DISTRITO.value: "distrito"
        }

        steps_to_search = phase if phase is not None else ONBOARDING_PHASE_1
        if start_from_idx is not None:
            base_idx = start_from_idx
        else:
            starts_with_invitation = bool(
                steps_to_search
                and getattr(steps_to_search[0], "value", None) == OnboardingStep.INVITACION.value
            )
            base_idx = 1 if starts_with_invitation else 0
        for step in steps_to_search[base_idx:]:
            if skip_step and step.value == skip_step:
                continue
                
            col = col_map.get(step.value)
            if not col: continue

            if ignore_cols and col in ignore_cols:
                continue
            
            if not ignore_skips and skipped.get(step.value):
                continue

            val = p.get(col)
            is_empty = val is None or (isinstance(val, str) and len(val.strip()) == 0)
            
            if treat_ninguna_as_missing and isinstance(val, str) and val.upper() == "NINGUNA":
                is_empty = True

            if is_empty:
                return step.value

        return None

    async def _mark_field_as_skipped(self, session: AsyncSession, uid: int, field: str):
        sql_init = """
            INSERT INTO perfil_nutricional (usuario_id, actualizado_en)
            VALUES (:uid, :upd)
            ON CONFLICT (usuario_id) DO NOTHING
        """
        await session.execute(text(sql_init), {"uid": uid, "upd": get_now_peru()})

        sql_skip = f"""
            UPDATE perfil_nutricional 
            SET skipped_fields = skipped_fields || jsonb_build_object(CAST(:field AS text), true),
                actualizado_en = :upd
            WHERE usuario_id = :uid
        """
        await session.execute(text(sql_skip), {"uid": uid, "field": field, "upd": get_now_peru()})

    async def _analyze_turn(self, user_text: str, current_step: str, history: Optional[list[dict]]) -> dict:
        """
        The Switchboard: One LLM call to rule them all.
        Classifies intent and extracts raw data.
        """
        history_summary = ""
        if history:
            history_summary = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])

        current_q = ONBOARDING_QUESTIONS.get(current_step, "desconocida")
        
        prompt = f"""PASO ACTUAL: {current_step}
PREGUNTA: "{current_q}"

HISTORIAL:
{history_summary}

MENSAJE USUARIO: "{user_text}"
"""
        try:
            resp = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": self.SWITCHBOARD_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            import json
            analysis = json.loads(resp.choices[0].message.content)
            logger.info("Onboarding Switchboard: user=%s, intent=%s, data=%s", current_step, analysis.get("intent"), analysis.get("data"))
            return analysis
        except Exception as e:
            logger.error("Error in Switchboard: %s", e)
            return {"intent": "ANSWER", "data": {}, "explanation": None}

    async def _handle_system_reset(self, uid: int, session: AsyncSession):
        """Clean up user data and onboarding progress."""
        logger.info("System Reset triggered for user %s", uid)
        await session.execute(
            text(
                """
                DELETE FROM perfil_nutricional_medicion
                WHERE perfil_nutricional_id IN (
                    SELECT id FROM perfil_nutricional WHERE usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(
            text(
                """
                DELETE FROM perfil_nutricional_enfermedad
                WHERE perfil_nutricional_id IN (
                    SELECT id FROM perfil_nutricional WHERE usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(
            text(
                """
                DELETE FROM perfil_nutricional_restriccion
                WHERE perfil_nutricional_id IN (
                    SELECT id FROM perfil_nutricional WHERE usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(
            text(
                """
                DELETE FROM orden_dietetica_dieta
                WHERE orden_dietetica_id IN (
                    SELECT od.id
                    FROM orden_dietetica od
                    JOIN perfil_nutricional p ON p.id = od.perfil_nutricional_id
                    WHERE p.usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(
            text(
                """
                DELETE FROM orden_dietetica_restriccion
                WHERE orden_dietetica_id IN (
                    SELECT od.id
                    FROM orden_dietetica od
                    JOIN perfil_nutricional p ON p.id = od.perfil_nutricional_id
                    WHERE p.usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(
            text(
                """
                DELETE FROM orden_dietetica
                WHERE perfil_nutricional_id IN (
                    SELECT id FROM perfil_nutricional WHERE usuario_id = :uid
                )
                """
            ),
            {"uid": uid},
        )
        await session.execute(text("DELETE FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM memoria_chat WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM formulario_en_progreso WHERE usuario_id = :uid"), {"uid": uid})
        # Note: ConversationState is managed in the caller (advance_flow).
        # No commit here: this method runs inside an active transaction.

    def _check_frustration(self, history: Optional[list[dict]], current_step: str) -> bool:
        """
        Detecta si el usuario está estancado o mostrando signos de molestia.
        """
        if not history or len(history) < 2:
            return False
            
        last_user_msg = history[-1]["content"].lower() if history[-1]["role"] == "user" else ""
        frustration_keywords = ["aburr", "harto", "no quiero", "no deseo", "basta", "dame lo que", "muchos datos", "pesado", "stuck", "que fue", "no contestas"]
        if any(kw in last_user_msg for kw in frustration_keywords):
            return True

        # Regla de repetición (Assistant preguntó lo mismo 2 veces)
        assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
        if len(assistant_msgs) >= 2:
            last_q = assistant_msgs[-1].lower()
            prev_q = assistant_msgs[-2].lower()
            q_text = ONBOARDING_QUESTIONS.get(current_step, "").lower()
            if q_text and q_text in last_q and q_text in prev_q:
                return True
            
        return False
