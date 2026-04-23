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
from domain.parsers import parse_age, parse_weight, parse_height, standardize_text_list
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_read_service import ProfileReadService
from application.services.nutrition_assessment_service import NutritionAssessmentService
from application.services.conversation_state_service import ConversationStateService

logger = logging.getLogger(__name__)

ONBOARDING_QUESTIONS: dict[str, str] = {
    OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂\n",
    OnboardingStep.PESO.value: "¿Cuánto pesas aproximadamente en kilos? ⚖️\n",
    OnboardingStep.ALTURA.value: "¿Cuánto mides? 📐\nPuedes decirme en metros o centímetros.\nEj: 1.65 m, 170 cm...",
    OnboardingStep.TIPO_DIETA.value: (
        "¿Sigues algún tipo de alimentación en particular? 🥗\n"
        "Ejemplos:\n"
        "- *Omnívora*: comes de todo (carnes, verduras, etc.)\n"
        "- *Vegetariana*: no comes carnes pero sí huevo y lácteos\n"
        "- *Vegana*: no consumes ningún producto animal\n"
        "- *Ninguna en especial*\n"
        "Si no sigues ninguna dieta específica, dime 'ninguna' 😊"
    ),
    OnboardingStep.ALERGIAS.value: (
        "¿Tienes alguna alergia o intolerancia a alimentos? 🍎\n"
        "Ejemplos: alergia al maní, intolerancia a la lactosa, alergia a los mariscos...\n"
        "Si no tienes ninguna, dime 'ninguna'"
    ),
    OnboardingStep.ENFERMEDADES.value: (
        "¿Tienes alguna enfermedad o condición médica que deba tener en cuenta? 🏥\n"
        "Ejemplos: diabetes, hipertensión (presión alta), hipotiroidismo, anemia, gastritis...\n"
        "Esto me ayuda a darte recomendaciones más seguras para ti.\n"
        "Si no tienes ninguna, dime 'ninguna'"
    ),
    OnboardingStep.RESTRICCIONES.value: (
        "¿Hay alimentos que prefieras evitar o no puedas comer? 🚫\n"
        "Ejemplos: no como cerdo, evito los lácteos, no como mariscos...\n"
        "Si no tienes ninguna restricción, dime 'ninguna'"
    ),
    OnboardingStep.OBJETIVO.value: (
        "¿Cuál es tu objetivo principal con la alimentación? 🎯\n"
        "Ejemplos:\n"
        "- Bajar de peso\n"
        "- Ganar masa muscular\n"
        "- Mejorar mis hábitos alimenticios\n"
        "- Comer más saludable\n"
        "- Controlar mi diabetes/presión"
    ),
    OnboardingStep.PROVINCIA.value: "¿En qué provincia del Perú te encuentras? 😊\nEj: Lima, Arequipa, Cusco, Trujillo...",
    OnboardingStep.DISTRITO.value: "¿Y en qué distrito estás? 🏠\nEj: San Miguel, Miraflores, Cayma, Wanchaq..."
}

class OnboardingService:
    HEALTH_STEPS = {
        OnboardingStep.ALERGIAS.value,
        OnboardingStep.ENFERMEDADES.value,
        OnboardingStep.RESTRICCIONES.value,
    }
    HEALTH_FIELD_BY_STEP = {
        OnboardingStep.ALERGIAS.value: "alergias",
        OnboardingStep.ENFERMEDADES.value: "enfermedades",
        OnboardingStep.RESTRICCIONES.value: "restricciones_alimentarias",
    }
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
        OnboardingStep.EDAD.value: "tu edad",
        OnboardingStep.PESO.value: "tu peso",
        OnboardingStep.ALTURA.value: "tu talla (estatura)",
        OnboardingStep.ALERGIAS.value: "si tienes alguna alergia o restriccion",
        OnboardingStep.TIPO_DIETA.value: "si sigues algun tipo de dieta",
        OnboardingStep.ENFERMEDADES.value: "si padeces alguna condicion de salud",
        OnboardingStep.RESTRICCIONES.value: "si tienes alguna restriccion alimentaria",
        OnboardingStep.OBJETIVO.value: "tu objetivo nutricional",
        OnboardingStep.PROVINCIA.value: "la provincia donde te encuentras",
        OnboardingStep.DISTRITO.value: "tu distrito",
    }
    STEP_PURPOSES = {
        OnboardingStep.EDAD.value: "ajustar recomendaciones segun tu etapa de vida",
        OnboardingStep.PESO.value: "estimar porciones y energia diaria de forma mas precisa",
        OnboardingStep.ALTURA.value: "calcular tu IMC referencial",
        OnboardingStep.TIPO_DIETA.value: "sugerirte opciones que se adapten a tu estilo de alimentacion",
        OnboardingStep.ALERGIAS.value: "evitar alimentos que te puedan hacer dano",
        OnboardingStep.ENFERMEDADES.value: "adaptar recomendaciones con mayor seguridad",
        OnboardingStep.RESTRICCIONES.value: "respetar lo que prefieres evitar al comer",
        OnboardingStep.OBJETIVO.value: "enfocar la orientacion en tu meta principal",
        OnboardingStep.PROVINCIA.value: "compartirte orientacion y campanas de salud mas cercanas",
        OnboardingStep.DISTRITO.value: "compartirte orientacion y campanas de salud mas cercanas",
    }
    STEP_CLARIFICATIONS = {
        OnboardingStep.EDAD.value: "Me refiero a tu edad en anos cumplidos.",
        OnboardingStep.PESO.value: "Me refiero a tu peso aproximado actual.",
        OnboardingStep.ALTURA.value: "Me refiero a tu estatura (talla).",
        OnboardingStep.TIPO_DIETA.value: "Me refiero a tu estilo de alimentacion habitual.",
        OnboardingStep.ALERGIAS.value: "Me refiero a alergias o intolerancias a alimentos.",
        OnboardingStep.ENFERMEDADES.value: "Me refiero a condiciones de salud que influyen en tu alimentacion.",
        OnboardingStep.RESTRICCIONES.value: "Me refiero a alimentos que evitas por salud, preferencia o religion.",
        OnboardingStep.OBJETIVO.value: "Me refiero a tu meta principal con la alimentacion.",
        OnboardingStep.PROVINCIA.value: "Me refiero a la provincia donde vives actualmente.",
        OnboardingStep.DISTRITO.value: "Me refiero al distrito donde vives actualmente.",
    }
    STEP_EXAMPLES = {
        OnboardingStep.EDAD.value: "23",
        OnboardingStep.PESO.value: "68 kg",
        OnboardingStep.ALTURA.value: "1.70 m o 170 cm",
        OnboardingStep.TIPO_DIETA.value: "omnivora (comes de todo), vegetariana, vegana o ninguna",
        OnboardingStep.ALERGIAS.value: "mani, mariscos, lactosa o ninguna",
        OnboardingStep.ENFERMEDADES.value: "diabetes, hipertension, hipotiroidismo o ninguna",
        OnboardingStep.RESTRICCIONES.value: "no como cerdo, sin gluten o ninguna",
        OnboardingStep.OBJETIVO.value: "bajar peso, ganar masa muscular o mejorar habitos",
        OnboardingStep.PROVINCIA.value: "Lima",
        OnboardingStep.DISTRITO.value: "Miraflores",
    }
    CLARIFICATION_MARKERS = (
        "a que te refieres",
        "a que se refiere",
        "que significa",
        "que es",
        "como asi",
        "como asi?",
        "no entiendo",
        "explica",
        "ejemplo",
        "no me queda claro",
        "no me queda clara",
    )
    SOFT_REFUSAL_MARKERS = (
        "para que",
        "para qué",
        "para qu",
        "para q",
        "por que",
        "por qué",
        "porque",
        "xq",
        "pq",
        "no se para que",
        "no sé para qué",
    )
    HARD_REFUSAL_MARKERS = (
        "no quiero",
        "prefiero no",
        "no deseo",
        "no te dire",
        "no te diré",
        "no voy a",
        "no pienso",
        "no dare",
        "no daré",
    )
    INVITATION_ACCEPT_MARKERS = (
        "si",
        "sí",
        "ok",
        "okay",
        "claro",
        "dale",
        "vamos",
        "empez",
        "continu",
        "listo",
        "ya",
    )
    INVITATION_REJECT_MARKERS = (
        "no deseo",
        "prefiero no",
        "mas tarde",
        "más tarde",
        "luego",
        "otro dia",
        "otro día",
        "no ahora",
    )
    NUTRITION_REQUEST_MARKERS = (
        "menu",
        "menú",
        "receta",
        "comida",
        "desayuno",
        "almuerzo",
        "cena",
        "dieta",
    )

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
            r"^(yo\s+)?(soy\s+)?(intolerante|alergic[oa])\s+a\s+",
            r"^(mi\s+)?(intolerancia|alergia)\s+(es|son)?\s*(a\s+)?",
            r"^mi\s+enfermedad\s+(es|son)\s+",
            r"^(a\s*la|a\s*las|a\s*los|al|ala)\s+",
            r"^(es|son)\s+",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()
        return cleaned.strip(" .,!?:;")

    @classmethod
    def _is_health_step(cls, current_step: Optional[str]) -> bool:
        return bool(current_step and current_step in cls.HEALTH_STEPS)

    def _purpose_for_step(self, step: Optional[str]) -> str:
        if not step:
            return "personalizar mejor tus orientaciones"
        return self.STEP_PURPOSES.get(step, "personalizar mejor tus orientaciones")

    @classmethod
    def _is_clarification_request(cls, user_text: str) -> bool:
        txt = (user_text or "").strip().lower()
        if not txt:
            return False
        if txt.endswith("?"):
            return any(marker in txt for marker in cls.CLARIFICATION_MARKERS)
        return any(marker in txt for marker in cls.CLARIFICATION_MARKERS)

    def _build_step_clarification_reply(self, current_step: Optional[str]) -> str:
        if not current_step:
            return "Claro 😊 dime y te explico mejor."
            
        question = ONBOARDING_QUESTIONS.get(current_step, "")
        
        if current_step in (OnboardingStep.EDAD.value, OnboardingStep.PESO.value, OnboardingStep.ALTURA.value):
            return question
            
        purpose = self._purpose_for_step(current_step)
        clarification = self.STEP_CLARIFICATIONS.get(current_step, "Me refiero a ese dato de tu perfil.")
        example = self.STEP_EXAMPLES.get(current_step, "un valor simple")
        
        return (
            f"Claro 😊 {clarification}\n"
            f"Esto me ayuda a {purpose}.\n"
            f"Ejemplo: {example}.\n\n"
            f"{question}"
        ).strip()

    def _extract_numeric_step_fallback(self, current_step: Optional[str], user_text: str) -> dict:
        """Rescate deterministico para respuestas numericas cortas durante onboarding."""
        if not current_step:
            return {}
        raw = (user_text or "").strip()
        if not raw or "?" in raw:
            return {}

        if current_step == OnboardingStep.EDAD.value:
            age = parse_age(raw)
            if age is not None:
                return {"edad": str(age)}
            return {}

        if current_step == OnboardingStep.PESO.value:
            weight = parse_weight(raw)
            if weight is not None:
                return {"peso_kg": str(weight)}
            return {}

        if current_step == OnboardingStep.ALTURA.value:
            height = parse_height(raw)
            if height is not None:
                return {"altura_cm": str(height)}
            return {}

        return {}

    @classmethod
    def _is_invitation_reject(cls, user_text: str) -> bool:
        txt = (user_text or "").strip().lower()
        if not txt:
            return False
        if txt in {"no", "nop", "nah"}:
            return True
        return any(marker in txt for marker in cls.INVITATION_REJECT_MARKERS)

    @classmethod
    def _is_invitation_accept(cls, user_text: str) -> bool:
        txt = (user_text or "").strip().lower()
        if not txt:
            return False
        return any(re.search(rf"\b{re.escape(marker)}\b", txt) for marker in cls.INVITATION_ACCEPT_MARKERS)

    @classmethod
    def _is_nutrition_request(cls, user_text: str) -> bool:
        txt = (user_text or "").strip().lower()
        if not txt:
            return False
        return any(marker in txt for marker in cls.NUTRITION_REQUEST_MARKERS)

    def _extract_invitation_profile_data(self, user_text: str) -> dict:
        txt = (user_text or "").strip()
        if not txt:
            return {}
        txt_lower = txt.lower()
        if "?" in txt:
            return {}

        clean_data: dict[str, str] = {}

        age = parse_age(txt)
        if age is not None and (
            re.fullmatch(r"\d{1,3}", txt.strip()) is not None
            or any(marker in txt_lower for marker in ("edad", "años", "anos", "tengo", "cumpli", "cumplí"))
        ):
            clean_data["edad"] = str(age)

        weight = parse_weight(txt)
        if weight is not None and any(marker in txt_lower for marker in ("peso", "kg", "kilo", "quilo", "libras", "lb")):
            clean_data["peso_kg"] = str(weight)

        height = parse_height(txt)
        if height is not None and any(marker in txt_lower for marker in ("talla", "altura", "mido", "cm", "metro", "metros", " m ")):
            clean_data["altura_cm"] = str(height)

        if ("alerg" in txt_lower or "intoler" in txt_lower) and "alergias" not in clean_data:
            parsed = standardize_text_list(txt)
            if parsed:
                clean_data["alergias"] = parsed

        if "bajar peso" in txt_lower:
            clean_data["objetivo_nutricional"] = "BAJAR PESO"
        elif "ganar masa" in txt_lower:
            clean_data["objetivo_nutricional"] = "GANAR MASA MUSCULAR"
        elif "mejorar habitos" in txt_lower or "mejorar hábitos" in txt_lower:
            clean_data["objetivo_nutricional"] = "MEJORAR HABITOS"

        return clean_data

    @staticmethod
    def _onboarding_step_for_field(field_code: str) -> Optional[str]:
        mapping = {
            "edad": OnboardingStep.EDAD.value,
            "peso_kg": OnboardingStep.PESO.value,
            "altura_cm": OnboardingStep.ALTURA.value,
            "alergias": OnboardingStep.ALERGIAS.value,
            "objetivo_nutricional": OnboardingStep.OBJETIVO.value,
            "tipo_dieta": OnboardingStep.TIPO_DIETA.value,
            "enfermedades": OnboardingStep.ENFERMEDADES.value,
            "restricciones_alimentarias": OnboardingStep.RESTRICCIONES.value,
            "provincia": OnboardingStep.PROVINCIA.value,
            "distrito": OnboardingStep.DISTRITO.value,
        }
        return mapping.get(field_code)

    async def _handle_invitation_turn(
        self,
        *,
        user_text: str,
        state: ConversationState,
        session: AsyncSession,
    ) -> Optional[str]:
        txt = (user_text or "").strip()
        if not txt:
            return None

        inferred_data = self._extract_invitation_profile_data(txt)
        if self._is_invitation_reject(txt):
            self._set_onboarding_state(
                state,
                OnboardingStatus.SKIPPED,
                None,
                skip_count=state.onboarding_skip_count + 1,
            )
            return (
                "Entendido 😊 seguimos conversando normal.\n\n"
                "Cuando quieras personalizar tu perfil, solo me avisas."
            )

        if self._is_nutrition_request(txt) and not inferred_data and not self._is_invitation_accept(txt):
            self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
            return None

        if inferred_data:
            first_field = next(iter(inferred_data.keys()))
            await self._profile_extractor.save_clean_data(
                state.usuario_id,
                inferred_data,
                session,
                source_text=txt,
                current_step=self._onboarding_step_for_field(first_field),
            )

        accepted = bool(inferred_data) or self._is_invitation_accept(txt)
        if not accepted:
            return (
                "Estoy aqui para ayudarte 😊\n\n"
                "Preguntame lo que necesites sobre alimentacion o nutricion 🍎"
            )

        next_step = await self._find_next_missing_step(
            session,
            state.usuario_id,
            phase=ONBOARDING_PHASE_1,
        )
        if next_step is None:
            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
            return "Ya tengo tu perfil basico completo 😊 ¿En que te ayudo hoy?"

        self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
        if inferred_data:
            return f"Perfecto, ya registre ese dato.\n\n{ONBOARDING_QUESTIONS[next_step]}"
        return f"Genial 😊 vamos paso a paso con una sola pregunta.\n\n{ONBOARDING_QUESTIONS[next_step]}"

    def _looks_like_valid_health_negative(self, step: Optional[str], user_text: str) -> bool:
        if not self._is_health_step(step):
            return False
        txt = (user_text or "").strip().lower()
        valid_markers = (
            "ninguna",
            "ninguno",
            "ninguna alerg",
            "ninguna intoler",
            "no tengo",
            "sin alerg",
            "sin intoler",
            "nada",
        )
        return any(marker in txt for marker in valid_markers)

    def _classify_data_refusal(
        self,
        *,
        current_step: Optional[str],
        user_text: str,
        is_food_request: bool,
        history: Optional[list[dict]] = None,
    ) -> str:
        if not current_step or is_food_request:
            return "NONE"
        txt = (user_text or "").strip().lower()
        if not txt or self._looks_like_valid_health_negative(current_step, txt):
            return "NONE"
        if txt in {"no", "nop", "nah"}:
            return "SOFT_EXPLAIN"

        has_soft = any(marker in txt for marker in self.SOFT_REFUSAL_MARKERS)
        has_hard = any(marker in txt for marker in self.HARD_REFUSAL_MARKERS)
        if not has_soft and "?" in txt and ("porque" in txt or "para q" in txt or "para qu" in txt):
            has_soft = True
        if has_hard and has_soft:
            return "SOFT_EXPLAIN"
        if has_hard:
            if history and self._check_frustration(history, current_step):
                return "HARD_SKIP"
            return "SOFT_EXPLAIN"
        if has_soft:
            return "SOFT_EXPLAIN"
        return "NONE"

    async def _skip_current_step_and_advance(
        self,
        *,
        session: AsyncSession,
        state: ConversationState,
        current_step: str,
    ) -> str:
        await self._mark_field_as_skipped(session, state.usuario_id, current_step)
        next_step = await self._find_next_missing_step(
            session,
            state.usuario_id,
            phase=ONBOARDING_PHASE_1,
        )
        if next_step:
            self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
            return (
                "No hay problema 😊 lo dejamos como opcional.\n\n"
                f"{ONBOARDING_QUESTIONS[next_step]}"
            )
        self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
        return await self._build_phase1_completion_message(session, state.usuario_id)

    async def _build_phase1_completion_message(self, session: AsyncSession, usuario_id: int) -> str:
        completion_msg = "Listo 😊 ya tengo lo básico de tu perfil."
        try:
            p_final_data = await self._get_profile_flat(session, usuario_id)
            bmi_msg = self._nutrition_assessment.build_referential_message_from_flat(p_final_data)
            if bmi_msg:
                completion_msg += f"\n\n{bmi_msg}"
        except Exception as e:
            logger.warning("No se pudo calcular IMC al completar Phase 1: %s", e)
        completion_msg += (
            "\n\nSi quieres, luego completamos más datos poquito a poco para personalizar aún más tus orientaciones 🍏."
        )
        return completion_msg

    def _can_try_health_rescue(self, current_step: Optional[str], user_text: str, is_food_request: bool) -> bool:
        if not self._is_health_step(current_step):
            return False
        text = (user_text or "").strip().lower()
        if not text or "?" in text or is_food_request:
            return False
        if any(marker in text for marker in self.HEALTH_FALLBACK_SKIP_MARKERS):
            return False
        question_like_markers = (
            "que es",
            "qué es",
            "por que",
            "por qué",
            "como hago",
            "cómo hago",
            "explic",
        )
        return not any(marker in text for marker in question_like_markers)

    async def _extract_health_fallback(
        self,
        current_step: str,
        user_text: str,
        session: AsyncSession,
        usuario_id: int,
        is_food_request: bool,
    ) -> dict:
        if not self._can_try_health_rescue(current_step, user_text, is_food_request):
            return {}

        candidate_text = self._clean_health_fallback_text(user_text)
        candidate = standardize_text_list(candidate_text)
        candidate_upper = candidate.strip().upper() if candidate else ""
        if (
            not candidate
            or candidate_upper in self.HEALTH_FALLBACK_INVALID_VALUES
            or self._profile_extractor.contains_absurd_claim(candidate)
        ):
            return {}

        field_code = self.HEALTH_FIELD_BY_STEP.get(current_step)
        if not field_code:
            return {}

        extracted = {field_code: candidate}
        await self._profile_extractor.save_clean_data(
            usuario_id,
            extracted,
            session,
            source_text=user_text,
            current_step=current_step,
        )
        logger.info(
            "Onboarding fallback: user=%s, step=%s, value=%s",
            usuario_id,
            current_step,
            candidate,
        )
        return extracted

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

        # La invitacion inicial se resuelve de forma deterministica para evitar
        # repreguntas y aceptar datos directos (ej: "tengo 23 anos").
        if current_step == OnboardingStep.INVITACION.value:
            return await self._handle_invitation_turn(
                user_text=user_text,
                state=state,
                session=session,
            )

        # --- NEW Switchboard Logic (The Unified Brain) ---
        analysis = await self._analyze_turn(user_text, current_step, history)
        intent = analysis["intent"]
        numeric_fallback = self._extract_numeric_step_fallback(current_step, user_text)
        if numeric_fallback and (intent != "ANSWER" or not analysis.get("data")):
            intent = "ANSWER"
            analysis["data"] = numeric_fallback

        if intent == "RESET" or user_text.strip() == "/reset":
            await self._handle_system_reset(state.usuario_id, session)
            self._set_onboarding_state(state, OnboardingStatus.INVITED, OnboardingStep.INVITACION.value)
            return "¡Entendido! He borrado tus datos de perfil para que podamos empezar de cero cuando gustes. 🔄\n\n¿Quieres que empecemos ahora?"

        if current_step and current_step != OnboardingStep.INVITACION.value:
            # --- MODO OBSTINADO (Prioritarios = Phase 1 fields) ---
            PRIORITARY_STEPS = [s.value for s in ONBOARDING_PHASE_1 if s != OnboardingStep.INVITACION]
            is_food_request = any(w in vl for w in ["menu", "menú", "receta", "dieta", "comida", "desayuno", "almuerzo", "cena"])
            clarification_request = self._is_clarification_request(user_text)
            refusal_kind = self._classify_data_refusal(
                current_step=current_step,
                user_text=user_text,
                is_food_request=is_food_request,
                history=history,
            )
            if refusal_kind == "HARD_SKIP":
                return await self._skip_current_step_and_advance(
                    session=session,
                    state=state,
                    current_step=current_step,
                )
            if refusal_kind == "SOFT_EXPLAIN":
                purpose = self._purpose_for_step(current_step)
                return (
                    f"Buena pregunta 😊 te lo pido para {purpose}.\n\n"
                    f"{self._build_step_clarification_reply(current_step)}"
                )
            
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
                if not self._can_try_health_rescue(current_step, user_text, is_food_request):
                    if clarification_request:
                        return self._build_step_clarification_reply(current_step)
                    missing_label = self.FIELD_LABELS.get(current_step, current_step)
                    purpose = self._purpose_for_step(current_step)
                    return (
                        "Te ayudo con eso 🍏\n\n"
                        f"Para afinar la recomendacion, me ayuda confirmar {missing_label} para {purpose}.\n\n"
                        f"{ONBOARDING_QUESTIONS.get(current_step, '')}"
                    )
                intent = "ANSWER"

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
                        return f"No hay problema 😊\n\nSigamos con este dato:\n\n{ONBOARDING_QUESTIONS[next_step]}"

                if explanation:
                    if clarification_request:
                        return self._build_step_clarification_reply(current_step)
                    return f"{explanation}\n\nSigamos con este dato:\n\n{ONBOARDING_QUESTIONS.get(current_step, '')}"

                if is_food_request:
                    campo_lindo = self.FIELD_LABELS.get(current_step, f"tu {current_step}")
                    return (
                        f"Para darte una recomendacion mas precisa, primero confirmame {campo_lindo}.\n\n"
                        f"{ONBOARDING_QUESTIONS.get(current_step, '')}"
                    ).strip()

                return None
            
            if intent == "SKIP":
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
            
            if intent == "GREETING":
                return None

            if intent == "STOP":
                self._set_onboarding_state(state, OnboardingStatus.PAUSED, current_step)
                return "De acuerdo, pausamos aquí. Si quieres seguir más tarde, solo dime 'continuar'. 👋"

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

        # Fallback inteligente para campos de salud (alergias/enfermedades/restricciones)
        if (
            intent == "ANSWER"
            and not extracted
            and self._is_health_step(current_step)
        ):
            extracted = await self._extract_health_fallback(
                current_step=current_step,
                user_text=user_text,
                session=session,
                usuario_id=state.usuario_id,
                is_food_request=is_food_request if "is_food_request" in locals() else False,
            )

        if not extracted:
            if intent == "ANSWER":
                if self._check_frustration(history, current_step):
                    return (
                        "Tranqui, lo hacemos simple 😊\n\n"
                        f"Si quieres, responde solo esto:\n\n{ONBOARDING_QUESTIONS.get(current_step, '')}"
                    )

                return (
                    "No logre captar ese dato 😅\n\n"
                    f"Me lo repites asi de simple:\n\n{ONBOARDING_QUESTIONS.get(current_step, '')}"
                )
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
        was_current_skipped = intent == "SKIP" or (
            not extracted
            and any(w in vl.split() or vl.startswith(w) for w in ["saltar", "paso", "omitir", "luego", "siguiente"])
        )
        
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
                return f"{transition}\n\n{ONBOARDING_QUESTIONS[next_step]}"
            
            if next_step == current_step:
                return f"{ONBOARDING_QUESTIONS[current_step]}"

            return f"{ONBOARDING_QUESTIONS[next_step]}"
        else:
            # Phase 1 completada → dar valor inmediato y pasar a active_chat
            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
            return await self._build_phase1_completion_message(session, state.usuario_id)

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
