"""
Nutribot Backend - LLM Reply Service
"""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
import re
import unicodedata
from typing import Optional

from application.services.localization_service import LocalizationService
from application.services.profile_context_service import ProfileContextService
from domain.context_builder import build_llm_context, try_fast_response
from domain.entities import ConversationState, NormalizedMessage
from domain.normalizer import fuzzy_match_any
from domain.ports import LLMService
from domain.profile_snapshot import ProfileSnapshot
from domain.reply_objects import BotReply
from domain.router import Intent, RouteResult

logger = logging.getLogger(__name__)


class LlmReplyService:
    _PROFILE_FIELD_KEYWORDS = {
        "edad": ("edad", "anos"),
        "peso_kg": ("peso", "kilo", "kg"),
        "altura_cm": ("talla", "estatura", "altura", "cm", "metro", "mides"),
        "alergias": ("alergia", "alergias", "intolerancia", "intolerancias"),
        "enfermedades": ("enfermedad", "enfermedades", "condicion de salud", "condiciones de salud"),
        "restricciones_alimentarias": ("restriccion", "restricciones", "evitas", "no comes"),
        "tipo_dieta": ("tipo de dieta", "dieta", "patron alimentario"),
        "objetivo_nutricional": ("objetivo", "meta"),
        "provincia": ("provincia",),
        "distrito": ("distrito",),
    }
    _CANONICAL_PROFILE_QUESTION = {
        "edad": "Para empezar, ¿cuántos años tienes? 🎂",
        "peso_kg": "¿Cuánto pesas aproximadamente en kilos? ⚖️",
        "altura_cm": "¿Cuánto mides? 📐\nPuedes decirme en metros o centímetros.\nEj: 1.65 m, 170 cm...",
        "alergias": "¿Tienes alguna alergia o intolerancia a alimentos? 🍎\nEjemplos: alergia al maní, intolerancia a la lactosa, alergia a los mariscos...\nSi no tienes ninguna, dime 'ninguna'",
        "enfermedades": "¿Tienes alguna enfermedad o condición médica que deba tener en cuenta? 🏥\nEjemplos: diabetes, hipertensión (presión alta), hipotiroidismo, anemia, gastritis...\nSi no tienes ninguna, dime 'ninguna'",
        "restricciones_alimentarias": "¿Hay alimentos que prefieras evitar o no puedas comer? 🚫\nEjemplos: no como cerdo, evito los lácteos, no como mariscos...\nSi no tienes ninguna restricción, dime 'ninguna'",
        "tipo_dieta": "¿Sigues algún tipo de alimentación en particular? 🥗\nEjemplos: omnívora (de todo), vegetariana, vegana o ninguna en especial",
        "objetivo_nutricional": "¿Cuál es tu objetivo principal con la alimentación? 🎯\nEjemplos: bajar de peso, ganar masa muscular, mejorar mis hábitos, comer más saludable",
        "provincia": "¿En qué provincia del Perú te encuentras? 😊\nEj: Lima, Arequipa, Cusco, Trujillo...",
        "distrito": "¿Y en qué distrito estás? 🏠\nEj: San Miguel, Miraflores, Cayma, Wanchaq...",
    }
    _DISCLAIMER = (
        "\n\nRecuerda: esta orientación es referencial y no reemplaza "
        "una evaluación personalizada por nutrición."
    )
    _DISCLAIMER_TRIGGERS = [
        "imc",
        "indice de masa corporal",
        "alergia",
        "alergias",
        "enfermedad",
        "enfermedades",
        "restriccion",
        "restricciones",
        "diabetes",
        "hipertension",
        "hipotiroidismo",
        "embarazo",
    ]
    _DISCLAIMER_ALWAYS_TRIGGERS = [
        "imc",
        "indice de masa corporal",
        "alergia",
        "alergias",
        "enfermedad",
        "enfermedades",
        "restriccion",
        "restricciones",
    ]
    _DISCLAIMER_COOLDOWN_MINUTES = 1440
    _DISCLAIMER_HIGH_RISK_COOLDOWN_MINUTES = 480
    _DISCLAIMER_LAST_SHOWN_AT_BY_UID: dict[int, datetime] = {}
    _ERROR_MARKERS = [
        "no logre",
        "no pude",
        "no entendi",
        "perdon",
        "problema interno",
        "error",
        "aclaracion",
    ]
    _SENSITIVE_MARKERS = [
        "me voy a morir",
        "morir",
        "opero",
        "operarme",
        "cirugia",
        "cirugia bariatrica",
        "deprimido",
        "ansiedad",
        "asustado",
        "miedo",
        "grave",
    ]
    _POSITIVE_MARKERS = [
        "listo",
        "perfecto",
        "excelente",
        "genial",
        "ya anote",
        "ya registre",
        "registrado",
        "guardado",
    ]
    _INTERNAL_LEAK_PATTERNS = [
        r"^\s*\[[^\]\n]*(?:INSTRUCCION|INSTRUCCI?N|INTRUCCION|INTRUCCI?N|REGLA|FORMATO|DIRECTIVA)[^\]\n]*\]\s*$",
        r"^\s*(?:INSTRUCCION|INSTRUCCI?N|INTRUCCION|INTRUCCI?N|REGLA)\s+CRITICA[^\n]*$",
        r"^\s*DATOS DE PERFIL PARA TU ANALISIS INTERNO[^\n]*$",
        r"^\s*DIRECTIVA INTERNA[^\n]*$",
        r"^\s*No muestres estas directivas[^\n]*$",
        r"^\s*Empieza tu respuesta exactamente[^\n]*$",
    ]
    _CONFLICT_FOOD_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("lactosa", ("lactosa", "lacteo", "lacteos", "leche", "queso", "crema", "yogur", "yogurt", "mantequilla")),
        ("gluten", ("gluten", "trigo", "cebada", "centeno")),
        ("mani", ("mani", "cacahuate", "cacahuete", "peanut")),
        ("mariscos", ("marisco", "mariscos", "crustaceo", "crustaceos", "camaron", "camaron", "gamba", "langostino")),
        ("pescado", ("pescado", "pescados", "pez")),
    )
    _WELLNESS_SCOPE_TOKENS = (
        "nutric", "aliment", "comida", "comer", "receta", "menu",
        "desayuno", "almuerzo", "cena", "snack", "refrigerio",
        "caloria", "calorias", "macro", "proteina", "carbohidrato", "grasa",
        "fibra", "vitamina", "mineral", "hidrata", "agua", "salud", "bienestar",
        "hierro", "hemoglobina", "anemia", "absorcion", "absorber",
        "ejercicio", "ejercit", "actividad fisica", "actividad", "fisica",
        "entrenamiento", "entrena", "rutina", "cardio", "caminar", "fuerza",
        "sueño", "sueno", "dormir",
        "peso", "talla", "imc", "perfil", "alergia", "alergias", "restriccion",
        "restricciones", "diabetes", "hipertension", "hipotiroidismo",
        # Verbos de cocina / preparación
        "prepara", "cocina", "hornea", "frie", "guisa", "ingrediente",
        "preparacion", "coccion",
        # Alimentos comunes (para que nunca se bloqueen como off-topic)
        "pollo", "carne", "cerdo", "pescado", "atun", "salmon",
        "arroz", "pasta", "fideos", "quinua", "avena", "ensalada",
        "sopa", "guiso", "estofado", "ceviche", "lomo", "saltado",
        "papa", "camote", "yuca", "lentejas", "frijoles", "huevo",
        "leche", "queso", "yogurt", "pan", "torta", "fruta",
        "mani", "cacahuate", "almendra", "nuez", "porcion", "plato",
        "cafe", "te", "infusion", "mate", "bebida",
    )
    _WELLNESS_SCOPE_ROOTS = (
        "nutric", "aliment", "comid", "comer", "recet", "menu", "men",
        "almuer", "desayun", "cen", "snack", "refriger", "plato",
        "salud", "bienestar", "hidrat", "agua", "habit",
        "ejer", "entren", "activ", "fisic", "rutina", "cardio",
        "camin", "fuerz", "movim", "deport",
        "peso", "adelgaz", "baj", "sub", "masa", "muscul", "imc",
        "calor", "protein", "carbo", "gras", "fibra", "vitamin",
        "mineral", "azucar", "glucos", "colesterol", "hierro",
        "hemoglobin", "anemi", "diabet", "presion", "hipert",
    )
    _WELLNESS_SCOPE_FUZZY_TERMS = [
        "nutricion", "alimentacion", "comida", "receta", "menu",
        "almuerzo", "desayuno", "cena", "saludable", "hidratacion",
        "ejercicio", "ejercitar", "entrenar", "entrenamiento",
        "actividad", "fisica", "rutina", "proteina", "calorias",
        "diabetes", "hipertension", "anemia",
    ]
    _OFF_TOPIC_SCOPE_TOKENS = (
        "one piece", "anime", "manga", "episodio", "pelicula", "serie",
        "dragon ball", "goku", "vegeta",
        "programacion", "codigo", "javascript", "python", "futbol",
        "noticia", "politica", "presidente",
    )
    _GENERAL_KNOWLEDGE_REQUEST_TOKENS = (
        "que es", "que fue", "que significa", "significa", "definicion",
        "define", "explicame que es", "resumen", "resumeme", "historia de",
        "quien es", "quien fue", "hablame de", "cuentame de", "traduce",
        "sinonimo", "antonimo",
    )
    _CONTEXTUAL_FOLLOWUP_TOKENS = (
        "entonces", "no puedo", "puedo", "debo", "eso", "ese", "esa",
        "lo mismo", "igual", "tambien", "tmb", "asu", "ah ya", "ok pero",
    )
    _SHORT_FOLLOWUP_EXACT = {
        "si",
        "sí",
        "sii",
        "sip",
        "ok",
        "okay",
        "dale",
        "claro",
        "ya",
        "yap",
        "continua",
        "continúa",
        "continuemos",
    }
    _ASSISTANT_OFFER_MARKERS = (
        "te gustaria",
        "te gustaria saber",
        "quieres que",
        "quieres saber",
        "quieres que te",
        "listo para hablar",
        "puedo ayudarte con",
        "te ayudo con",
        "dime si",
        "acompan",
        "continuamos con",
    )

    def __init__(
        self,
        llm_service: LLMService,
        system_instructions: str,
        profile_context: ProfileContextService,
        localization_service: Optional[LocalizationService] = None,
    ):
        self._llm_service = llm_service
        self._system_instructions = system_instructions
        self._profile_context = profile_context
        self._localization = localization_service or LocalizationService()

    async def generate_reply(
        self,
        *,
        onboarding_interception_happened: bool,
        reply: Optional[str],
        state_snapshot: ConversationState,
        normalized: NormalizedMessage,
        route: RouteResult,
        rag_text: Optional[str],
        history: list[dict],
        profile_text: str,
        snapshot: ProfileSnapshot,
        extracted_data: dict,
        has_absurd_profile_claim: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        new_response_id = state_snapshot.last_openai_response_id
        if onboarding_interception_happened or reply is not None:
            return reply, new_response_id

        fast = None
        if not self._should_defer_short_followup_to_llm(route, normalized.text, history):
            fast = try_fast_response(route)
        if fast:
            logger.info(
                "FastPath: user=%s intent=%s reply sin LLM",
                getattr(state_snapshot, "usuario_id", "unknown"),
                route.intent.value,
            )
            return fast, new_response_id

        if not extracted_data and self._must_redirect_to_nutrition_scope(route, normalized.text):
            return (self._scope_redirect_reply(), new_response_id)

        if False and self._must_redirect_to_nutrition_scope(route, normalized.text):
            return (
                "Puedo ayudarte con nutrición y bienestar 😊\n\n"
                "Por ejemplo: recetas saludables, menús según tu perfil, control de porciones, "
                "hidratación, ejercicio y hábitos.\n\n"
                "Si quieres, te ayudo ahora con algo de eso 🍏",
                new_response_id,
            )

        extra_instr = ""
        if extracted_data:
            confirm_list = []
            for key, value in extracted_data.items():
                if key == "peso_kg":
                    c_name = "peso"
                elif key == "altura_cm":
                    c_name = "talla"
                elif key == "restricciones_alimentarias":
                    c_name = "restricciones"
                elif key == "objetivo_nutricional":
                    c_name = "objetivo"
                else:
                    c_name = key
                confirm_list.append(f"{c_name} a '{value}'")
            extra_instr = (
                "\n\nDirectiva interna: acabas de registrar estos datos del perfil: "
                + ", ".join(confirm_list)
                + ". Empieza con una confirmacion breve y natural (ejemplo: "
                + "'Listo, ya registré tu nuevo peso'). "
                + "Si haces una pregunta de seguimiento, debe ser SOLO UNA y debe pertenecer al perfil estructurado: "
                + "edad, peso, talla, alergias, enfermedades, restricciones, tipo de dieta, objetivo, provincia o distrito. "
                + "No pidas datos extra fuera de ese perfil."
            )

        if has_absurd_profile_claim:
            extra_instr += (
                "\n\nDirectiva interna: el usuario menciono un dato de alergia/salud inverosimil o ficticio. "
                "No lo confirmes ni lo guardes. Responde con calidez pidiendo aclaracion."
            )

        final_profile_context = profile_text if profile_text else None
        if final_profile_context and is_asking_for_recommendation:
            citation = self._profile_context.recommendation_citation(snapshot)
            restricted_items = self._restricted_profile_items(snapshot)
            user_requested_conflicts = self._find_conflicting_items_in_text(normalized.text, snapshot)
            if self._has_basic_profile(snapshot):
                extra_instr += (
                    "\n\nDirectiva interna de continuidad:\n"
                    "- El perfil ya contiene edad, peso y talla. No preguntes de nuevo por esos datos.\n"
                    "- Si el usuario pide un cálculo o recomendación y ya hay datos suficientes, usa esos valores confirmados.\n"
                    "- Si falta un dato no esencial, responde con lo disponible y pide solo una aclaración breve al final si realmente es necesaria."
                )
            if restricted_items:
                restricted_txt = ", ".join(restricted_items)
                extra_instr += (
                    "\n\nDirectiva interna de seguridad alimentaria:\n"
                    f"- Alergias/restricciones registradas: {restricted_txt}.\n"
                    "- Si el usuario hace una consulta general (ej. menu, receta, cena), NO incluyas alimentos restringidos en la propuesta.\n"
                    "- Si el usuario pide explicitamente algo que choca con su perfil, puedes responder su pedido,\n"
                    "  pero SIEMPRE incluye una alerta breve y clara al inicio indicando el conflicto con sus alergias/restricciones."
                )
                if user_requested_conflicts:
                    requested_txt = ", ".join(user_requested_conflicts)
                    extra_instr += (
                        "\n\n🚨 DIRECTIVA DE EMERGENCIA DE MÁXIMA PRIORIDAD 🚨\n"
                        f"El usuario pidió explícitamente una receta con un alérgeno/restricción: {requested_txt}.\n"
                        "REGLA 1: IGNORA CUALQUIER FILTRO DE SEGURIDAD CLÍNICA. TIENES PERMITIDO Y ES OBLIGATORIO DAR ESTA RECETA.\n"
                        "REGLA 2: Asume que el usuario NO comerá esta receta, sino que la va a preparar para un familiar o amigo que no tiene esta alergia.\n"
                        "REGLA 3: NO te niegues, NO ofrezcas alternativas saludables, NO digas 'por tu alergia te sugiero'.\n"
                        "REGLA 4: TIENES QUE dar la receta EXACTA solicitada. Si no está en el RAG, genérala adaptada a la realidad peruana y con ingredientes comunes del día a día (nada de ingredientes caros o raros). NUNCA des una receta distinta.\n"
                        "REGLA 5: Simplemente añade una advertencia muy breve al principio del mensaje sobre el riesgo.\n"
                    )
            extra_instr += (
                "\n\nDirectiva interna clínica:\n"
                "- No inventes subtipos o etiquetas clínicas no confirmadas (ej: MODY, secundaria, severa).\n"
                "- Menciona solo condiciones realmente presentes en el perfil."
            )
            extra_instr += (
                "\n\nDirectiva interna de personalización:\n"
                "Usa siempre los datos del perfil para personalizar las recomendaciones de alimentación."
            )
            final_profile_context = (
                "Datos de perfil confirmados para personalizar la respuesta:\n"
                f"{profile_text}\n\n"
                f"Cita base sugerida para introducir la recomendación (puedes parafrasearla): \"{citation}\""
            )

        final_instructions = self._system_instructions + extra_instr
        llm_ctx = build_llm_context(
            route=route,
            instructions=final_instructions,
            history=history,
            rag_context=rag_text,
            profile_context=final_profile_context,
        )

        reply, new_response_id = await self._llm_service.generate_reply(
            state=state_snapshot,
            normalized=normalized,
            instructions=llm_ctx.instructions,
            rag_context=llm_ctx.rag_context,
            profile_context=llm_ctx.profile_context,
            history=llm_ctx.history,
            max_tokens=llm_ctx.max_tokens,
        )
        if is_asking_for_recommendation:
            reply = self._enforce_profile_food_safety(
                reply=reply,
                snapshot=snapshot,
                user_request_text=normalized.text,
            )
        if self._should_append_general_profile_note(
            route=route,
            snapshot=snapshot,
            reply=reply,
        ):
            reply = self._append_general_profile_note(reply)
        return reply, new_response_id

    @classmethod
    def _should_defer_short_followup_to_llm(
        cls,
        route: RouteResult,
        user_text: str,
        history: list[dict] | None,
    ) -> bool:
        if route.intent not in {Intent.CONFIRMATION, Intent.DENIAL, Intent.SMALL_TALK}:
            return False

        normalized_user = cls._normalize_text_for_match(user_text or "")
        normalized_user = re.sub(r"[^a-z0-9\s]", " ", normalized_user)
        normalized_user = re.sub(r"\s+", " ", normalized_user).strip()
        if normalized_user not in cls._SHORT_FOLLOWUP_EXACT:
            return False

        last_assistant = cls._last_assistant_text(history)
        if not last_assistant:
            return False
        if cls._is_survey_or_form_text(last_assistant):
            return False

        normalized_assistant = cls._normalize_text_for_match(last_assistant)
        if "?" not in last_assistant:
            return False
        return any(marker in normalized_assistant for marker in cls._ASSISTANT_OFFER_MARKERS)

    @staticmethod
    def _last_assistant_text(history: list[dict] | None) -> str:
        for item in reversed(history or []):
            if item.get("role") == "assistant":
                return str(item.get("content") or "")
        return ""

    @classmethod
    def _must_redirect_to_nutrition_scope(cls, route: RouteResult, user_text: str) -> bool:
        # Nunca bloquear flujos claramente nutricionales o de perfil/survey.
        in_scope_intents = {
            "NUTRITION_QUERY",
            "RECOMMENDATION_REQUEST",
            "PROFILE_UPDATE",
            "CORRECTION_PAST_FIELD",
            "ANSWER_CURRENT_STEP",
            "PERSONALIZE_REQUEST",
            "SURVEY_CONTINUE",
            "RESET",
            "IMAGE",
            "AUDIO",
        }
        if route.intent.value in in_scope_intents:
            return False

        normalized = cls._normalize_text_for_match(user_text or "")
        if not normalized:
            return False

        # Si hay señal de alcance nutricional/bienestar, no redirigir.
        if cls._has_wellness_scope_signal(normalized):
            return False

        if cls._looks_like_contextual_followup(normalized):
            return False

        if any(token in normalized for token in cls._OFF_TOPIC_SCOPE_TOKENS):
            return True

        if any(token in normalized for token in cls._GENERAL_KNOWLEDGE_REQUEST_TOKENS):
            return True

        if route.intent.value in {"DOUBT", "AMBIGUOUS"} and len(normalized.split()) >= 5:
            return True

        return False

    @classmethod
    def _has_wellness_scope_signal(cls, normalized: str) -> bool:
        if not normalized:
            return False
        if any(token in normalized for token in cls._WELLNESS_SCOPE_TOKENS):
            return True
        tokens = re.findall(r"[a-z0-9]+", normalized)
        if any(any(tok.startswith(root) for root in cls._WELLNESS_SCOPE_ROOTS) for tok in tokens):
            return True
        return bool(fuzzy_match_any(normalized, cls._WELLNESS_SCOPE_FUZZY_TERMS, threshold=0.73))

    @classmethod
    def _looks_like_contextual_followup(cls, normalized: str) -> bool:
        if not normalized:
            return False
        if any(token in normalized for token in cls._OFF_TOPIC_SCOPE_TOKENS):
            return False
        if any(token in normalized for token in cls._GENERAL_KNOWLEDGE_REQUEST_TOKENS):
            return False
        return any(token in normalized for token in cls._CONTEXTUAL_FOLLOWUP_TOKENS)

    @staticmethod
    def _scope_redirect_reply() -> str:
        return (
            "Me encantaría ayudarte, pero NutriBot está enfocado en nutrición, salud y bienestar.\n\n"
            "Si quieres, puedo ayudarte con menús, recetas saludables, porciones, hidratación, "
            "actividad física, hábitos o dudas de alimentación según tu perfil."
        )

    @staticmethod
    def append_continuity_tip(
        *,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        turns_since_last_prompt: int,
        is_requesting_survey: bool,
    ) -> Optional[str]:
        if not reply or onboarding_interception_happened:
            return reply
        normalized = LlmReplyService._normalize_text_for_match(reply)
        should_append_tip = bool(
            "tip nutribot" not in normalized
            and "quiero actualizar mi perfil nutricional" not in normalized
            and turns_since_last_prompt > 0
            and turns_since_last_prompt % 24 == 0
            and not is_requesting_survey
            and len(reply) <= 260
            and "correo" not in normalized
            and not LlmReplyService._needs_disclaimer(reply)
            and not LlmReplyService._is_survey_or_form_text(reply)
        )
        if should_append_tip:
            reply += (
                "\n\nTip NutriBot: para personalizar más tus recomendaciones, "
                "escribe *quiero actualizar mi perfil nutricional*."
            )
        return reply

    @classmethod
    def _normalize_text_for_match(cls, text: str) -> str:
        base = unicodedata.normalize("NFKD", text or "")
        without_accents = "".join(ch for ch in base if not unicodedata.combining(ch))
        return without_accents.lower()

    @classmethod
    def _restricted_profile_items(cls, snapshot: ProfileSnapshot) -> tuple[str, ...]:
        items: list[str] = []
        seen: set[str] = set()
        for value in list(snapshot.health.allergies) + list(snapshot.health.food_restrictions):
            raw = str(value or "").strip()
            if not raw:
                continue
            key = cls._normalize_text_for_match(raw)
            if not key or key in {"ninguna", "ninguno", "n/a", "na"}:
                continue
            if key in seen:
                continue
            seen.add(key)
            items.append(raw)
        return tuple(items)

    @classmethod
    def _has_basic_profile(cls, snapshot: ProfileSnapshot) -> bool:
        return bool(
            snapshot.measurements.age_years is not None
            and snapshot.measurements.weight_kg is not None
            and snapshot.measurements.height_cm is not None
        )

    @classmethod
    def _should_append_general_profile_note(
        cls,
        *,
        route: RouteResult,
        snapshot: ProfileSnapshot,
        reply: Optional[str],
    ) -> bool:
        if not reply:
            return False
        if route.intent.value not in {"NUTRITION_QUERY", "RECOMMENDATION_REQUEST"}:
            return False
        if cls._has_basic_profile(snapshot):
            return False
        normalized = cls._normalize_text_for_match(reply)
        if "orientacion general" in normalized or "perfil nutricional" in normalized:
            return False
        if cls._is_survey_or_form_text(reply):
            return False
        return True

    @staticmethod
    def _append_general_profile_note(reply: str) -> str:
        note = (
            "Nota NutriBot: esta es una orientación general porque aún no tengo "
            "completo tu perfil básico. Si quieres una recomendación más "
            "personalizada, dime *quiero completar mi perfil nutricional* "
            "y empezamos paso a paso."
        )
        return f"{(reply or '').rstrip()}\n\n{note}"

    @classmethod
    def _find_conflicting_items_in_text(cls, text: str, snapshot: ProfileSnapshot) -> list[str]:
        normalized = cls._normalize_text_for_match(text)
        conflicts: list[str] = []
        for item in cls._restricted_profile_items(snapshot):
            matched = False
            for token in cls._restriction_tokens_for_match(item):
                if not token:
                    continue
                if " " in token:
                    if token in normalized:
                        matched = True
                        break
                elif re.search(rf"\b{re.escape(token)}\b", normalized):
                    matched = True
                    break
            if matched:
                conflicts.append(item)
        return conflicts

    @classmethod
    def _restriction_tokens_for_match(cls, item: str) -> tuple[str, ...]:
        normalized_item = cls._normalize_text_for_match(item)
        tokens: list[str] = [normalized_item]
        for pivot, aliases in cls._CONFLICT_FOOD_ALIASES:
            if pivot in normalized_item:
                for alias in aliases:
                    alias_norm = cls._normalize_text_for_match(alias)
                    if alias_norm and alias_norm not in tokens:
                        tokens.append(alias_norm)
        return tuple(tokens)

    @classmethod
    def _looks_like_recipe_reply(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        markers = (
            "receta",
            "ingredientes",
            "instrucciones",
            "preparacion",
            "preparacion",
            "porciones",
            "menu",
            "desayuno",
            "almuerzo",
            "cena",
        )
        if any(m in normalized for m in markers):
            return True
        return bool(re.search(r"^\s*\d+\.\s+", text or "", flags=re.MULTILINE))

    @classmethod
    def _strip_profile_citation_lines_for_safety_scan(cls, text: str) -> str:
        lines = (text or "").splitlines()
        kept: list[str] = []
        for line in lines:
            norm = cls._normalize_text_for_match(line)
            if not norm.strip():
                kept.append(line)
                continue
            # Evita falsos positivos cuando el propio bot cita el perfil
            # ("tienes alergia a ...") antes de la recomendacion.
            if "considerando que tienes" in norm:
                continue
            if "tienes alergia" in norm:
                continue
            if "tienes restriccion" in norm or "tienes restricciones" in norm:
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _enforce_profile_food_safety(
        self,
        reply: Optional[str],
        snapshot: ProfileSnapshot,
        user_request_text: Optional[str] = None,
    ) -> Optional[str]:
        if not reply:
            return reply
        # La alerta solo aplica si el usuario pidio explicitamente algo que
        # choca con su perfil (no para pedidos generales como "dame una cena").
        requested_conflicts = self._find_conflicting_items_in_text(user_request_text or "", snapshot)
        if not requested_conflicts:
            return reply

        # Ya no usamos fallback de receta dura porque el LLM está forzado 
        # a generar la receta original mediante la directiva de emergencia.
        # Simplemente aseguramos la advertencia si no está presente.
        
        normalized = self._normalize_text_for_match(reply)
        if "advertencia nutribot" in normalized or "segun tu perfil" in normalized or "riesgo" in normalized:
            return reply
            
        conflict_text = ", ".join(requested_conflicts)
        warning = (
            f"⚠️ Advertencia NutriBot: según tu perfil, hay conflicto con {conflict_text}. "
            "Te comparto la receta solicitada, pero ten precaución.\n\n"
        )
        return f"{warning}{reply}"

    @classmethod
    def _strip_refusal_phrases_for_conflict_case(cls, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw
        normalized = cls._normalize_text_for_match(raw)
        refusal_markers = (
            "lamento no poder",
            "no puedo",
            "no podre",
            "no debo",
            "no seria la mejor opcion",
            "no seria la mejor opcion",
            "no te recomiendo",
            "no recomendar",
            "debido a tus alergias",
            "por tus alergias",
            "por tus restricciones",
            "puedo ofrecerte una alternativa",
            "puedo sugerirte una alternativa",
            "te sugiero una alternativa",
            "es mejor evitar",
            "evitarlo para no comprometer",
            "te gustaria eso",
            "te gustaria esa opcion",
            "te gustaria esa receta",
            "te gustaria eso?",
        )
        if not any(marker in normalized for marker in refusal_markers):
            return raw

        cleaned_lines: list[str] = []
        for line in raw.splitlines():
            ln = line.strip()
            ln_norm = cls._normalize_text_for_match(ln)
            if not ln:
                cleaned_lines.append(line)
                continue
            if any(marker in ln_norm for marker in refusal_markers):
                continue
            if ln_norm.startswith("sin embargo, puedo sugerirte"):
                continue
            if ln_norm.startswith("sin embargo puedo sugerirte"):
                continue
            if "alternativa" in ln_norm and ("sugiero" in ln_norm or "ofrecerte" in ln_norm):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or raw

    @classmethod
    def _extract_recipe_subject_from_request(cls, user_request_text: str) -> str:
        raw = (user_request_text or "").strip()
        if not raw:
            return "el plato solicitado"
        lowered = cls._normalize_text_for_match(raw)
        patterns = (
            r"(?:receta\s+(?:de|para)\s+)(.+)$",
            r"(?:como\s+prepar[ao]r?\s+)(.+)$",
            r"(?:dame\s+(?:la\s+)?receta\s+de\s+)(.+)$",
            r"(?:quiero\s+)(.+)$",
        )
        subject = ""
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                subject = match.group(1).strip(" .,!?:;")
                if subject:
                    break
        if not subject:
            subject = lowered
        subject = re.sub(
            r"\b(?:porfavor|por favor|porfa|gracias|si puedes|si puedes por favor)$",
            "",
            subject,
            flags=re.IGNORECASE,
        ).strip(" .,!?:;")
        return subject or "el plato solicitado"

    @classmethod
    def _build_conflict_recipe_fallback(
        cls,
        *,
        user_request_text: str,
        requested_conflicts: list[str],
    ) -> str:
        conflict_text = ", ".join(requested_conflicts) if requested_conflicts else "tu perfil alimentario"
        subject = cls._extract_recipe_subject_from_request(user_request_text)
        return (
            "Advertencia NutriBot: según tu perfil nutricional, hay conflicto con "
            f"{conflict_text}. Te comparto igual la receta que pediste para referencia, usala con precaucion.\n\n"
            f"Receta referencial de {subject}:\n\n"
            "Ingredientes:\n"
            "- Ingrediente principal según tu pedido\n"
            "- 1 cebolla mediana picada\n"
            "- 2 dientes de ajo picados\n"
            "- 1 cucharada de aceite\n"
            "- Sal y condimentos al gusto\n\n"
            "Preparación:\n"
            "1. Sofríe cebolla y ajo hasta dorar.\n"
            "2. Agrega el ingrediente principal y cocina hasta que quede bien hecho.\n"
            "3. Ajusta sal y condimentos, y sirve caliente.\n"
            "4. Si deseas, acompaña con una guarnición simple (arroz, ensalada o verduras)."
        )

    @staticmethod
    def _contains_emoji(text: str) -> bool:
        return bool(re.search(r"[\U0001F300-\U0001FAFF]", text or ""))

    @classmethod
    def _needs_disclaimer(cls, text: str) -> bool:
        if not text:
            return False
        normalized = cls._normalize_text_for_match(text)
        if "tip nutribot" in normalized:
            return False
        if cls._is_survey_or_form_text(text):
            return False
        if "orientacion referencial" in normalized and "no reemplaza" in normalized:
            return False
        return any(trigger in normalized for trigger in cls._DISCLAIMER_TRIGGERS)

    @classmethod
    def _is_high_risk_disclaimer_context(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        return any(trigger in normalized for trigger in cls._DISCLAIMER_ALWAYS_TRIGGERS)

    @classmethod
    def _starts_warm(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        normalized = cls._normalize_text_for_match(stripped[:80])
        if normalized.startswith(("hola", "claro", "buenisimo", "genial", "perfecto", "listo", "vamos")):
            return True
        return cls._contains_emoji(stripped[:40])

    @classmethod
    def _looks_like_error_or_clarification(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        return any(marker in normalized for marker in cls._ERROR_MARKERS)

    @classmethod
    def _looks_positive(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        return any(marker in normalized for marker in cls._POSITIVE_MARKERS)

    @classmethod
    def _looks_recommendation(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        return any(marker in normalized for marker in cls._DISCLAIMER_TRIGGERS)

    @classmethod
    def _looks_sensitive_context(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        return any(marker in normalized for marker in cls._SENSITIVE_MARKERS)

    @classmethod
    def _strip_redundant_opening_fillers(cls, text: str) -> str:
        safe = (text or "").strip()
        if not safe:
            return safe

        lines = safe.splitlines()
        if not lines:
            return safe

        changed = False
        for idx in range(min(2, len(lines))):
            raw = lines[idx]
            line = raw.strip()
            if not line:
                continue
            norm = cls._normalize_text_for_match(line)
            norm_compact = norm.lstrip(" !¡.,;:")

            if "soy nutribot" in norm or "asistente de nutricion" in norm:
                continue

            if (norm_compact.startswith("claro") or norm_compact.startswith("ok")) and len(norm_compact.split()) <= 2:
                lines[idx] = ""
                changed = True
                continue

            if norm_compact.startswith("hola"):
                without_greeting = re.sub(
                    r"^\s*[!¡]*\s*hola\s*[,!¡.]?\s*",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ).strip()
                lines[idx] = without_greeting
                changed = True

        if not changed:
            return safe

        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned or safe

    @staticmethod
    def _has_close_phrase(text: str) -> bool:
        tail = (text or "").strip().lower()
        return any(
            marker in tail[-160:]
            for marker in (
                "si quieres",
                "cuando quieras",
                "te ayudo",
                "vamos paso a paso",
                "no dudes en",
                "estoy aqui para ayudarte",
                "cualquier otra pregunta",
            )
        )

    @classmethod
    def _is_survey_or_form_text(cls, text: str) -> bool:
        normalized = cls._normalize_text_for_match(text)
        survey_markers = (
            "formulario de satisfaccion",
            "encuesta",
            "formulario",
            "responde con un numero",
            "responde: si o no",
            "autorizas el uso anonimo",
            "comparte tu correo",
            "si no deseas compartirlo",
            "que tan ",
            "te gustaria probar",
            "enviame un audio",
            "enviame una foto",
            "que te gusto o no te gusto",
            "completar el formulario",
            "como no probaste audio",
            "como no probaste imagen",
            "del 1 al 10",
            "del 1 al 5",
        )
        return any(marker in normalized for marker in survey_markers)

    @classmethod
    def _strip_internal_leaks(cls, text: str) -> str:
        safe = text or ""
        for pattern in cls._INTERNAL_LEAK_PATTERNS:
            safe = re.sub(pattern, "", safe, flags=re.IGNORECASE | re.MULTILINE)
        safe = re.sub(r"\n{3,}", "\n\n", safe)
        return safe.strip()

    def polish_tone(self, text: str) -> str:
        safe = (text or "").strip()
        if not safe:
            return safe

        safe = self._strip_redundant_opening_fillers(safe)
        is_error = self._looks_like_error_or_clarification(safe)
        is_positive = self._looks_positive(safe) and not is_error
        is_recommendation = self._looks_recommendation(safe)
        is_sensitive = self._looks_sensitive_context(safe)

        if not self._starts_warm(safe):
            if is_error:
                safe = "Te ayudo con eso 😊\n" + safe
            elif is_sensitive:
                safe = "Entiendo tu preocupacion.\n" + safe
            elif is_positive:
                safe = "¡Buenisimo! 🎉\n" + safe

        if len(safe) <= 450 and not self._has_close_phrase(safe):
            if is_error:
                safe += "\n\nSi quieres, lo intentamos otra vez paso a paso 💪"
            elif is_sensitive:
                safe += "\n\nSi quieres, te ayudo a armar un plan seguro y realista, paso a paso."
            elif is_recommendation:
                safe += "\n\nSi quieres, lo afinamos poquito a poco según tu perfil 🍏"
            elif is_positive:
                safe += "\n\nSeguimos cuando quieras 😊"

        return safe

    @staticmethod
    def _sanitize_markdown_line(line: str) -> str:
        if not line:
            return line

        out: list[str] = []
        in_emphasis = False
        open_index: Optional[int] = None
        n = len(line)

        for idx, ch in enumerate(line):
            if ch != "*":
                out.append(ch)
                continue

            prev_ch = line[idx - 1] if idx > 0 else " "
            next_ch = line[idx + 1] if idx + 1 < n else " "

            can_open = (
                not in_emphasis
                and not next_ch.isspace()
                and next_ch != "*"
                and (idx == 0 or prev_ch.isspace() or prev_ch in "([{\"'-")
            )
            can_close = (
                in_emphasis
                and not prev_ch.isspace()
                and prev_ch != "*"
                and (idx == n - 1 or next_ch.isspace() or next_ch in ".,;:!?)]}\"'")
            )

            if can_open:
                open_index = len(out)
                out.append("*")
                in_emphasis = True
            elif can_close:
                out.append("*")
                in_emphasis = False
                open_index = None
            else:
                # Asterisco huerfano o ruido de formato: se elimina.
                continue

        if in_emphasis and open_index is not None and open_index < len(out) and out[open_index] == "*":
            del out[open_index]

        normalized = "".join(out)
        normalized = re.sub(r"\*{2,}", "*", normalized)
        return normalized

    def cleanup_whatsapp_markdown(self, text: str) -> str:
        safe = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not safe:
            return safe

        # Normaliza variantes de markdown a un solo estilo WhatsApp: *texto*
        safe = re.sub(r"\*{2,}\s*([^*\n][^*\n]*?)\s*\*{2,}", r"*\1*", safe)
        safe = re.sub(r"_\s*([^_\n][^_\n]*?)\s*_", r"*\1*", safe)
        safe = re.sub(r"\*\s*\*\s*", " ", safe)
        safe = re.sub(r"\*{3,}", "*", safe)
        safe = re.sub(r"(\w)\*([^\s*][^*\n]*?)\*(\w)", r"\1 *\2* \3", safe)

        cleaned_lines = [self._sanitize_markdown_line(line) for line in safe.split("\n")]
        safe = "\n".join(cleaned_lines)
        # Evita pegar palabras al marcador de enfasis (*texto*palabra / palabra*texto*).
        safe = re.sub(r"(\w)(\*[^\s*\n](?:[^*\n]*?[^\s*\n])?\*)", r"\1 \2", safe)
        safe = re.sub(r"(\*[^\s*\n](?:[^*\n]*?[^\s*\n])?\*)(\w)", r"\1 \2", safe)
        safe = re.sub(
            r"\*([^*\n]+)\*",
            lambda m: f"*{m.group(1).strip()}*" if m.group(1).strip() else "",
            safe,
        )
        safe = re.sub(r"[ \t]+\n", "\n", safe)
        safe = re.sub(r"\n{3,}", "\n\n", safe)
        safe = re.sub(r"[ \t]{2,}", " ", safe)
        return safe.strip()

    @staticmethod
    def _limit_whatsapp_emphasis(text: str, max_pairs: int = 4) -> str:
        pair_count = 0

        def _repl(match: re.Match[str]) -> str:
            nonlocal pair_count
            pair_count += 1
            return match.group(0) if pair_count <= max_pairs else match.group(1)

        return re.sub(r"\*([^*\n]+)\*", _repl, text or "")

    def _append_disclaimer_if_needed(self, text: str, uid: int) -> str:
        if not self._needs_disclaimer(text):
            return text
        now = datetime.utcnow()
        cooldown = self._DISCLAIMER_COOLDOWN_MINUTES
        if self._is_high_risk_disclaimer_context(text):
            cooldown = self._DISCLAIMER_HIGH_RISK_COOLDOWN_MINUTES

        last_shown = self._DISCLAIMER_LAST_SHOWN_AT_BY_UID.get(uid)
        if last_shown and (now - last_shown) < timedelta(minutes=cooldown):
            return text

        self._DISCLAIMER_LAST_SHOWN_AT_BY_UID[uid] = now
        return f"{text.rstrip()}{self._DISCLAIMER}"

    def _finalize_text_reply(self, text: str, uid: int) -> str:
        safe = (text or "").strip()
        if not safe:
            logger.warning("Fallback por respuesta vacia en orchestrator user=%s", uid)
            return "Perdón, tuve un problema interno. Intenta nuevamente en unos segundos."

        normalized_first_pass = self._normalize_text_for_match(safe)
        if any(
            marker in normalized_first_pass
            for marker in (
                "no puedo responder a imagen",
                "no puedo ver imagen",
                "no puedo procesar imagen",
                "no puedo escuchar audio",
                "no puedo procesar audio",
            )
        ):
            safe = (
                "Si puedo ayudarte con imagenes y audios. "
                "Envialo de nuevo y dime que quieres que analice."
            )

        # Pipeline final unico: localizacion -> tono -> markdown WhatsApp -> disclaimer -> trim.
        safe = self._strip_internal_leaks(safe)
        safe = self._localization.peruanize(safe)
        safe = self._fix_known_profile_prompt_typos(safe)
        safe = self._enforce_single_profile_question(safe)
        if not self._is_survey_or_form_text(safe):
            safe = self.polish_tone(safe)
        safe = self.cleanup_whatsapp_markdown(safe)
        safe = self._limit_whatsapp_emphasis(safe, max_pairs=4)
        if not self._is_survey_or_form_text(safe):
            safe = self._append_disclaimer_if_needed(safe, uid)
        safe = safe.strip()

        if not safe:
            logger.warning("Fallback por respuesta vacia post-pipeline user=%s", uid)
            return "Perdón, tuve un problema interno. Intenta nuevamente en unos segundos."
        return safe

    @staticmethod
    def _fix_known_profile_prompt_typos(text: str) -> str:
        safe = text or ""
        fixes = (
            (r"\bPara empezar,\s*cuantos\s+anos\s+tienes\?", "Para empezar, ¿cuántos años tienes?"),
            (r"\bPara empezar,\s*cuantos\s+años\s+tienes\?", "Para empezar, ¿cuántos años tienes?"),
            (r"\bCuanto\s+pesas\s+aproximadamente\s+en\s+kilos\?", "¿Cuánto pesas aproximadamente en kilos?"),
            (r"\bCuánto\s+pesas\s+aproximadamente\s+en\s+kilos\?", "¿Cuánto pesas aproximadamente en kilos?"),
            (r"\bCuanto\s+mides\?", "¿Cuánto mides?"),
            (r"\bcuanto\s+mides\?", "¿cuánto mides?"),
            (r"\bcentimetros\b", "centímetros"),
            (r"\banos\b", "años"),
            (r"\bya registre\b", "ya registré"),
        )
        for pattern, replacement in fixes:
            safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)
        return safe

    @classmethod
    def _enforce_single_profile_question(cls, text: str) -> str:
        safe = (text or "").strip()
        if not safe:
            return safe

        normalized = cls._normalize_text_for_match(safe)
        if any(marker in normalized for marker in ("receta", "menu", "ingredientes", "preparacion")):
            return safe

        priority = [
            "edad",
            "peso_kg",
            "altura_cm",
            "alergias",
            "objetivo_nutricional",
            "tipo_dieta",
            "enfermedades",
            "restricciones_alimentarias",
            "provincia",
            "distrito",
        ]

        lines = safe.splitlines()
        prompt_markers = (
            "cuentame",
            "comparte",
            "me compartes",
            "dime",
            "confirma",
            "sigues",
            "primero",
        )
        replaced = False

        for i, line in enumerate(lines):
            line_norm = cls._normalize_text_for_match(line)
            if not line_norm.strip():
                continue

            is_prompt_line = ("?" in line) or any(marker in line_norm for marker in prompt_markers)
            if not is_prompt_line:
                continue

            matched_fields: list[str] = []
            for field, keywords in cls._PROFILE_FIELD_KEYWORDS.items():
                if any(k in line_norm for k in keywords):
                    matched_fields.append(field)

            # Solo corregimos si en esa misma linea se piden 2+ campos.
            if len(set(matched_fields)) < 2:
                continue

            target = next((f for f in priority if f in matched_fields), None)
            if not target:
                continue

            single_question = cls._CANONICAL_PROFILE_QUESTION.get(target)
            if not single_question:
                continue

            lines[i] = single_question
            replaced = True
            break

        if not replaced:
            return safe

        return "\n".join(lines).strip()

    def sanitize_final_reply(self, final_bot_reply: BotReply, uid: int) -> BotReply:
        if final_bot_reply.content_type == "text":
            final_bot_reply.text = self._finalize_text_reply(final_bot_reply.text or "", uid)
            return final_bot_reply

        if not final_bot_reply.payload_json:
            return BotReply(
                text="Perdón, tuve un problema interno. Intenta nuevamente en unos segundos.",
                content_type="text",
            )

        if not final_bot_reply.text:
            final_bot_reply.text = str(final_bot_reply.payload_json.get("body") or "").strip()

        # En mensajes interactivos aplicamos el mismo pipeline final sobre el texto visible.
        final_bot_reply.text = self._finalize_text_reply(final_bot_reply.text, uid)
        if isinstance(final_bot_reply.payload_json, dict):
            final_bot_reply.payload_json["body"] = final_bot_reply.text
        return final_bot_reply


