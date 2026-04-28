"""
Nutribot Backend — Context Builder
====================================
Construye el contexto mínimo necesario para cada llamada al LLM
basado en el intent del router barato.

Objetivo: reducir tokens enviados por turno sin perder calidad.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from domain.router import Intent, RouteResult

logger = logging.getLogger(__name__)


@dataclass
class LLMContext:
    """Contexto optimizado para enviar al LLM."""
    instructions: str
    history: Optional[list[dict]]
    rag_context: Optional[str]
    profile_context: Optional[str]
    max_tokens: int


# ──────────────────────────────────────────────
# Configuración de contexto por intent
# ──────────────────────────────────────────────

CONTEXT_RULES = {
    # GREETING: no necesita historial largo, ni RAG, ni perfil detallado
    Intent.GREETING: {
        "history_limit": 4,     # últimos 2 pares
        "include_rag": False,
        "include_profile": False,
        "max_tokens": 200,
    },
    # SMALL_TALK: respuesta breve
    Intent.SMALL_TALK: {
        "history_limit": 6,
        "include_rag": False,
        "include_profile": False,
        "max_tokens": 200,
    },
    # CONFIRMATION/DENIAL: contexto mínimo
    Intent.CONFIRMATION: {
        "history_limit": 4,
        "include_rag": False,
        "include_profile": False,
        "max_tokens": 150,
    },
    Intent.DENIAL: {
        "history_limit": 4,
        "include_rag": False,
        "include_profile": False,
        "max_tokens": 150,
    },
    # DOUBT: necesita historial para saber qué pregunta
    Intent.DOUBT: {
        "history_limit": 8,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 400,
    },
    # PROFILE_UPDATE / CORRECTION: necesita perfil para contexto
    Intent.PROFILE_UPDATE: {
        "history_limit": 6,
        "include_rag": False,
        "include_profile": True,
        "max_tokens": 300,
    },
    Intent.CORRECTION_PAST_FIELD: {
        "history_limit": 6,
        "include_rag": False,
        "include_profile": True,
        "max_tokens": 300,
    },
    # PERSONALIZE_REQUEST: perfil sí, RAG no
    Intent.PERSONALIZE_REQUEST: {
        "history_limit": 6,
        "include_rag": False,
        "include_profile": True,
        "max_tokens": 400,
    },
    # NUTRITION_QUERY: TODO el contexto (la consulta principal)
    Intent.NUTRITION_QUERY: {
        "history_limit": 12,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 800,
    },
    Intent.RECOMMENDATION_REQUEST: {
        "history_limit": 12,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 1000,
    },
    # IMAGE: necesita todo (Vision analysis)
    Intent.IMAGE: {
        "history_limit": 8,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 800,
    },
    # AUDIO: transcripción, contexto medio
    Intent.AUDIO: {
        "history_limit": 8,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 600,
    },
    # AMBIGUOUS: todo porque no sabemos qué es
    Intent.AMBIGUOUS: {
        "history_limit": 12,
        "include_rag": True,
        "include_profile": True,
        "max_tokens": 600,
    },
}

# Default para intents no listados
DEFAULT_RULES = {
    "history_limit": 10,
    "include_rag": True,
    "include_profile": True,
    "max_tokens": 600,
}

RAG_TECHNICAL_MARKERS = (
    "anemia", "hierro", "hemoglobina", "diabetes", "glucosa",
    "hipertension", "presion", "colesterol", "trigliceridos",
    "gastritis", "rinon", "renal", "embarazo", "lactancia",
    "imc", "caloria", "proteina", "carbohidrato", "vitamina",
    "mineral", "fibra", "absorcion", "absorber", "deficiencia",
    "guia", "recomendacion", "recomendaciones", "menu", "receta",
    "plan", "porciones", "dieta", "que comer", "alimentos",
)

RAG_EXPLANATION_SHAPES = (
    "por que", "porque", "explicame", "dime mas", "cuanto",
    "cuanta", "cuantas", "cuantos", "que alimentos", "que puedo",
    "que debo", "puedo comer", "no puedo", "es malo", "es bueno",
)


def should_fetch_rag(route: RouteResult, user_text: str) -> bool:
    """
    Decide si vale la pena pagar embedding + busqueda RAG antes del LLM.

    La puerta no decide si el bot entiende o responde: solo decide si conviene
    adjuntar documento externo. Si hay duda, el LLM sigue recibiendo el turno
    con perfil e historial compacto.
    """
    if route.intent in {
        Intent.GREETING,
        Intent.CONFIRMATION,
        Intent.DENIAL,
        Intent.SKIP,
        Intent.RESET,
        Intent.SMALL_TALK,
        Intent.ANSWER_CURRENT_STEP,
        Intent.PROFILE_UPDATE,
        Intent.CORRECTION_PAST_FIELD,
        Intent.PERSONALIZE_REQUEST,
        Intent.SURVEY_CONTINUE,
    }:
        return False

    if route.intent in {Intent.IMAGE, Intent.AUDIO}:
        return True

    normalized = (user_text or "").lower()
    if not normalized.strip():
        return False

    if route.intent == Intent.RECOMMENDATION_REQUEST:
        return True

    if route.intent in {Intent.NUTRITION_QUERY, Intent.DOUBT, Intent.AMBIGUOUS}:
        has_marker = any(marker in normalized for marker in RAG_TECHNICAL_MARKERS)
        has_shape = any(shape in normalized for shape in RAG_EXPLANATION_SHAPES)
        return has_marker and (has_shape or len(normalized.split()) >= 5)

    return False


def build_llm_context(
    route: RouteResult,
    instructions: str,
    history: Optional[list[dict]],
    rag_context: Optional[str],
    profile_context: Optional[str],
) -> LLMContext:
    """
    Construye el contexto mínimo necesario para la llamada al LLM
    basado en el intent clasificado por el router.

    Ahorra tokens al no enviar historial/RAG/perfil cuando no se necesita.
    """
    rules = CONTEXT_RULES.get(route.intent, DEFAULT_RULES)

    # Limitar historial
    trimmed_history = None
    if history:
        limit = rules["history_limit"]
        trimmed_history = history[-limit:] if len(history) > limit else history

    # Filtrar RAG y perfil según el intent
    filtered_rag = rag_context if rules["include_rag"] else None
    filtered_profile = profile_context if rules["include_profile"] else None

    ctx = LLMContext(
        instructions=instructions,
        history=trimmed_history,
        rag_context=filtered_rag,
        profile_context=filtered_profile,
        max_tokens=rules["max_tokens"],
    )

    # Log de ahorro
    original_parts = sum([
        1 if history else 0,
        1 if rag_context else 0,
        1 if profile_context else 0,
    ])
    final_parts = sum([
        1 if trimmed_history else 0,
        1 if filtered_rag else 0,
        1 if filtered_profile else 0,
    ])
    if original_parts > final_parts:
        logger.info(
            "ContextBuilder: intent=%s stripped %d context parts (history=%s→%s, rag=%s, profile=%s)",
            route.intent.value,
            original_parts - final_parts,
            len(history) if history else 0,
            len(trimmed_history) if trimmed_history else 0,
            bool(filtered_rag),
            bool(filtered_profile),
        )

    return ctx


# ──────────────────────────────────────────────
# Respuestas fast-path (corto-circuito sin LLM)
# ──────────────────────────────────────────────

FAST_RESPONSES = {
    Intent.GREETING: [
        (
            "Hola, soy NutriBot. Que gusto leerte. "
            "Estoy aqui para acompanarte con nutricion, salud y bienestar: "
            "menus, recetas, porciones, habitos y recomendaciones segun tu perfil."
        ),
        (
            "Hola, soy NutriBot. Me alegra saludarte. "
            "Puedo ayudarte con nutricion, salud y bienestar de forma simple y personalizada."
        ),
    ],
    Intent.CONFIRMATION: [
        "Listo.",
        "Perfecto.",
    ],
    Intent.DENIAL: [
        "Entendido.",
        "Ok, lo dejamos asi.",
    ],
    Intent.SMALL_TALK: [
        "Te leo. Si es sobre nutricion o bienestar, cuentame.",
        "Dime nomas, te ayudo con nutricion y bienestar.",
    ],
    Intent.SKIP: [
        "¡Sin problema! 😊 Lo dejamos para otro momento. Si quieres continuar después, solo dime.",
        "¡Perfecto, lo omitimos! Si cambias de opinión, avísame cuando gustes. 😊",
    ],
}


def try_fast_response(route: RouteResult) -> Optional[str]:
    """
    Para algunos intents con alta confianza, retorna una respuesta
    predefinida sin llamar al LLM.

    Retorna None si este intent requiere LLM.
    """
    # Solo corto-circuitar con alta confianza y en intents muy predecibles
    if route.confidence < 0.85:
        return None

    responses = FAST_RESPONSES.get(route.intent)
    if responses:
        # Rotar respuestas para variedad
        import hashlib
        h = int(hashlib.md5((route.reason or "").encode()).hexdigest(), 16)
        return responses[h % len(responses)]

    return None
