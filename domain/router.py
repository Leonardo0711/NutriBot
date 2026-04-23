"""
Nutribot Backend — Cheap Universal Router
==========================================
Clasifica mensajes dentro del flujo conversacional SIN usar IA.
Usa señales baratas: texto normalizado, números, unidades,
contexto del paso actual y similitud aproximada.

Meta: resolver 70-85% de mensajes operativos sin llamar a OpenAI.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from domain.normalizer import (
    normalize_text,
    extract_numbers,
    extract_number_with_unit,
    fuzzy_match,
    fuzzy_match_any,
)

logger = logging.getLogger(__name__)


# Ruido conversacional que jamás debe aceptarse como dato de perfil
_CONVERSATIONAL_NOISE = frozenset({
    "gracias", "ok", "okey", "dale", "ya", "listo", "bueno",
    "bien", "perfecto", "genial", "entendido", "vale", "claro",
    "aja", "simon", "sip", "sep", "va", "vamos",
    "jaja", "jajaja", "xd", "haha", "jejeje",
    "que", "como", "porque", "cuando",
    "hola", "hey", "alo", "buenas",
    "normal", "lo de siempre", "nada mas", "eso",
    "lo mismo", "igualmente", "chido", "bacano",
})

# Contenido semántico mínimo: frases vacías que parecen dato pero no lo son
_SEMANTIC_NOISE_PHRASES = frozenset({
    "lo de siempre", "nada mas", "nada más", "eso", "lo mismo",
    "normal", "lo normal", "todo bien", "nada especial",
    "no mucho", "regular", "asi nomas", "así nomás",
})


# ──────────────────────────────────────────────
# Intenciones del Router
# ──────────────────────────────────────────────

class Intent(str, Enum):
    GREETING = "GREETING"
    RESET = "RESET"
    ANSWER_CURRENT_STEP = "ANSWER_CURRENT_STEP"
    CORRECTION_PAST_FIELD = "CORRECTION_PAST_FIELD"
    PROFILE_UPDATE = "PROFILE_UPDATE"
    PERSONALIZE_REQUEST = "PERSONALIZE_REQUEST"
    NUTRITION_QUERY = "NUTRITION_QUERY"
    RECOMMENDATION_REQUEST = "RECOMMENDATION_REQUEST"
    SURVEY_CONTINUE = "SURVEY_CONTINUE"
    SMALL_TALK = "SMALL_TALK"
    SKIP = "SKIP"
    CONFIRMATION = "CONFIRMATION"
    DENIAL = "DENIAL"
    DOUBT = "DOUBT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class RouteResult:
    """Resultado de la clasificación del router."""
    intent: Intent
    confidence: float  # 0.0 - 1.0
    resolved_field: Optional[str] = None  # campo detectado (ej: "edad")
    resolved_value: Optional[str] = None  # valor detectado (ej: "43")
    reason: str = ""  # explicación para logging


# ──────────────────────────────────────────────
# Señales base
# ──────────────────────────────────────────────

GREETING_WORDS = [
    "hola", "buenas", "buen dia", "buenas tardes",
    "buenas noches", "que tal", "holis", "hey", "alo", "ola",
    "empezar", "arrancar", "comenzar", "inicio",
]

SKIP_WORDS = [
    "omitir", "saltar", "skip", "paso", "siguiente", "no quiero",
    "prefiero no", "no deseo", "despues", "luego", "no por ahora",
    "ahorita no", "mas tarde", "otro dia", "no gracias",
]

CONFIRMATION_WORDS = [
    "si", "sip", "sep", "claro", "dale", "va", "ok", "okey",
    "perfecto", "correcto", "exacto", "asi es", "de acuerdo",
    "ya", "listo", "bueno", "vamos", "empecemos", "dale pues",
    "simon", "aja", "afirmativo",
]

DENIAL_WORDS = [
    "no", "nop", "nel", "nah", "para nada", "negativo",
    "no quiero", "no gracias", "tampoco",
]

DOUBT_WORDS = [
    "como", "que es", "que significa", "para que", "por que",
    "cual", "cuanto", "no entiendo", "a que te refieres",
    "no comprendo", "explicame", "dime mas",
]

NUTRITION_KEYWORDS = [
    "menu", "receta", "dieta", "que como", "comida", "recomienda",
    "recomendacion", "almuerzo", "cena", "desayuno", "comer",
    "imc", "calorias", "grasa", "proteina", "carbohidratos",
    "gluten", "ayuno", "saludable", "nutricion", "alimentacion",
    "vitamina", "mineral", "fibra", "azucar", "colesterol",
    "frutas", "verduras", "vegetales",
]

PERSONALIZATION_ACTION_ROOTS = (
    "actualiz",
    "personaliz",
    "complet",
    "llen",
    "modific",
    "edit",
    "cambi",
    "ajust",
    "correg",
    "update",
)

PERSONALIZATION_PROFILE_ROOTS = (
    "perfil",
    "dato",
    "nutric",
    "cuenta",
)

SURVEY_KEYWORDS = [
    "encuesta", "formulario", "satisfaccion", "experiencia",
    "seguir con el formulario", "continuar formulario",
    "retomar encuesta", "mejorar nutribot",
]

CORRECTION_PATTERNS = [
    "me equivoque", "me equivoqué", "equivoque", "corrijo",
    "corregir", "era", "quise decir", "en realidad",
    "perdon", "disculpa", "no era", "cambie", "cambiar",
]

# Mapa de campos del perfil con alias para detección
PROFILE_FIELD_ALIASES = {
    "edad": ["edad", "anos", "years", "mi edad"],
    "peso_kg": ["peso", "kg", "kilos", "kilogramos", "mi peso"],
    "altura_cm": ["altura", "talla", "mido", "estatura", "cm", "metros", "mi talla", "mi altura"],
    "alergias": ["alergia", "alergias", "alergico", "intolerancia", "intolerante"],
    "enfermedades": ["enfermedad", "enfermedades", "diabetes", "hipertension", "condicion", "padezco"],
    "tipo_dieta": ["dieta", "tipo dieta", "vegano", "vegetariano", "keto", "omnivoro"],
    "objetivo_nutricional": ["objetivo", "meta", "quiero lograr", "bajar peso", "ganar masa", "subir peso"],
    "restricciones_alimentarias": ["restriccion", "restricciones", "no como", "no me gusta", "evito"],
    "region": ["region", "departamento"],
    "provincia": ["provincia"],
    "distrito": ["distrito"],
}

DIET_PATTERNS = {
    "vegetariano": "VEGETARIANA",
    "vegetariana": "VEGETARIANA",
    "vegano": "VEGANA",
    "vegana": "VEGANA",
    "keto": "KETO",
    "cetogenica": "KETO",
    "cetogenico": "KETO",
    "omnivoro": "OMNIVORA",
    "omnivora": "OMNIVORA",
    "sin gluten": "SIN GLUTEN",
    "pescetariano": "PESCETARIANA",
    "pescetariana": "PESCETARIANA",
}

DISEASE_HINTS = [
    "diabetes", "hipertension", "hipotiroidismo", "anemia",
    "colesterol", "gastritis", "obesidad", "insulinoresistencia",
]

# Rangos válidos para validación numérica de campos
FIELD_RANGES = {
    "edad": (1, 120),
    "peso_kg": (2, 400),
    "altura_cm": (30, 250),
}


def _contains_keyword(text: str, keyword: str) -> bool:
    """Coincidencia robusta: palabra completa para tokens simples."""
    if not keyword:
        return False
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _starts_with_keyword(text: str, keyword: str) -> bool:
    if text == keyword:
        return True
    return text.startswith(keyword + " ")


# ──────────────────────────────────────────────
# Router Principal
# ──────────────────────────────────────────────

def classify_message(
    raw_text: str,
    current_mode: str,
    onboarding_status: str,
    onboarding_step: Optional[str],
    content_type: str = "text",
) -> RouteResult:
    """
    Clasifica un mensaje entrante SIN usar IA.

    Args:
        raw_text: Texto crudo del usuario
        current_mode: Modo actual del conversation_state (active_chat, collecting_profile, etc.)
        onboarding_status: Estado del onboarding (not_started, invited, in_progress, completed)
        onboarding_step: Paso actual del onboarding si aplica
        content_type: Tipo de contenido (text, audio, image)

    Returns:
        RouteResult con la intención, confianza, y opcionalmente campo/valor detectados
    """
    # Tipos no-texto se delegan directo a IA
    if content_type == "image":
        return RouteResult(Intent.IMAGE, 1.0, reason="Imagen recibida")
    if content_type == "audio":
        return RouteResult(Intent.AUDIO, 0.8, reason="Audio recibido (requiere STT)")

    norm = normalize_text(raw_text)
    numbers = extract_numbers(norm)
    units = extract_number_with_unit(norm)

    # 1. RESET
    if norm in ("/reset", "reset", "reiniciar", "borrar datos", "empezar de cero"):
        return RouteResult(Intent.RESET, 1.0, reason="Comando de reset explícito")

    # 2. SALUDO (mensajes muy cortos y directos)
    if len(norm.split()) <= 3:
        for gw in GREETING_WORDS:
            if _starts_with_keyword(norm, gw):
                return RouteResult(Intent.GREETING, 0.9, reason=f"Saludo detectado: '{gw}'")

    # 3. SKIP (durante onboarding)
    if onboarding_status in ("invited", "in_progress"):
        for sw in SKIP_WORDS:
            if _contains_keyword(norm, sw):
                return RouteResult(Intent.SKIP, 0.9, reason=f"Skip detectado: '{sw}'")

    # 4. CONFIRMACIÓN / NEGACIÓN simples
    restriction_expression = (
        norm.startswith("no como ")
        or norm.startswith("no me gusta")
        or norm.startswith("no consumo ")
        or norm.startswith("evito ")
    )
    if len(norm.split()) <= 3:
        for cw in CONFIRMATION_WORDS:
            if _starts_with_keyword(norm, cw):
                return RouteResult(Intent.CONFIRMATION, 0.9, reason=f"Confirmación: '{cw}'")
        for dw in DENIAL_WORDS:
            if restriction_expression:
                continue
            if _starts_with_keyword(norm, dw):
                return RouteResult(Intent.DENIAL, 0.85, reason=f"Negación: '{dw}'")

    # 5. RESPUESTA AL PASO ACTUAL (onboarding en progreso)
    if onboarding_status == "in_progress" and onboarding_step:
        answer = _try_answer_current_step(norm, onboarding_step, numbers, units)
        if answer:
            return answer

    # 6. CORRECCIÓN de campo anterior
    correction = _try_detect_correction(norm, numbers, units)
    if correction:
        return correction

    # 7. SOLICITUD DE PERSONALIZACIÓN (robusta a typos, sin frases exactas)
    personalization = _detect_personalization_request_intent(norm)
    if personalization:
        return personalization

    # 8. ENCUESTA
    for sk in SURVEY_KEYWORDS:
        if _contains_keyword(norm, sk):
            return RouteResult(Intent.SURVEY_CONTINUE, 0.9, reason=f"Encuesta: '{sk}'")

    # 9. ACTUALIZACIÓN DE PERFIL (tiene número + unidad o campo mencionado)
    profile_update = _try_detect_profile_update(norm, numbers, units)
    if profile_update:
        return profile_update

    # 10. DUDA/PREGUNTA
    if norm.endswith("?") or any(_contains_keyword(norm, dw) for dw in DOUBT_WORDS):
        # Verificar si es pregunta nutricional
        has_nutrition = any(_contains_keyword(norm, nk) for nk in NUTRITION_KEYWORDS)
        if has_nutrition:
            return RouteResult(Intent.NUTRITION_QUERY, 0.85, reason="Pregunta nutricional detectada")
        return RouteResult(Intent.DOUBT, 0.7, reason="Pregunta genérica detectada")

    # 11. CONSULTA/RECOMENDACIÓN NUTRICIONAL
    nutrition_match = _detect_nutrition_intent(norm)
    if nutrition_match:
        return nutrition_match

    # 12. SMALL TALK (mensajes muy cortos sin contenido relevante)
    if len(norm.split()) <= 4 and not numbers:
        return RouteResult(Intent.SMALL_TALK, 0.6, reason="Mensaje corto sin datos")

    # 13. AMBIGUO — fallback a IA
    return RouteResult(Intent.AMBIGUOUS, 0.3, reason="No se pudo clasificar con confianza")


# ──────────────────────────────────────────────
# Sub-clasificadores
# ──────────────────────────────────────────────

def _detect_personalization_request_intent(norm: str) -> Optional[RouteResult]:
    if not norm:
        return None

    tokens = re.findall(r"[a-z0-9]+", norm)
    if not tokens:
        return None

    has_profile_signal = any(
        tok.startswith(PERSONALIZATION_PROFILE_ROOTS)
        or fuzzy_match(tok, "perfil", threshold=0.72)
        for tok in tokens
    )
    if not has_profile_signal:
        has_profile_signal = ("perfil" in norm) or ("nutric" in norm and "dato" in norm)

    has_action_signal = any(
        tok.startswith(PERSONALIZATION_ACTION_ROOTS)
        or fuzzy_match(tok, "actualizar", threshold=0.72)
        or fuzzy_match(tok, "personalizar", threshold=0.72)
        or fuzzy_match(tok, "completar", threshold=0.72)
        for tok in tokens
    )

    direct_compound = (
        ("perfil" in norm and ("actual" in norm or "personal" in norm or "complet" in norm))
        or ("mis datos" in norm and ("cambi" in norm or "actual" in norm))
    )

    if (has_profile_signal and has_action_signal) or direct_compound:
        return RouteResult(
            Intent.PERSONALIZE_REQUEST,
            0.9,
            reason="Personalización detectada por intención semántica",
        )
    return None

def _try_answer_current_step(
    norm: str,
    step: str,
    numbers: list[float],
    units: list[tuple[float, str]],
) -> Optional[RouteResult]:
    """
    Intenta interpretar el mensaje como respuesta al paso actual del onboarding.
    """
    step_lower = step.lower()

    # Pasos numéricos (edad, peso, talla)
    if step_lower in ("edad", "peso_kg", "altura_cm", "peso", "altura"):
        canonical = _canonical_field(step_lower)

        # Solo un número sin contexto → probablemente respuesta directa
        if len(numbers) == 1 and len(norm.split()) <= 4:
            val = numbers[0]

            # Detectar conflicto (ej: paso es edad pero dice "43 kilos")
            if units:
                unit_val, unit_type = units[0]
                detected_field = _field_from_unit(unit_type)
                if detected_field and detected_field != canonical:
                    return RouteResult(
                        Intent.ANSWER_CURRENT_STEP,
                        0.7,
                        resolved_field=detected_field,
                        resolved_value=str(unit_val),
                        reason=f"Respuesta con unidad '{unit_type}' conflicto con step '{step}' → campo real: {detected_field}",
                    )

            # Validar rango
            rng = FIELD_RANGES.get(canonical)
            if rng and rng[0] <= val <= rng[1]:
                return RouteResult(
                    Intent.ANSWER_CURRENT_STEP,
                    0.95,
                    resolved_field=canonical,
                    resolved_value=str(val),
                    reason=f"Número {val} válido para {canonical}",
                )

        # Número con unidad explícita que coincide con el step
        if units:
            for val, unit_type in units:
                detected_field = _field_from_unit(unit_type)
                if detected_field == canonical:
                    return RouteResult(
                        Intent.ANSWER_CURRENT_STEP,
                        0.95,
                        resolved_field=canonical,
                        resolved_value=str(val),
                        reason=f"Número con unidad {unit_type} para {canonical}",
                    )

    # Pasos de texto libre (alergias, enfermedades, etc.)
    if step_lower in ("alergias", "enfermedades", "tipo_dieta", "objetivo_nutricional",
                       "restricciones_alimentarias", "restricciones", "provincia", "distrito", "region"):
        canonical = _canonical_field(step_lower)

        # "Ninguna" / "Nada" / "No tengo"
        negation_patterns = ["ninguna", "nada", "no tengo", "ninguno", "no padezco", "no sufro"]
        for neg in negation_patterns:
            if neg in norm:
                return RouteResult(
                    Intent.ANSWER_CURRENT_STEP,
                    0.95,
                    resolved_field=canonical,
                    resolved_value="NINGUNA",
                    reason=f"Negación explícita para {canonical}",
                )

        # FILTRO ANTI-RUIDO: rechazar si es puro ruido conversacional
        words = set(norm.split())
        if words.issubset(_CONVERSATIONAL_NOISE):
            return None  # Delegar a LLM

        # FILTRO SEMÁNTICO: rechazar frases vacías de contenido
        if norm in _SEMANTIC_NOISE_PHRASES:
            return None  # Delegar a LLM

        # Mensaje corto sin pregunta → probablemente respuesta directa al campo
        if len(norm.split()) <= 8 and not norm.endswith("?"):
            return RouteResult(
                Intent.ANSWER_CURRENT_STEP,
                0.75,
                resolved_field=canonical,
                resolved_value=norm,
                reason=f"Respuesta de texto corta para {canonical}",
            )

    return None


def _try_detect_correction(
    norm: str,
    numbers: list[float],
    units: list[tuple[float, str]],
) -> Optional[RouteResult]:
    """Detecta correcciones de campos anteriores."""
    has_correction_signal = False
    for cp in CORRECTION_PATTERNS:
        if fuzzy_match(norm, cp, threshold=0.75):
            has_correction_signal = True
            break

    if not has_correction_signal:
        return None

    # Buscar qué campo se quiere corregir
    for field, aliases in PROFILE_FIELD_ALIASES.items():
        if any(_contains_keyword(norm, alias) for alias in aliases):
            # Buscar valor
            value = None
            if numbers:
                value = str(numbers[0])
            return RouteResult(
                Intent.CORRECTION_PAST_FIELD,
                0.85,
                resolved_field=field,
                resolved_value=value,
                reason=f"Corrección detectada para campo '{field}'",
            )

    # Corrección sin campo claro pero con número
    if numbers:
        return RouteResult(
            Intent.CORRECTION_PAST_FIELD,
            0.6,
            resolved_value=str(numbers[0]),
            reason="Corrección detectada con número pero sin campo claro",
        )

    return None


def _try_detect_profile_update(
    norm: str,
    numbers: list[float],
    units: list[tuple[float, str]],
) -> Optional[RouteResult]:
    """Detecta actualización de perfil fuera de onboarding."""
    # Patrón: "mi peso es 80", "mido 1.68", "tengo 43 años"
    for field, aliases in PROFILE_FIELD_ALIASES.items():
        matched_alias = None
        for alias in aliases:
            if _contains_keyword(norm, alias):
                matched_alias = alias
                break

        if matched_alias and numbers and field in ("edad", "peso_kg", "altura_cm"):
            return RouteResult(
                Intent.PROFILE_UPDATE,
                0.85,
                resolved_field=field,
                resolved_value=str(numbers[0]),
                reason=f"Actualización de perfil: '{matched_alias}' = {numbers[0]}",
            )

    # Patrón con unidades explícitas (extract_numbers también detecta estos números)
    if units:
        for val, unit_type in units:
            detected_field = _field_from_unit(unit_type)
            if detected_field:
                return RouteResult(
                    Intent.PROFILE_UPDATE,
                    0.8,
                    resolved_field=detected_field,
                    resolved_value=str(val),
                    reason=f"Unidad explícita '{unit_type}' = {val}",
                )

    # Patrones de texto libre: alergias, enfermedades, dieta, objetivo, restricciones.
    text_update = _try_detect_textual_profile_update(norm)
    if text_update:
        return text_update

    return None


def _try_detect_textual_profile_update(norm: str) -> Optional[RouteResult]:
    clauses = [c.strip() for c in re.split(r"[,\.;]+", norm) if c.strip()]
    if not clauses:
        clauses = [norm]

    for clause in clauses:
        # Alergias
        if "alerg" in clause or "intoleran" in clause:
            if "no tengo" in clause or "ninguna" in clause or "ninguno" in clause:
                return RouteResult(
                    Intent.PROFILE_UPDATE,
                    0.85,
                    resolved_field="alergias",
                    resolved_value="NINGUNA",
                    reason="Texto libre: alergias negadas",
                )
            m = re.search(r"(?:alerg(?:ia|ias)?|alergic[oa]|intoleran(?:cia|cias)?)\s+(?:a\s+la|al|a)?\s*(.+)", clause)
            if m:
                value = _clean_profile_value(m.group(1))
                if value:
                    return RouteResult(
                        Intent.PROFILE_UPDATE,
                        0.85,
                        resolved_field="alergias",
                        resolved_value=value,
                        reason="Texto libre: alergias detectadas",
                    )

        # Restricciones alimentarias
        if clause.startswith("no como ") or clause.startswith("evito ") or clause.startswith("no consumo ") or clause.startswith("no me gusta"):
            m = re.search(r"(?:no como|evito|no consumo|no me gusta(?:n)?)\s+(.+)", clause)
            if m:
                value = _clean_profile_value(m.group(1))
                if value:
                    return RouteResult(
                        Intent.PROFILE_UPDATE,
                        0.85,
                        resolved_field="restricciones_alimentarias",
                        resolved_value=value,
                        reason="Texto libre: restricciones detectadas",
                    )

        # Tipo de dieta
        for diet_key, canonical in DIET_PATTERNS.items():
            if _contains_keyword(clause, diet_key) and ("soy" in clause or "dieta" in clause or "sigo" in clause):
                return RouteResult(
                    Intent.PROFILE_UPDATE,
                    0.85,
                    resolved_field="tipo_dieta",
                    resolved_value=canonical,
                    reason=f"Texto libre: tipo de dieta '{diet_key}'",
                )

        # Enfermedades
        if "enfermedad" in clause or "padezco" in clause or "sufro" in clause or any(h in clause for h in DISEASE_HINTS):
            if "no tengo" in clause or "ninguna" in clause or "ninguno" in clause:
                return RouteResult(
                    Intent.PROFILE_UPDATE,
                    0.85,
                    resolved_field="enfermedades",
                    resolved_value="NINGUNA",
                    reason="Texto libre: enfermedades negadas",
                )
            m = re.search(
                r"(?:padezco|sufro(?: de)?|me diagnosticaron|tengo"
                r"|mi enfermedad es|mi condicion es)"
                r"\s+(?:la\s+|una?\s+|de\s+)?(?:enfermedad\s+|condicion\s+)?(.+)",
                clause,
            )
            if m:
                value = _clean_profile_value(m.group(1))
                # Limpiar prefijos genéricos redundantes
                value = re.sub(r"^(?:enfermedad|condicion|condición)\s+", "", value).strip()
                if value:
                    return RouteResult(
                        Intent.PROFILE_UPDATE,
                        0.85,
                        resolved_field="enfermedades",
                        resolved_value=value,
                        reason="Texto libre: enfermedades detectadas",
                    )

        # Objetivo nutricional
        objective_hint_words = (
            "bajar", "subir", "ganar", "masa", "muscular", "peso",
            "habito", "salud", "saludable", "controlar", "mejorar",
            "reducir", "aumentar", "grasa", "definir",
        )
        is_objective_sentence = ("objetivo" in clause or "meta" in clause)
        starts_goal_verb = clause.startswith("quiero ") or clause.startswith("busco ")
        if starts_goal_verb and not any(w in clause for w in objective_hint_words):
            starts_goal_verb = False
        if starts_goal_verb and any(k in clause for k in ("receta", "menu", "menú", "cena", "almuerzo", "desayuno", "comida")):
            starts_goal_verb = False

        if is_objective_sentence or starts_goal_verb:
            m = re.search(r"(?:mi objetivo(?: principal)? es|quiero|busco)\s+(.+)", clause)
            if m:
                value = _clean_profile_value(m.group(1))
                if value and not value.startswith("que ") and not value.startswith("qué "):
                    return RouteResult(
                        Intent.PROFILE_UPDATE,
                        0.85,
                        resolved_field="objetivo_nutricional",
                        resolved_value=value,
                        reason="Texto libre: objetivo nutricional detectado",
                    )

    return None


def _clean_profile_value(value: str) -> str:
    cleaned = (value or "").strip(" :")
    # Cortar coletillas frecuentes que no aportan al valor de perfil.
    cleaned = re.sub(r"\s+(por favor|gracias|pls|plz)$", "", cleaned).strip()
    return cleaned


def _detect_nutrition_intent(norm: str) -> Optional[RouteResult]:
    """Detecta si el mensaje es una consulta nutricional."""
    match_count = sum(1 for nk in NUTRITION_KEYWORDS if _contains_keyword(norm, nk))

    if match_count >= 2:
        return RouteResult(
            Intent.RECOMMENDATION_REQUEST,
            0.9,
            reason=f"Múltiples keywords nutricionales ({match_count})",
        )
    elif match_count == 1:
        short_request_markers = ("receta", "menu", "dieta", "desayuno", "almuerzo", "cena", "comida")
        if any(_contains_keyword(norm, marker) for marker in short_request_markers):
            return RouteResult(
                Intent.RECOMMENDATION_REQUEST,
                0.82,
                reason="Solicitud nutricional corta detectada",
            )
        if len(norm.split()) > 4:
            return RouteResult(
                Intent.NUTRITION_QUERY,
                0.75,
                reason="Keyword nutricional con contexto",
            )

    return None


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────

def _canonical_field(step: str) -> str:
    """Normaliza un step/alias a su nombre canónico de campo."""
    mapping = {
        "peso": "peso_kg",
        "altura": "altura_cm",
        "talla": "altura_cm",
        "restricciones": "restricciones_alimentarias",
        "objetivo": "objetivo_nutricional",
        "meta": "objetivo_nutricional",
    }
    return mapping.get(step, step)


def _field_from_unit(unit: str) -> Optional[str]:
    """Infiere el campo de perfil a partir de una unidad detectada."""
    unit_to_field = {
        "kg": "peso_kg",
        "lbs": "peso_kg",
        "cm": "altura_cm",
        "m": "altura_cm",
        "years": "edad",
    }
    return unit_to_field.get(unit)
