"""
Nutribot Backend â€” ProfileExtractionService
Servicio de AplicaciÃ³n Orientado a Objetos para extraer perfil.
"""
import asyncio
import json
import logging
import math
import re
import unicodedata
from datetime import timedelta
from difflib import SequenceMatcher
from time import perf_counter
from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from domain.parsers import parse_weight, parse_height, parse_age, standardize_text_list
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    clean_data: dict[str, Any]
    updates: dict[str, Any]
    meta_flags: dict[str, Any]

class ProfileExtractionService:
    OP_CLEAR = "CLEAR"
    OP_REPLACE = "REPLACE"
    OP_ADD = "ADD"
    OP_REMOVE = "REMOVE"
    MEASUREMENT_CORRECTION = "CORRECTION"
    MEASUREMENT_HISTORICAL_UPDATE = "HISTORICAL_UPDATE"
    SEMANTIC_SCOPE_PROFILE_FIELD = "PROFILE_FIELD"
    SEMANTIC_EMBED_MODEL = "text-embedding-3-small"
    SEMANTIC_LLM_FALLBACK_MIN_SCORE = 0.55
    SEMANTIC_ENTITY_TYPE_BY_TABLE = {
        "mae_enfermedad_cie10": "ENFERMEDAD_CIE10",
        "mae_grupo_nutricional": "GRUPO_NUTRICIONAL",
        "mae_restriccion_alimentaria": "RESTRICCION_ALIMENTARIA",
        "mae_patron_alimentario": "PATRON_ALIMENTARIO",
        "mae_dieta_terapeutica": "DIETA_TERAPEUTICA",
        "mae_textura_dieta": "TEXTURA_DIETA",
        "mae_objetivo_nutricional": "OBJETIVO_NUTRICIONAL",
        "mae_distrito": "DISTRITO",
        "mae_provincia": "PROVINCIA",
        "mae_departamento": "DEPARTAMENTO",
    }

    ABSURD_TERMS = {
        "sayayin",
        "saiyajin",
        "super sayayin",
        "goku",
        "kamehameha",
        "naruto",
        "marciano",
        "extraterrestre",
        "alienigena",
        "avenger",
        "pokemon",
        "aire",
        "piedras",
        "tierra",
        "nada",
        "humo",
        "invisible",
        "fantasmas",
        "mentiras",
    }

    NEGATIVE_MARKERS = {
        "ninguna",
        "ninguno",
        "nada",
        "no tengo",
        "no padezco",
        "sin",
    }

    CORRECTION_MARKERS = (
        "me equivoque",
        "corrijo",
        "correccion",
        "quise decir",
        "no era",
        "no, era",
        "en realidad",
        "mas bien",
    )

    CHANGE_OVER_TIME_MARKERS = (
        "ahora peso",
        "ahora mido",
        "he bajado",
        "he subido",
        "subi a",
        "baje a",
        "ultimo peso",
        "mi ultimo peso",
    )

    REMOVE_MARKERS = (
        "ya no",
        "quita",
        "quitar",
        "elimina",
        "eliminar",
        "borra",
        "saca",
        "remueve",
        "remove",
    )

    ADD_MARKERS = (
        "agrega",
        "agregar",
        "anade",
        "suma",
        "tambien",
        "ademas",
    )

    REPLACE_MARKERS = (
        "mis ",
        "son ",
        "ahora son",
        "actualiza",
        "actualizar",
        "reemplaza",
        "cambia a",
    )

    _NON_INFORMATIVE_LIST_VALUES = {
        "cosas",
        "varias cosas",
        "muchas cosas",
        "de todo",
        "todo",
        "etc",
        "etcetera",
    }

    _RESTRICTION_CANONICAL_HINTS: tuple[tuple[str, str], ...] = (
        (r"\blacte[oa]s?\b|\blacteos\b|\blacteo\b|\bleche\b|\blactosa\b", "lactosa"),
        (r"\bmarisc(?:o|os)\b|\bcrustace(?:o|os)\b|\bcamaron(?:es)?\b|\bgamba(?:s)?\b", "mariscos"),
        (r"\bmani\b|\bcacahuate(?:s)?\b|\bcacahuete(?:s)?\b|\bpeanut(?:s)?\b", "mani"),
        (r"\bgluten\b|\btrigo\b|\bcebada\b|\bcenteno\b", "gluten"),
        (r"\bpescado(?:s)?\b|\bpez\b", "pescado"),
        (r"\bfructosa\b", "fructosa"),
        (r"\bhistamina\b", "histamina"),
    )

    FIELD_CONFIG = {
        "edad": {"col": "edad", "parser": parse_age},
        "years": {"col": "edad", "parser": parse_age},
        "mi_edad": {"col": "edad", "parser": parse_age},
        
        "peso": {"col": "peso_kg", "parser": parse_weight},
        "peso_kg": {"col": "peso_kg", "parser": parse_weight},
        "kg": {"col": "peso_kg", "parser": parse_weight},
        "peso_actual": {"col": "peso_kg", "parser": parse_weight},
        "mi_peso": {"col": "peso_kg", "parser": parse_weight},
        
        "altura_cm": {"col": "altura_cm", "parser": parse_height},
        "talla": {"col": "altura_cm", "parser": parse_height},
        "altura": {"col": "altura_cm", "parser": parse_height},
        "cm": {"col": "altura_cm", "parser": parse_height},
        "talla_actual": {"col": "altura_cm", "parser": parse_height},
        "mi_talla": {"col": "altura_cm", "parser": parse_height},
        
        "alergias": {"col": "alergias", "parser": standardize_text_list},
        "alergia": {"col": "alergias", "parser": standardize_text_list},
        "allergies": {"col": "alergias", "parser": standardize_text_list},
        "mis_alergias": {"col": "alergias", "parser": standardize_text_list},
        
        "enfermedades": {"col": "enfermedades", "parser": standardize_text_list},
        "enfermedad": {"col": "enfermedades", "parser": standardize_text_list},
        "condicion": {"col": "enfermedades", "parser": standardize_text_list},
        "patologia": {"col": "enfermedades", "parser": standardize_text_list},
        "mis_enfermedades": {"col": "enfermedades", "parser": standardize_text_list},
        
        "tipo_dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
        "dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
        "mi_dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
        
        "objetivo_nutricional": {"col": "objetivo_nutricional", "parser": standardize_text_list},
        "objetivo": {"col": "objetivo_nutricional", "parser": standardize_text_list},
        "meta": {"col": "objetivo_nutricional", "parser": standardize_text_list},
        "objetivo_actual": {"col": "objetivo_nutricional", "parser": standardize_text_list},

        "restricciones_alimentarias": {"col": "restricciones_alimentarias", "parser": standardize_text_list},
        "restricciones": {"col": "restricciones_alimentarias", "parser": standardize_text_list},
        "restriccion": {"col": "restricciones_alimentarias", "parser": standardize_text_list},
        
        "provincia": {"col": "provincia", "parser": lambda x: x.upper() if x else None},
        "distrito": {"col": "distrito", "parser": lambda x: x.upper() if x else None},
        "region": {"col": "region", "parser": lambda x: x.upper() if x else None},
    }

    ONBOARDING_STEP_TO_FIELD = {
        "edad": "edad",
        "peso": "peso_kg",
        "peso_kg": "peso_kg",
        "altura": "altura_cm",
        "altura_cm": "altura_cm",
        "talla": "altura_cm",
        "alergias": "alergias",
        "enfermedades": "enfermedades",
        "restricciones": "restricciones_alimentarias",
        "restricciones_alimentarias": "restricciones_alimentarias",
        "tipo_dieta": "tipo_dieta",
        "objetivo": "objetivo_nutricional",
        "objetivo_nutricional": "objetivo_nutricional",
        "provincia": "provincia",
        "distrito": "distrito",
        "region": "region",
    }

    FIELD_INTENT_HINTS = {
        "edad": ("edad", "anos", "cumpli"),
        "peso_kg": ("peso", "kilo", "kg", "libr"),
        "altura_cm": ("talla", "altura", "estatura", "cm", "metro", "mido"),
        "alergias": ("alerg", "intoler", "mani", "lact", "marisc", "gluten", "crustace", "fructosa", "histamina"),
        "enfermedades": ("diabet", "hipert", "hipotiro", "gastrit", "anemia", "colesterol", "enfermedad", "condicion"),
        "restricciones_alimentarias": ("no como", "evito", "restric", "sin ", "intoler", "alerg"),
        "tipo_dieta": ("veg", "omniv", "keto", "carniv", "dieta", "patron aliment"),
        "objetivo_nutricional": ("objetivo", "meta", "bajar", "subir", "ganar", "masa", "mejorar", "habito", "controlar", "mantener"),
        "provincia": ("provincia", "lima", "arequipa", "cusco", "trujillo", "piura"),
        "distrito": ("distrito", "miraflores", "san miguel", "surco", "cayma", "wanchaq"),
        "region": ("region", "departamento"),
    }

    EXTRACTION_SYSTEM_PROMPT = """Eres un Analista de Datos experto en COMPRENDER la intenciÃ³n del usuario para Nutribot.
REGLAS CRITICAS DE ROBUSTEZ:
1. PRIORIDAD ABSOLUTA: Si el usuario menciona un dato (ej: 'mido 1.71', 'mi peso es 80'), EXTRAELO siempre.
2. ESCUDO CONTRA DUDAS: Si el usuario hace una PREGUNTA o expresa confusion (ej: 'Como?', 'Por que?', 'Que es?', 'no entiendo', '??', 'que alergias?'), NO extraigas nada. Devuelve un objeto vacio {}.
3. PESIMISMO OPERATIVO: Ante la menor duda de si el texto es un dato o una consulta, NO extraigas. Es mejor preguntar de nuevo que anotar basura.
4. RESTRICCIONES ALIMENTARIAS: Solo extrae si hay una negacion explicita (ej: 'no me gusta el X', 'no como Y', 'no soporto Z', 'evito el W'). Extrae solo el alimento (X, Y, Z, W).
5. ALERGIAS vs RESTRICCIONES: Si el usuario dice 'soy alergico a X', es una alergia. Si dice 'no me gusta X', es una restriccion_alimentaria.
6. PETICIONES vs DATOS (REGLA DE ORO): Si el usuario PIDE algo (ej: 'dame una receta de pescado'), NO extraigas ese alimento como dato de perfil.
7. NEGACION TOTAL: Si el usuario dice 'ninguna', 'nada', 'no tengo' a una pregunta sobre salud/alergia/restriccion, extrae 'NINGUNA'.
8. COHERENCIA BIOLOGICA Y MEDICA:
   - RECHAZA datos absurdos (ej: 'alergia al aire', 'enfermedad marciana', 'diabetes tipo T').
   - RECHAZA metricas imposibles (ej: altura < 50cm o > 250cm, peso < 2kg o > 400kg para adultos).
9. FORMATO: Responde SOLO un objeto JSON PLANO. Si no hay datos claros, responde {}.
10. PROHIBICION DE PREGUNTAS: Si el texto termina en '?' o comienza con palabras interrogativas ('como', 'que', 'por que', 'para que', 'cual', 'cuanto'), responde SIEMPRE con un objeto vacio {}. NO intentes salvar datos de una pregunta.
"""

    # (pattern, prompt de aclaracion interactivo)
    _AMBIGUOUS_CONDITIONS = [
        ("diabetes", "te refieres a tipo 1, tipo 2 o gestacional? (Si no lo sabes, no te preocupes, solo dime 'no' o 'solo diabetes')"),
        ("anemia", "te refieres a algún tipo específico de anemia (ej: ferropénica)? Si no lo sabes, no hay problema, solo dime 'no' o 'solo anemia'"),
        ("tiroides", "te refieres a hipotiroidismo o hipertiroidismo? Si no estás seguro, solo dime 'no' o 'solo tiroides'"),
        ("problemas hormonales", "te refieres a algún problema hormonal específico? Si no lo sabes, dime 'solo eso' o 'no se'"),
        ("colon", "te refieres a colitis, colon irritable u otra condición? Si no sabes, solo dime 'no'"),
        ("presion", "te refieres a hipertensión o hipotensión? Si no estás seguro, solo dime 'no'"),
        ("gastritis", "te refieres a gastritis aguda o crónica? Si no lo sabes, solo dime 'no'"),
    ]

    # Palabras que indican especificacion suficiente (no necesitan aclaracion)
    _SPECIFICITY_MARKERS = [
        "tipo 1", "tipo 2", "tipo i", "tipo ii", "gestacional",
        "hipertiroidismo", "hipotiroidismo",
        "hipertension", "hipotension",
        "irritable", "colitis", "crohn",
        "aguda", "cronica",
        "ferropenica", "megaloblastica",
    ]

    def __init__(self, openai_client: AsyncOpenAI, model: str, nutritional_rules=None):
        self._openai_client = openai_client
        self._model = model
        self._nutritional_rules = nutritional_rules

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        txt = unicodedata.normalize("NFKD", str(value))
        txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
        txt = txt.lower()
        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    @classmethod
    def _is_negative_value(cls, value: Optional[str]) -> bool:
        norm = cls._normalize_text(value)
        if not norm:
            return False
        if norm in cls.NEGATIVE_MARKERS:
            return True
        if norm.startswith("no ") and len(norm.split()) <= 4:
            return True
        return False

    @classmethod
    def _contains_any_marker(cls, source_text: str, markers: tuple[str, ...]) -> bool:
        norm = cls._normalize_text(source_text)
        if not norm:
            return False
        return any(cls._normalize_text(marker) in norm for marker in markers)

    @classmethod
    def _step_matches_field(cls, current_step: Optional[str], field_code: str) -> bool:
        if not current_step:
            return False
        step = cls._normalize_text(current_step)
        field = cls._normalize_text(field_code)
        aliases = {
            "peso_kg": {"peso", "peso kg", "peso_kg"},
            "altura_cm": {"altura", "talla", "altura cm", "altura_cm"},
            "alergias": {"alergias", "alergia"},
            "enfermedades": {"enfermedades", "enfermedad", "condicion"},
            "restricciones_alimentarias": {"restricciones", "restriccion", "restricciones alimentarias"},
            "tipo_dieta": {"tipo_dieta", "tipo dieta", "dieta", "patron alimentario"},
            "objetivo_nutricional": {"objetivo", "meta", "objetivo_nutricional"},
            "provincia": {"provincia"},
            "distrito": {"distrito"},
            "region": {"region", "departamento"},
        }
        valid = aliases.get(field, {field})
        return step in valid

    @classmethod
    def _expected_field_for_step(cls, current_step: Optional[str]) -> Optional[str]:
        if not current_step:
            return None
        return cls.ONBOARDING_STEP_TO_FIELD.get(cls._normalize_text(current_step))

    @classmethod
    def _text_explicitly_mentions_field(cls, source_text: str, field_code: Optional[str]) -> bool:
        if not source_text or not field_code:
            return False
        norm = cls._normalize_text(source_text)
        if not norm:
            return False
        hints = cls.FIELD_INTENT_HINTS.get(field_code, ())
        return any(h in norm for h in hints if h)

    @classmethod
    def _guess_field_from_text(cls, source_text: str) -> Optional[str]:
        norm = cls._normalize_text(source_text)
        if not norm:
            return None
        best_field: Optional[str] = None
        best_score = 0
        for field_code, hints in cls.FIELD_INTENT_HINTS.items():
            score = 0
            for hint in hints:
                if hint and hint in norm:
                    score += 1
            if score > best_score:
                best_score = score
                best_field = field_code
        if best_score <= 0:
            return None
        return best_field

    def _classify_measurement_update(
        self,
        *,
        source_text: str,
        field_code: str,
        current_step: Optional[str],
        current_measurement_row: Optional[dict[str, Any]],
        now_peru,
    ) -> str:
        has_correction_marker = self._contains_any_marker(source_text, self.CORRECTION_MARKERS)
        has_time_change_marker = self._contains_any_marker(source_text, self.CHANGE_OVER_TIME_MARKERS)
        matches_step = self._step_matches_field(current_step, field_code)

        recent_current = False
        if current_measurement_row and current_measurement_row.get("fecha_medicion"):
            try:
                delta = now_peru - current_measurement_row["fecha_medicion"]
                recent_current = delta.total_seconds() <= 20 * 60
            except Exception:
                recent_current = False

        # Regla 1: si menciona explicitamente correccion, tratamos como correction.
        if has_correction_marker and not has_time_change_marker:
            return self.MEASUREMENT_CORRECTION

        # Regla 2: si menciona cambio temporal ("ahora peso", "he bajado"), tratamos como historico.
        if has_time_change_marker and not has_correction_marker:
            return self.MEASUREMENT_HISTORICAL_UPDATE

        # Regla 3: respuesta al mismo paso en onboarding o dato recien registrado: correccion.
        if matches_step or recent_current:
            return self.MEASUREMENT_CORRECTION

        # Regla 4: por defecto fuera de onboarding, consideramos actualizacion historica.
        return self.MEASUREMENT_HISTORICAL_UPDATE

    def _infer_list_operation(
        self,
        *,
        raw_value: str,
        source_text: str,
        field_code: str,
        current_step: Optional[str],
    ) -> str:
        if self._is_negative_value(raw_value):
            return self.OP_CLEAR

        if self._contains_any_marker(source_text, self.REMOVE_MARKERS):
            return self.OP_REMOVE
        if self._contains_any_marker(source_text, self.ADD_MARKERS):
            return self.OP_ADD
        if self._contains_any_marker(source_text, self.CORRECTION_MARKERS):
            return self.OP_REPLACE
        if self._contains_any_marker(source_text, self.REPLACE_MARKERS):
            return self.OP_REPLACE
        if self._step_matches_field(current_step, field_code):
            return self.OP_REPLACE
        return self.OP_ADD

    @classmethod
    def _split_values(cls, value: str) -> list[str]:
        if not value:
            return []
        normalized = str(value).replace("\n", ",")
        parts = re.split(r"\s*(?:,|;|/|\by\b)\s*", normalized, flags=re.IGNORECASE)
        out: list[str] = []
        seen: set[str] = set()
        for raw in parts:
            candidate = raw.strip(" .:-")
            if not candidate:
                continue
            # Limpia muletillas y prefijos semanticos de forma iterativa.
            while True:
                prev = candidate
                candidate = re.sub(
                    r"^(?:y|e|ahora|encima|tambien|ademas|otra vez|de nuevo)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip(" .:-")
                candidate = re.sub(
                    r"^(?:como(?:\s+a(?:l| la| los| las)?)?)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip(" .:-")
                candidate = re.sub(
                    r"^(?:ya no|no|tengo|padezco|sufro de|soy alergic[oa]\s+a(?:l| la| los| las)?|alergia\s+a(?:l| la| los| las)?|evito|no como|no consumo)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip(" .:-")
                candidate = re.sub(
                    r"^(?:a la|a los|a las|al|a|la|el|los|las)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip(" .:-")
                if candidate == prev:
                    break
            candidate = re.sub(r"\b(?:por favor|gracias)$", "", candidate, flags=re.IGNORECASE).strip(" .:-")
            candidate = re.sub(
                r"\b(?:tambien|nomas|nomasito|otra vez|de nuevo)$",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip(" .:-")
            if not candidate:
                continue
            key = cls._normalize_text(candidate)
            if key in cls._NON_INFORMATIVE_LIST_VALUES:
                continue
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

    @classmethod
    def _restriction_resolution_candidates(cls, value: str) -> list[str]:
        raw = (value or "").strip()
        if not raw:
            return []
        normalized = cls._normalize_text(raw)
        candidates: list[str] = [raw]
        seen = {normalized}
        for pattern, canonical in cls._RESTRICTION_CANONICAL_HINTS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                canonical_key = cls._normalize_text(canonical)
                if canonical_key and canonical_key not in seen:
                    candidates.append(canonical)
                    seen.add(canonical_key)
        return candidates

    @classmethod
    def _score_master_candidate(cls, target_norm: str, code: Optional[str], name: Optional[str]) -> float:
        code_norm = cls._normalize_text(code)
        name_norm = cls._normalize_text(name)
        if not target_norm:
            return 0.0
        if target_norm == code_norm or target_norm == name_norm:
            return 1.0
        if code_norm and (target_norm in code_norm or code_norm in target_norm):
            return 0.92
        if name_norm and (target_norm in name_norm or name_norm in target_norm):
            overlap = min(len(target_norm), len(name_norm))
            if overlap >= 5:
                return 0.9
        return 0.0

    @staticmethod
    def _build_embedding_literal(values: list[float]) -> Optional[str]:
        if not values:
            return None
        cleaned: list[str] = []
        for val in values:
            try:
                num = float(val)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            cleaned.append(f"{num:.12g}")
        return "[" + ",".join(cleaned) + "]"

    async def _semantic_cache_get(
        self,
        session: AsyncSession,
        *,
        field_code: str,
        query_normalized: str,
        increment_hit: bool = True,
    ) -> Optional[dict[str, Any]]:
        row = (
            await session.execute(
                text(
                    """
                    SELECT id, entidad_tipo_resuelta, entidad_codigo_resuelto, estrategia_usada, confidence, expires_at
                    FROM semantic_resolution_cache
                    WHERE scope = :scope
                      AND field_code = :field
                      AND query_normalizada = :qnorm
                      AND (expires_at IS NULL OR expires_at > :now)
                    ORDER BY actualizado_en DESC NULLS LAST, id DESC
                    LIMIT 1
                    """
                ),
                {
                    "scope": self.SEMANTIC_SCOPE_PROFILE_FIELD,
                    "field": field_code,
                    "qnorm": query_normalized,
                    "now": get_now_peru(),
                },
            )
        ).mappings().first()
        if not row:
            return None
        if increment_hit:
            try:
                await session.execute(
                    text(
                        """
                        UPDATE semantic_resolution_cache
                        SET hit_count = COALESCE(hit_count, 0) + 1,
                            actualizado_en = :now
                        WHERE id = :id
                        """
                    ),
                    {"id": row.get("id"), "now": get_now_peru()},
                )
            except Exception:
                logger.debug("No se pudo incrementar hit_count de semantic_resolution_cache id=%s", row.get("id"))
        return dict(row)

    async def _semantic_cache_put(
        self,
        session: AsyncSession,
        *,
        field_code: str,
        raw_query: str,
        query_normalized: str,
        entity_type: Optional[str],
        entity_code: Optional[str],
        strategy: str,
        confidence: float,
        top_candidates: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        now_peru = get_now_peru()
        expires_at = now_peru + timedelta(days=180)
        await session.execute(
            text(
                """
                INSERT INTO semantic_resolution_cache (
                    scope, field_code, query_texto, query_normalizada,
                    entidad_tipo_resuelta, entidad_codigo_resuelto,
                    estrategia_usada, confidence, compatible_con_reglas,
                    top_candidates_json, hit_count, expires_at, creado_en, actualizado_en
                )
                VALUES (
                    :scope, :field, :qtext, :qnorm,
                    :etype, :ecode,
                    :strategy, :confidence, TRUE,
                    CAST(:candidates AS jsonb), 1, :expires_at, :now, :now
                )
                ON CONFLICT (scope, field_code, query_normalizada) DO UPDATE SET
                    entidad_tipo_resuelta = EXCLUDED.entidad_tipo_resuelta,
                    entidad_codigo_resuelto = EXCLUDED.entidad_codigo_resuelto,
                    estrategia_usada = EXCLUDED.estrategia_usada,
                    confidence = EXCLUDED.confidence,
                    top_candidates_json = EXCLUDED.top_candidates_json,
                    expires_at = EXCLUDED.expires_at,
                    actualizado_en = EXCLUDED.actualizado_en
                """
            ),
            {
                "scope": self.SEMANTIC_SCOPE_PROFILE_FIELD,
                "field": field_code,
                "qtext": raw_query,
                "qnorm": query_normalized,
                "etype": entity_type,
                "ecode": entity_code,
                "strategy": strategy,
                "confidence": float(max(0.0, min(confidence, 1.0))),
                "candidates": json.dumps(top_candidates or [], ensure_ascii=False),
                "expires_at": expires_at,
                "now": now_peru,
            },
        )

    async def _semantic_cache_peek(
        self,
        session: AsyncSession,
        *,
        field_code: str,
        raw_query: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if not raw_query:
            return None
        query_norm = self._normalize_text(raw_query)
        if not query_norm:
            return None
        try:
            return await self._semantic_cache_get(
                session,
                field_code=field_code,
                query_normalized=query_norm,
                increment_hit=False,
            )
        except DBAPIError:
            logger.debug("semantic_cache_peek omitido por transaccion no limpia", exc_info=True)
            return None

    async def _log_semantic_match(
        self,
        session: AsyncSession,
        *,
        usuario_id: Optional[int],
        incoming_message_id: Optional[int],
        field_code: str,
        query_text: str,
        query_normalized: str,
        strategy: str,
        exact_match: bool,
        trigram_score: Optional[float],
        vector_score: Optional[float],
        confidence: Optional[float],
        entity_type: Optional[str],
        entity_code: Optional[str],
        escalated_to_ai: bool,
        decided_by_rule: bool,
        latency_ms: int,
        error_detail: Optional[str] = None,
    ) -> None:
        if not query_text or not query_normalized:
            return
        try:
            # Logging semantico no debe abortar el turno si falla.
            async with session.begin_nested():
                await session.execute(
                    text(
                        """
                        INSERT INTO semantic_match_log (
                            usuario_id, incoming_message_id, scope, field_code,
                            query_texto, query_normalizada, estrategia_usada,
                            exact_match, trigram_score, vector_score, confidence_final,
                            entidad_tipo_resuelta, entidad_codigo_resuelto,
                            escalado_a_ia, decidido_por_regla, latency_ms, error_detail, creado_en
                        )
                        VALUES (
                            :uid, :imid, :scope, :field,
                            :qtext, :qnorm, :strategy,
                            :exact_match, :trgm, :vec, :confidence,
                            :etype, :ecode,
                            :escalado, :regla, :latency, :error, :now
                        )
                        """
                    ),
                    {
                        "uid": usuario_id,
                        "imid": incoming_message_id,
                        "scope": self.SEMANTIC_SCOPE_PROFILE_FIELD,
                        "field": field_code,
                        "qtext": query_text,
                        "qnorm": query_normalized,
                        "strategy": strategy[:20],
                        "exact_match": bool(exact_match),
                        "trgm": None if trigram_score is None else round(float(trigram_score), 4),
                        "vec": None if vector_score is None else round(float(vector_score), 4),
                        "confidence": None if confidence is None else round(max(0.0, min(float(confidence), 1.0)), 2),
                        "etype": entity_type,
                        "ecode": entity_code,
                        "escalado": bool(escalated_to_ai),
                        "regla": bool(decided_by_rule),
                        "latency": int(max(0, latency_ms)),
                        "error": (error_detail or None),
                        "now": get_now_peru(),
                    },
                )
        except Exception:
            logger.debug("No se pudo insertar semantic_match_log", exc_info=True)

    async def _enqueue_semantic_review(
        self,
        session: AsyncSession,
        *,
        usuario_id: Optional[int],
        incoming_message_id: Optional[int],
        field_code: str,
        query_text: str,
        query_normalized: str,
        reason: str,
        top_candidates: Optional[list[dict[str, Any]]] = None,
        observation: Optional[str] = None,
    ) -> None:
        if not query_text or not query_normalized:
            return
        payload = json.dumps(top_candidates or [], ensure_ascii=False)
        try:
            # Cola de revision semantica no debe abortar la transaccion principal.
            # Se usa chequeo explicito + insert para evitar edge-cases de SQL no insertado.
            async with session.begin_nested():
                exists = (
                    await session.execute(
                        text(
                            """
                            SELECT 1
                            FROM semantic_review_queue q
                            WHERE q.scope = :scope
                              AND q.field_code = :field
                              AND q.query_normalizada = :qnorm
                              AND q.estado = 'PENDING'
                            LIMIT 1
                            """
                        ),
                        {
                            "scope": self.SEMANTIC_SCOPE_PROFILE_FIELD,
                            "field": field_code,
                            "qnorm": query_normalized,
                        },
                    )
                ).first()
                if exists:
                    return

                await session.execute(
                    text(
                        """
                        INSERT INTO semantic_review_queue (
                            usuario_id, incoming_message_id, scope, field_code,
                            query_texto, query_normalizada, top_candidates_json,
                            razon, estado, observacion, creado_en
                        )
                        VALUES (
                            :uid, :imid, :scope, :field,
                            :qtext, :qnorm, CAST(:candidates AS jsonb),
                            :reason, 'PENDING', :obs, :now
                        )
                        """
                    ),
                    {
                        "uid": usuario_id,
                        "imid": incoming_message_id,
                        "scope": self.SEMANTIC_SCOPE_PROFILE_FIELD,
                        "field": field_code,
                        "qtext": query_text,
                        "qnorm": query_normalized,
                        "candidates": payload,
                        "reason": (reason or "NO_MATCH")[:50],
                        "obs": observation,
                        "now": get_now_peru(),
                    },
                )
        except Exception:
            logger.debug("No se pudo insertar semantic_review_queue", exc_info=True)

    async def _enqueue_embedding_job(
        self,
        session: AsyncSession,
        *,
        source_table: str,
        source_id: int,
        source_text: str,
        model: Optional[str] = None,
    ) -> None:
        if not source_table or not source_id or not source_text:
            return
        now_peru = get_now_peru()
        model_name = model or self.SEMANTIC_EMBED_MODEL
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO embedding_jobs (
                        source_table, source_id, texto_fuente, modelo,
                        estado, retry_count, error_detail, creado_en, actualizado_en, procesado_en
                    )
                    VALUES (
                        :stable, :sid, :stext, :model,
                        'PENDING', 0, NULL, :now, :now, NULL
                    )
                    ON CONFLICT (source_table, source_id, modelo)
                    DO UPDATE SET
                        texto_fuente = EXCLUDED.texto_fuente,
                        estado = 'PENDING',
                        error_detail = NULL,
                        actualizado_en = EXCLUDED.actualizado_en,
                        procesado_en = NULL
                    """
                ),
                {
                    "stable": source_table[:50],
                    "sid": int(source_id),
                    "stext": str(source_text)[:5000],
                    "model": model_name[:100],
                    "now": now_peru,
                },
            )
        except Exception:
            logger.debug("No se pudo encolar embedding_jobs %s/%s", source_table, source_id, exc_info=True)

    async def _upsert_semantic_catalog_entry(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        entity_code: Optional[str],
        canonical_name: Optional[str],
        search_text: Optional[str],
        alias_semantico_id: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        if not entity_type or not entity_code or not canonical_name or not search_text:
            return None
        text_norm = self._normalize_text(search_text)[:255]
        if not text_norm:
            return None
        now_peru = get_now_peru()
        try:
            row = (
                await session.execute(
                    text(
                        """
                        INSERT INTO semantic_catalog (
                            entidad_tipo, entidad_codigo, alias_semantico_id,
                            nombre_canonico, texto_busqueda, texto_normalizado,
                            peso_lexico, peso_semantico, activo, generado_en, actualizado_en
                        )
                        VALUES (
                            :etype, :ecode, :alias_id,
                            :cname, :stext, :snorm,
                            1.0, 1.0, TRUE, :now, :now
                        )
                        ON CONFLICT (entidad_tipo, entidad_codigo, texto_normalizado)
                        DO UPDATE SET
                            alias_semantico_id = COALESCE(EXCLUDED.alias_semantico_id, semantic_catalog.alias_semantico_id),
                            nombre_canonico = EXCLUDED.nombre_canonico,
                            texto_busqueda = EXCLUDED.texto_busqueda,
                            activo = TRUE,
                            actualizado_en = EXCLUDED.actualizado_en
                        RETURNING id, embedding, texto_busqueda
                        """
                    ),
                    {
                        "etype": entity_type[:30],
                        "ecode": str(entity_code)[:60],
                        "alias_id": alias_semantico_id,
                        "cname": str(canonical_name)[:255],
                        "stext": str(search_text)[:255],
                        "snorm": text_norm,
                        "now": now_peru,
                    },
                )
            ).mappings().first()
            return dict(row) if row else None
        except Exception:
            logger.debug("No se pudo upsert semantic_catalog etype=%s code=%s", entity_type, entity_code, exc_info=True)
            return None

    async def _enqueue_missing_catalog_embeddings(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        limit: int = 12,
    ) -> int:
        if not entity_type:
            return 0
        try:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT id, texto_busqueda
                        FROM semantic_catalog
                        WHERE entidad_tipo = :etype
                          AND activo = TRUE
                          AND embedding IS NULL
                        ORDER BY id ASC
                        LIMIT :lim
                        """
                    ),
                    {"etype": entity_type, "lim": int(max(1, min(limit, 100)))},
                )
            ).mappings().all()
        except Exception:
            logger.debug("No se pudieron consultar faltantes de semantic_catalog", exc_info=True)
            return 0

        queued = 0
        for row in rows:
            await self._enqueue_embedding_job(
                session,
                source_table="semantic_catalog",
                source_id=int(row.get("id")),
                source_text=str(row.get("texto_busqueda") or ""),
                model=self.SEMANTIC_EMBED_MODEL,
            )
            queued += 1
        return queued

    @classmethod
    def _find_row_by_code(
        cls,
        rows: list[dict[str, Any]],
        code: Optional[str],
    ) -> Optional[dict[str, Any]]:
        target = cls._normalize_text(code)
        if not target:
            return None
        for row in rows:
            if cls._normalize_text(row.get("codigo")) == target:
                return row
        return None

    @classmethod
    def _rows_by_code(cls, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = cls._normalize_text(row.get("codigo"))
            if key and key not in mapping:
                mapping[key] = row
        return mapping

    async def _resolve_catalog_exact_match(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        query_normalized: str,
    ) -> Optional[dict[str, Any]]:
        if not entity_type or not query_normalized:
            return None
        row = (
            await session.execute(
                text(
                    """
                    SELECT entidad_codigo, texto_busqueda
                    FROM semantic_catalog
                    WHERE entidad_tipo = :etype
                      AND activo = TRUE
                      AND texto_normalizado = :qnorm
                    ORDER BY peso_lexico DESC, peso_semantico DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"etype": entity_type, "qnorm": query_normalized},
            )
        ).mappings().first()
        return dict(row) if row else None

    async def _resolve_catalog_trgm_match(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        query_normalized: str,
    ) -> Optional[dict[str, Any]]:
        if not entity_type or not query_normalized:
            return None
        try:
            async with session.begin_nested():
                row = (
                    await session.execute(
                        text(
                            """
                            SELECT
                                entidad_codigo,
                                texto_busqueda,
                                similarity(texto_normalizado, CAST(:qnorm AS text)) AS score
                            FROM semantic_catalog
                            WHERE entidad_tipo = :etype
                              AND activo = TRUE
                              AND texto_normalizado %% CAST(:qnorm AS text)
                            ORDER BY score DESC, peso_lexico DESC, id ASC
                            LIMIT 1
                            """
                        ),
                        {"etype": entity_type, "qnorm": query_normalized},
                    )
                ).mappings().first()
            return dict(row) if row else None
        except Exception:
            logger.debug("TRGM de semantic_catalog no disponible para entidad_tipo=%s", entity_type, exc_info=True)
            return None

    @classmethod
    def _should_attempt_ai_semantic(cls, normalized_query: str) -> bool:
        if not normalized_query:
            return False
        compact = normalized_query.replace(" ", "")
        if len(compact) < 3:
            return False
        tokens = normalized_query.split()
        if len(tokens) == 1 and len(tokens[0]) <= 2:
            return False
        return True

    async def _resolve_alias_match(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        query_normalized: str,
        rows: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not entity_type or not query_normalized:
            return None
        alias_row = (
            await session.execute(
                text(
                    """
                    SELECT id, entidad_codigo, alias_texto
                    FROM mae_alias_semantico
                    WHERE entidad_tipo = :etype
                      AND alias_normalizado = :alias
                      AND activo = TRUE
                    ORDER BY prioridad DESC, es_canonico DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"etype": entity_type, "alias": query_normalized},
            )
        ).mappings().first()
        if not alias_row:
            return None
        master_row = self._find_row_by_code(rows, alias_row.get("entidad_codigo"))
        if not master_row:
            return None
        return {
            "master_row": master_row,
            "alias_id": alias_row.get("id"),
            "alias_texto": alias_row.get("alias_texto"),
        }

    async def _resolve_trgm_match(
        self,
        session: AsyncSession,
        *,
        table_name: str,
        code_column: str,
        name_column: str,
        extra_where: str,
        query_normalized: str,
    ) -> Optional[dict[str, Any]]:
        if not query_normalized:
            return None
        try:
            # Aisla fallos de extensiones opcionales (pg_trgm) sin abortar
            # la transaccion principal del turno.
            async with session.begin_nested():
                row = (
                    await session.execute(
                        text(
                            f"""
                            SELECT
                                id,
                                {code_column} AS codigo,
                                {name_column} AS nombre,
                                GREATEST(
                                    COALESCE(similarity(lower(CAST({code_column} AS text)), CAST(:qnorm AS text)), 0),
                                    COALESCE(similarity(lower(CAST({name_column} AS text)), CAST(:qnorm AS text)), 0)
                                ) AS score
                            FROM {table_name}
                            WHERE activo = TRUE
                              AND {extra_where}
                              AND (
                                lower(CAST({code_column} AS text)) %% CAST(:qnorm AS text)
                                OR lower(CAST({name_column} AS text)) %% CAST(:qnorm AS text)
                              )
                            ORDER BY score DESC, id ASC
                            LIMIT 1
                            """
                        ),
                        {"qnorm": query_normalized},
                    )
                ).mappings().first()
            return dict(row) if row else None
        except Exception:
            logger.debug("TRGM no disponible para %s", table_name, exc_info=True)
            return None

    async def _resolve_vector_match(
        self,
        session: AsyncSession,
        *,
        entity_type: Optional[str],
        query_text: str,
        rows: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not self._openai_client or not entity_type or not query_text:
            return None
        if not self._should_attempt_ai_semantic(self._normalize_text(query_text)):
            return None
        await self._enqueue_missing_catalog_embeddings(session, entity_type=entity_type)
        has_embeddings = (
            await session.execute(
                text(
                    """
                    SELECT 1
                    FROM semantic_catalog
                    WHERE entidad_tipo = :etype
                      AND activo = TRUE
                      AND embedding IS NOT NULL
                    LIMIT 1
                    """
                ),
                {"etype": entity_type},
            )
        ).first()
        if not has_embeddings:
            return None
        try:
            emb_resp = await asyncio.wait_for(
                self._openai_client.embeddings.create(
                    input=[query_text],
                    model=self.SEMANTIC_EMBED_MODEL,
                ),
                timeout=2.8,
            )
            embedding = emb_resp.data[0].embedding
            literal = self._build_embedding_literal(embedding)
            if not literal:
                return None

            # Aisla fallos de pgvector/tablas opcionales sin abortar el turno.
            async with session.begin_nested():
                cat_row = (
                    await session.execute(
                        text(
                            """
                            SELECT entidad_codigo, (embedding <=> CAST(:emb AS vector)) AS distance
                            FROM semantic_catalog
                            WHERE entidad_tipo = :etype
                              AND activo = TRUE
                              AND embedding IS NOT NULL
                            ORDER BY embedding <=> CAST(:emb AS vector)
                            LIMIT 1
                            """
                        ),
                        {"etype": entity_type, "emb": literal},
                    )
                ).mappings().first()
            if not cat_row:
                return None
            master_row = self._find_row_by_code(rows, cat_row.get("entidad_codigo"))
            if not master_row:
                return None
            score = 1.0 - float(cat_row.get("distance") or 1.0)
            return {
                "id": master_row.get("id"),
                "codigo": master_row.get("codigo"),
                "nombre": master_row.get("nombre"),
                "score": max(0.0, min(score, 1.0)),
            }
        except Exception:
            logger.debug("Vector fallback no disponible para entidad_tipo=%s", entity_type, exc_info=True)
            return None

    async def _resolve_llm_fallback(
        self,
        *,
        query_text: str,
        candidates: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not self._openai_client or not candidates:
            return None
        if not self._should_attempt_ai_semantic(self._normalize_text(query_text)):
            return None
        shortlist = candidates[:8]
        prompt = (
            "Usuario dijo: "
            f"'{query_text}'.\n"
            "Elige el codigo mas probable de la lista o null si ninguno aplica.\n"
            f"Candidatos: {json.dumps(shortlist, ensure_ascii=False)}\n"
            "Responde JSON con {'codigo': <string|null>, 'confidence': <0-1>}."
        )
        try:
            resp = await asyncio.wait_for(
                self._openai_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "Resuelve entidades de catalogo con criterio conservador."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=120,
                ),
                timeout=3.2,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            code = data.get("codigo")
            conf = float(data.get("confidence") or 0.0)
            if not code or conf < 0.7:
                return None
            return {"codigo": str(code), "score": max(0.0, min(conf, 1.0))}
        except Exception:
            logger.debug("LLM fallback de resolucion semantica no disponible", exc_info=True)
            return None

    async def _resolve_master_id(
        self,
        session: AsyncSession,
        table_name: str,
        raw_value: Optional[str],
        *,
        code_column: str = "codigo",
        name_column: str = "nombre",
        extra_where: str = "TRUE",
        minimum_score: float = 0.9,
        usuario_id: Optional[int] = None,
        incoming_message_id: Optional[int] = None,
        semantic_field_code: Optional[str] = None,
    ) -> Optional[int]:
        if raw_value is None:
            return None
        raw_text = str(raw_value).strip()
        if not raw_text:
            return None
        target_norm = self._normalize_text(raw_text)
        if not target_norm or self._is_negative_value(raw_text):
            return None

        field_for_log = semantic_field_code or table_name
        # Cache por campo semantico de dominio (alergias, enfermedades, etc.)
        # para evitar mezclar resoluciones entre campos que comparten maestro.
        cache_field_code = field_for_log
        entity_type = self.SEMANTIC_ENTITY_TYPE_BY_TABLE.get(table_name)
        started = perf_counter()

        resolved_row: Optional[dict[str, Any]] = None
        alias_meta: Optional[dict[str, Any]] = None
        strategy = "NO_MATCH"
        confidence = 0.0
        exact_match = False
        trigram_score: Optional[float] = None
        vector_score: Optional[float] = None
        escalated_to_ai = False
        decided_by_rule = False
        cache_hit = False
        review_reason: Optional[str] = None
        error_detail: Optional[str] = None
        ranked_candidates: list[dict[str, Any]] = []

        try:
            async with session.begin_nested():
                cache = await self._semantic_cache_get(
                    session,
                    field_code=cache_field_code,
                    query_normalized=target_norm,
                )

                query = text(
                    f"""
                    SELECT id, {code_column} AS codigo, {name_column} AS nombre
                    FROM {table_name}
                    WHERE activo = TRUE
                      AND {extra_where}
                    """
                )
                rows = [dict(r) for r in (await session.execute(query)).mappings().all()]
                rows_by_code = self._rows_by_code(rows)
                if not rows:
                    review_reason = "NO_MATCH"
                    error_detail = f"Sin filas activas en {table_name}"
                else:
                    if cache and cache.get("entidad_codigo_resuelto"):
                        cached_row = self._find_row_by_code(rows, cache.get("entidad_codigo_resuelto"))
                        if cached_row:
                            resolved_row = cached_row
                            cache_hit = True
                            cached_strategy = str(cache.get("estrategia_usada") or "").upper()
                            strategy = f"CACHE_{cached_strategy}" if cached_strategy else "CACHE"
                            confidence = float(cache.get("confidence") or 0.0)
                            exact_match = cached_strategy == "EXACT"

                    if not resolved_row:
                        catalog_exact = await self._resolve_catalog_exact_match(
                            session,
                            entity_type=entity_type,
                            query_normalized=target_norm,
                        )
                        if catalog_exact and catalog_exact.get("entidad_codigo"):
                            resolved_row = rows_by_code.get(
                                self._normalize_text(catalog_exact.get("entidad_codigo"))
                            )
                            if resolved_row:
                                strategy = "CATALOG_EXACT"
                                confidence = 0.97
                                exact_match = True
                                await self._semantic_cache_put(
                                    session,
                                    field_code=cache_field_code,
                                    raw_query=raw_text,
                                    query_normalized=target_norm,
                                    entity_type=entity_type,
                                    entity_code=str(resolved_row.get("codigo")),
                                    strategy="CATALOG_EXACT",
                                    confidence=confidence,
                                    top_candidates=[
                                        {
                                            "codigo": resolved_row.get("codigo"),
                                            "nombre": resolved_row.get("nombre"),
                                            "score": confidence,
                                        }
                                    ],
                                )

                    if not resolved_row:
                        best_row: Optional[dict[str, Any]] = None
                        best_score = 0.0
                        for row in rows:
                            score = self._score_master_candidate(target_norm, row.get("codigo"), row.get("nombre"))
                            if score > best_score:
                                best_score = score
                                best_row = row
                        if best_row and best_score >= minimum_score:
                            resolved_row = best_row
                            strategy = "EXACT"
                            confidence = float(best_score)
                            exact_match = True
                            await self._semantic_cache_put(
                                session,
                                field_code=cache_field_code,
                                raw_query=raw_text,
                                query_normalized=target_norm,
                                entity_type=entity_type,
                                entity_code=str(best_row.get("codigo")),
                                strategy="EXACT",
                                confidence=best_score,
                                top_candidates=[
                                    {
                                        "codigo": best_row.get("codigo"),
                                        "nombre": best_row.get("nombre"),
                                        "score": round(best_score, 4),
                                    }
                                ],
                            )

                    if not resolved_row:
                        alias_meta = await self._resolve_alias_match(
                            session,
                            entity_type=entity_type,
                            query_normalized=target_norm,
                            rows=rows,
                        )
                        if alias_meta and alias_meta.get("master_row"):
                            resolved_row = dict(alias_meta.get("master_row"))
                            strategy = "ALIAS"
                            confidence = 0.96
                            await self._semantic_cache_put(
                                session,
                                field_code=cache_field_code,
                                raw_query=raw_text,
                                query_normalized=target_norm,
                                entity_type=entity_type,
                                entity_code=str(resolved_row.get("codigo")),
                                strategy="ALIAS",
                                confidence=confidence,
                                top_candidates=[
                                    {
                                        "codigo": resolved_row.get("codigo"),
                                        "nombre": resolved_row.get("nombre"),
                                        "score": confidence,
                                    }
                                ],
                            )

                    if not resolved_row:
                        catalog_trgm = await self._resolve_catalog_trgm_match(
                            session,
                            entity_type=entity_type,
                            query_normalized=target_norm,
                        )
                        catalog_trgm_score = float(catalog_trgm.get("score") or 0.0) if catalog_trgm else 0.0
                        if catalog_trgm and catalog_trgm_score >= max(0.78, minimum_score - 0.12):
                            resolved_row = rows_by_code.get(
                                self._normalize_text(catalog_trgm.get("entidad_codigo"))
                            )
                            if resolved_row:
                                strategy = "CATALOG_TRGM"
                                confidence = catalog_trgm_score
                                decided_by_rule = True
                                trigram_score = catalog_trgm_score
                                await self._semantic_cache_put(
                                    session,
                                    field_code=cache_field_code,
                                    raw_query=raw_text,
                                    query_normalized=target_norm,
                                    entity_type=entity_type,
                                    entity_code=str(resolved_row.get("codigo")),
                                    strategy="CATALOG_TRGM",
                                    confidence=confidence,
                                    top_candidates=[
                                        {
                                            "codigo": resolved_row.get("codigo"),
                                            "nombre": resolved_row.get("nombre"),
                                            "score": round(confidence, 4),
                                        }
                                    ],
                                )

                    if not resolved_row:
                        trgm_row = await self._resolve_trgm_match(
                            session,
                            table_name=table_name,
                            code_column=code_column,
                            name_column=name_column,
                            extra_where=extra_where,
                            query_normalized=target_norm,
                        )
                        trigram_score = float(trgm_row.get("score") or 0.0) if trgm_row else None
                        if trgm_row and trigram_score >= max(0.78, minimum_score - 0.12):
                            resolved_row = trgm_row
                            strategy = "TRGM"
                            confidence = float(trigram_score)
                            decided_by_rule = True
                            await self._semantic_cache_put(
                                session,
                                field_code=cache_field_code,
                                raw_query=raw_text,
                                query_normalized=target_norm,
                                entity_type=entity_type,
                                entity_code=str(trgm_row.get("codigo")),
                                strategy="TRGM",
                                confidence=confidence,
                                top_candidates=[
                                    {
                                        "codigo": trgm_row.get("codigo"),
                                        "nombre": trgm_row.get("nombre"),
                                        "score": round(confidence, 4),
                                    }
                                ],
                            )

                    if not resolved_row:
                        fuzzy_row: Optional[dict[str, Any]] = None
                        fuzzy_score = 0.0
                        for row in rows:
                            code_norm = self._normalize_text(row.get("codigo"))
                            name_norm = self._normalize_text(row.get("nombre"))
                            ratio = max(
                                SequenceMatcher(None, target_norm, code_norm).ratio() if code_norm else 0.0,
                                SequenceMatcher(None, target_norm, name_norm).ratio() if name_norm else 0.0,
                            )
                            if ratio > fuzzy_score:
                                fuzzy_score = ratio
                                fuzzy_row = row
                        if fuzzy_row and fuzzy_score >= max(0.8, minimum_score - 0.1):
                            resolved_row = fuzzy_row
                            strategy = "FUZZY"
                            confidence = float(fuzzy_score)
                            decided_by_rule = True
                            await self._semantic_cache_put(
                                session,
                                field_code=cache_field_code,
                                raw_query=raw_text,
                                query_normalized=target_norm,
                                entity_type=entity_type,
                                entity_code=str(fuzzy_row.get("codigo")),
                                strategy="FUZZY",
                                confidence=confidence,
                                top_candidates=[
                                    {
                                        "codigo": fuzzy_row.get("codigo"),
                                        "nombre": fuzzy_row.get("nombre"),
                                        "score": round(confidence, 4),
                                    }
                                ],
                            )

                    if not resolved_row and self._should_attempt_ai_semantic(target_norm):
                        vector_row = await self._resolve_vector_match(
                            session,
                            entity_type=entity_type,
                            query_text=raw_text,
                            rows=rows,
                        )
                        vector_score = float(vector_row.get("score") or 0.0) if vector_row else None
                        if vector_row and vector_score >= max(0.78, minimum_score - 0.12):
                            resolved_row = vector_row
                            strategy = "VECTOR"
                            confidence = float(vector_score)
                            await self._semantic_cache_put(
                                session,
                                field_code=cache_field_code,
                                raw_query=raw_text,
                                query_normalized=target_norm,
                                entity_type=entity_type,
                                entity_code=str(vector_row.get("codigo")),
                                strategy="VECTOR",
                                confidence=confidence,
                                top_candidates=[
                                    {
                                        "codigo": vector_row.get("codigo"),
                                        "nombre": vector_row.get("nombre"),
                                        "score": round(confidence, 4),
                                    }
                                ],
                            )

                    if not resolved_row:
                        for row in rows:
                            code_norm = self._normalize_text(row.get("codigo"))
                            name_norm = self._normalize_text(row.get("nombre"))
                            lexical = self._score_master_candidate(target_norm, row.get("codigo"), row.get("nombre"))
                            ratio = max(
                                SequenceMatcher(None, target_norm, code_norm).ratio() if code_norm else 0.0,
                                SequenceMatcher(None, target_norm, name_norm).ratio() if name_norm else 0.0,
                            )
                            score = max(lexical, ratio * 0.92)
                            if score < 0.35:
                                continue
                            ranked_candidates.append(
                                {
                                    "codigo": row.get("codigo"),
                                    "nombre": row.get("nombre"),
                                    "score": round(score, 4),
                                }
                            )
                        ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

                        top_score = float(ranked_candidates[0]["score"]) if ranked_candidates else 0.0
                        second_score = float(ranked_candidates[1]["score"]) if len(ranked_candidates) > 1 else 0.0
                        can_escalate = (
                            ranked_candidates
                            and top_score >= max(self.SEMANTIC_LLM_FALLBACK_MIN_SCORE, 0.7)
                            and (top_score - second_score) >= 0.05
                            and self._should_attempt_ai_semantic(target_norm)
                        )
                        if can_escalate:
                            escalated_to_ai = True
                            llm_pick = await self._resolve_llm_fallback(
                                query_text=raw_text,
                                candidates=ranked_candidates,
                            )
                            if llm_pick and llm_pick.get("codigo"):
                                llm_row = self._find_row_by_code(rows, llm_pick.get("codigo"))
                                if llm_row:
                                    resolved_row = llm_row
                                    strategy = "LLM"
                                    confidence = float(llm_pick.get("score") or 0.0)
                                    await self._semantic_cache_put(
                                        session,
                                        field_code=cache_field_code,
                                        raw_query=raw_text,
                                        query_normalized=target_norm,
                                        entity_type=entity_type,
                                        entity_code=str(llm_row.get("codigo")),
                                        strategy="LLM",
                                        confidence=confidence,
                                        top_candidates=ranked_candidates[:5],
                                    )
                        if not resolved_row:
                            review_reason = "LOW_CONFIDENCE" if ranked_candidates else "NO_MATCH"

                    if resolved_row and entity_type:
                        canonical_name = str(resolved_row.get("nombre") or "")
                        canonical_entry = await self._upsert_semantic_catalog_entry(
                            session,
                            entity_type=entity_type,
                            entity_code=str(resolved_row.get("codigo")),
                            canonical_name=canonical_name,
                            search_text=canonical_name,
                            alias_semantico_id=int(alias_meta.get("alias_id")) if alias_meta and alias_meta.get("alias_id") else None,
                        )
                        if canonical_entry and not canonical_entry.get("embedding"):
                            await self._enqueue_embedding_job(
                                session,
                                source_table="semantic_catalog",
                                source_id=int(canonical_entry.get("id")),
                                source_text=str(canonical_entry.get("texto_busqueda") or canonical_name),
                                model=self.SEMANTIC_EMBED_MODEL,
                            )

                        canonical_norm = self._normalize_text(canonical_name)
                        if target_norm and target_norm != canonical_norm:
                            query_entry = await self._upsert_semantic_catalog_entry(
                                session,
                                entity_type=entity_type,
                                entity_code=str(resolved_row.get("codigo")),
                                canonical_name=canonical_name,
                                search_text=raw_text,
                                alias_semantico_id=int(alias_meta.get("alias_id")) if alias_meta and alias_meta.get("alias_id") else None,
                            )
                            if query_entry and not query_entry.get("embedding"):
                                await self._enqueue_embedding_job(
                                    session,
                                    source_table="semantic_catalog",
                                    source_id=int(query_entry.get("id")),
                                    source_text=str(query_entry.get("texto_busqueda") or raw_text),
                                    model=self.SEMANTIC_EMBED_MODEL,
                                )

        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            review_reason = review_reason or "RULE_MISMATCH"
            logger.debug("Error en _resolve_master_id(%s, %s)", table_name, raw_text, exc_info=True)

        latency_ms = int((perf_counter() - started) * 1000)
        resolved_code = str(resolved_row.get("codigo")) if resolved_row else None
        await self._log_semantic_match(
            session,
            usuario_id=usuario_id,
            incoming_message_id=incoming_message_id,
            field_code=field_for_log,
            query_text=raw_text,
            query_normalized=target_norm,
            strategy=strategy,
            exact_match=exact_match,
            trigram_score=trigram_score,
            vector_score=vector_score,
            confidence=confidence if resolved_row else None,
            entity_type=entity_type if resolved_row else None,
            entity_code=resolved_code,
            escalated_to_ai=escalated_to_ai,
            decided_by_rule=decided_by_rule,
            latency_ms=latency_ms,
            error_detail=error_detail,
        )

        if review_reason:
            await self._enqueue_semantic_review(
                session,
                usuario_id=usuario_id,
                incoming_message_id=incoming_message_id,
                field_code=field_for_log,
                query_text=raw_text,
                query_normalized=target_norm,
                reason=review_reason,
                top_candidates=ranked_candidates[:5],
                observation=error_detail,
            )
            return None

        if not resolved_row:
            return None
        return int(resolved_row.get("id"))

    async def _log_profile_extraction(
        self,
        session: AsyncSession,
        usuario_id: int,
        field_code: str,
        raw_value: Any,
        now_peru,
        *,
        confidence: float = 1.0,
        evidence_text: str = "Automatic extraction",
        status: str = "confirmed",
        resolved_entity_type: Optional[str] = None,
        resolved_entity_code: Optional[str] = None,
        resolution_strategy: Optional[str] = None,
        semantic_cache_hit: bool = False,
    ) -> None:
        if raw_value is None:
            return
        raw_text = str(raw_value).strip()
        if not raw_text:
            return
        try:
            # Bitacora no critica: si falla, no rompe el turno.
            async with session.begin_nested():
                await session.execute(
                    text(
                        """
                        INSERT INTO profile_extractions
                            (
                                usuario_id, field_code, raw_value, normalized_value,
                                confidence, evidence_text, status,
                                resolved_entity_type, resolved_entity_code, resolution_strategy, semantic_cache_hit,
                                extracted_at
                            )
                        VALUES
                            (
                                :uid, :field, :raw, :normalized,
                                :confidence, :evidence, :status,
                                :etype, :ecode, :strategy, :cache_hit,
                                :now
                            )
                        """
                    ),
                    {
                        "uid": usuario_id,
                        "field": field_code,
                        "raw": raw_text,
                        "normalized": self._normalize_text(raw_text)[:500],
                        "confidence": float(max(0.0, min(confidence, 1.0))),
                        "evidence": evidence_text[:2000],
                        "status": status[:20],
                        "etype": resolved_entity_type[:30] if resolved_entity_type else None,
                        "ecode": resolved_entity_code[:60] if resolved_entity_code else None,
                        "strategy": resolution_strategy[:20] if resolution_strategy else None,
                        "cache_hit": bool(semantic_cache_hit),
                        "now": now_peru,
                    },
                )
        except Exception:
            logger.debug("No se pudo insertar profile_extractions", exc_info=True)

    async def _validate_semantic(
        self, 
        session: AsyncSession, 
        categoria: str, 
        value: str, 
        threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """Busca similitud semÃ¡ntica en el catÃ¡logo maestro."""
        if not value or value.upper() == "NINGUNA":
            return None

        # Generar embedding del valor extraÃ­do
        try:
            resp = await self._openai_client.embeddings.create(
                input=[value],
                model="text-embedding-3-small"
            )
            embedding = resp.data[0].embedding
        except Exception as e:
            logger.error(f"Error generando embedding para validaciÃ³n: {e}")
            return None

        # BÃºsqueda vectorial en BD
        query = text("""
            SELECT id, nombre, categoria, (embedding <=> :emb) as distancia
            FROM catalogo_maestro
            WHERE categoria = :cat
            ORDER BY distancia ASC
            LIMIT 1
        """)
        
        result = await session.execute(query, {"emb": embedding, "cat": categoria})
        row = result.fetchone()
        
        if row and (1 - row.distancia) >= threshold:
            return {
                "id": row.id,
                "nombre": row.nombre,
                "score": 1 - row.distancia
            }
        
        return None

    @classmethod
    def contains_absurd_claim(cls, value: str) -> bool:
        txt = cls._normalize_text(value)
        if not txt:
            return False
        # El escudo de tÃ©rminos absurdos sigue siendo Ãºtil para filtrado rÃ¡pido pre-BD
        return any(term in txt for term in cls.ABSURD_TERMS)

    def _check_health_ambiguity(self, value: str, user_text: str = "") -> Optional[str]:
        """Retorna un prompt de aclaracion si el valor es ambiguo, None si esta completo o el usuario se rehusa."""
        norm = value.lower().strip()
        text_norm = user_text.lower()

        # Si el usuario explícitamente dice que no sabe o que solo es eso, no insistimos
        refusal_markers = ["no se", "no lo se", "no estoy segur", "solo ", "solamente", "no hay", "no, ", "omitir", "saltar", "no importa", "ya te dije", "ya te", "asi nomas", "así nomás", "así nomas", "nada mas", "nada más"]
        if any(marker in text_norm for marker in refusal_markers):
            return None

        for pattern, prompt in self._AMBIGUOUS_CONDITIONS:
            if pattern in norm:
                has_specificity = any(marker in norm for marker in self._SPECIFICITY_MARKERS)
                if not has_specificity:
                    return f"Entendido. Lo registro de forma provisional; para ser mas precisos, {prompt}"
        return None

    async def extract_and_save(
        self,
        user_text: str,
        usuario_id: int,
        session: AsyncSession,
        current_step: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extrae informacion de perfil de forma sincrona usando IA y persiste en BD.
        Retorna ExtractionResult.
        """
        raw_extractions = await self._run_llm_extraction(user_text, current_step)
        clean_data, updates, meta_flags = self._apply_bulletproof_logic(raw_extractions, user_text, current_step)
        
        if clean_data:
            unresolved_by_field = await self._persist_updates(
                usuario_id,
                clean_data,
                session,
                source_text=user_text,
                current_step=current_step,
            )
            self._inject_unresolved_alerts(unresolved_by_field, meta_flags)
            
        return ExtractionResult(clean_data=clean_data, updates=updates, meta_flags=meta_flags)

    async def apply_profile_intent(
        self,
        session: AsyncSession,
        usuario_id: int,
        intent,  # ProfileIntentResult
        state=None,  # ConversationState, optional
    ) -> ExtractionResult:
        """
        Aplica una intención de perfil respetando operación, entidades
        pre-resueltas y confianza. NO re-resuelve lo que ya fue resuelto
        por el SemanticEntityResolver.

        Soporta: ADD, REMOVE, REPLACE, CLEAR, CORRECTION, HISTORICAL_UPDATE, NOOP.
        """
        from domain.profile_intent import ProfileIntentResult

        if not intent or not isinstance(intent, ProfileIntentResult):
            return ExtractionResult(clean_data={}, updates={}, meta_flags={})

        if not intent.is_profile_update or intent.operation == "NOOP":
            return ExtractionResult(clean_data={}, updates={}, meta_flags={})

        field_code = intent.field_code
        operation = intent.operation or "REPLACE"
        now_peru = get_now_peru()
        meta_flags: dict[str, Any] = {}
        clean_data: dict[str, Any] = {}
        updates: dict[str, Any] = {}

        # ── Obtener o crear perfil_nutricional ──
        profile_res = await session.execute(
            text("SELECT id FROM perfil_nutricional WHERE usuario_id = :uid"),
            {"uid": usuario_id},
        )
        row = profile_res.fetchone()
        if not row:
            insert_prof = await session.execute(
                text("""
                    INSERT INTO perfil_nutricional (usuario_id, creado_en, actualizado_en)
                    VALUES (:uid, :now, :now)
                    RETURNING id
                """),
                {"uid": usuario_id, "now": now_peru},
            )
            perfil_id = insert_prof.fetchone().id
        else:
            perfil_id = row.id

        # ── Despachar por tipo de campo ──

        # 1. Campos numéricos: edad, peso_kg, altura_cm
        if field_code in ("edad", "peso_kg", "altura_cm"):
            raw_val = intent.values[0].raw_value if intent.values else None
            if raw_val is None:
                return ExtractionResult(clean_data={}, updates={}, meta_flags={})

            if field_code == "edad":
                try:
                    edad = int(float(raw_val))
                    await session.execute(
                        text("""
                            UPDATE perfil_nutricional
                            SET edad_reportada = :edad, fecha_referencia_edad = :fecha, actualizado_en = :now
                            WHERE id = :pid
                        """),
                        {"edad": edad, "fecha": now_peru.date(), "now": now_peru, "pid": perfil_id},
                    )
                    clean_data["edad"] = edad
                    updates["edad"] = edad
                except (TypeError, ValueError):
                    logger.info("Edad inválida desde intent: %s", raw_val)

            elif field_code in ("peso_kg", "altura_cm"):
                mtype = "PESO_KG" if field_code == "peso_kg" else "ALTURA_CM"
                unit = "kg" if field_code == "peso_kg" else "cm"
                try:
                    val = float(raw_val)
                    # CORRECTION vs HISTORICAL_UPDATE
                    correction_mode = operation in ("CORRECTION", "REPLACE")
                    await self._upsert_measurement_with_semantics(
                        session, perfil_id, mtype, val, unit, now_peru,
                        correction_mode=correction_mode,
                    )
                    clean_data[field_code] = val
                    updates[field_code] = val
                except (TypeError, ValueError):
                    logger.info("%s inválido desde intent: %s", field_code, raw_val)

            await self._log_profile_extraction(session, usuario_id, field_code, raw_val, now_peru)

        # 2. Campos de lista: enfermedades, alergias, restricciones_alimentarias
        elif field_code in ("enfermedades", "alergias", "restricciones_alimentarias"):
            # Mapear operación del intent a constantes del servicio
            op_map = {
                "ADD": self.OP_ADD,
                "REMOVE": self.OP_REMOVE,
                "REPLACE": self.OP_REPLACE,
                "CLEAR": self.OP_CLEAR,
                "CORRECTION": self.OP_REPLACE,
                "HISTORICAL_UPDATE": self.OP_ADD,
            }
            op = op_map.get(operation, self.OP_ADD)

            if op == self.OP_CLEAR:
                raw_value_str = "NINGUNA"
            else:
                # Usar raw_value de cada intent.value para la resolución downstream
                raw_value_str = ", ".join(v.raw_value for v in intent.values if v.raw_value)

            if field_code == "enfermedades":
                unresolved = await self._sync_enfermedades(
                    session, perfil_id, raw_value_str, now_peru,
                    operation=op, usuario_id=usuario_id,
                )
            else:
                only_alergenos = (field_code == "alergias")
                unresolved = await self._sync_restricciones(
                    session, perfil_id, raw_value_str, now_peru,
                    only_alergenos=only_alergenos,
                    operation=op, usuario_id=usuario_id,
                )

            if unresolved:
                meta_flags[f"unresolved_{field_code}"] = unresolved
            clean_data[field_code] = raw_value_str
            updates[field_code] = raw_value_str
            await self._log_profile_extraction(
                session, usuario_id, field_code, raw_value_str, now_peru,
                resolved_entity_type=intent.values[0].entity_type if intent.values else None,
                resolved_entity_code=intent.values[0].entity_code if intent.values else None,
                resolution_strategy=intent.values[0].resolution_strategy if intent.values else None,
            )

        # 3. Campos escalares: tipo_dieta, objetivo_nutricional, provincia, distrito
        elif field_code in ("tipo_dieta", "objetivo_nutricional", "provincia", "distrito"):
            raw_val = intent.values[0].raw_value if intent.values else None
            if not raw_val:
                return ExtractionResult(clean_data={}, updates={}, meta_flags={})

            # Usar _persist_updates para un solo campo: reutiliza toda la lógica de resolución
            single_update = {field_code: raw_val}
            source_text = intent.evidence_text or raw_val
            unresolved_by_field = await self._persist_updates(
                usuario_id, single_update, session,
                source_text=source_text,
            )
            self._inject_unresolved_alerts(unresolved_by_field, meta_flags)
            clean_data[field_code] = raw_val
            updates[field_code] = raw_val

        else:
            # Campo no reconocido — fallback al pipeline legacy
            if intent.values:
                raw_val = intent.values[0].raw_value
                single_update = {field_code: raw_val}
                await self._persist_updates(
                    usuario_id, single_update, session,
                    source_text=intent.evidence_text or raw_val,
                )
                clean_data[field_code] = raw_val

        # ── Actualizar timestamp del perfil ──
        if not clean_data:
            return ExtractionResult(clean_data={}, updates={}, meta_flags=meta_flags)

        await session.execute(
            text("UPDATE perfil_nutricional SET actualizado_en = :now WHERE id = :pid"),
            {"now": now_peru, "pid": perfil_id},
        )

        # ── Trigger: reglas nutricionales si aplica ──
        _disease_or_restriction_fields = {"enfermedades", "alergias", "restricciones_alimentarias"}
        if self._nutritional_rules and field_code in _disease_or_restriction_fields:
            try:
                async with session.begin_nested():
                    await self._nutritional_rules.generate_or_update_dietary_order(session, usuario_id)
            except Exception as e:
                logger.error("apply_profile_intent: dietary order error user=%s: %s", usuario_id, e)

        logger.info(
            "apply_profile_intent: user=%s field=%s op=%s data=%s",
            usuario_id, field_code, operation, clean_data,
        )
        return ExtractionResult(clean_data=clean_data, updates=updates, meta_flags=meta_flags)

    async def apply_cleaning_and_save(
        self,
        raw_extractions: Dict[str, Any],
        user_text: str,
        usuario_id: int,
        session: AsyncSession,
        current_step: Optional[str] = None
    ) -> ExtractionResult:
        """
        Limpia datos crudos (extraidos previamente) y los persiste en BD.
        Util para cuando el Switchboard ya hizo la extraccion.
        """
        if not raw_extractions:
            return ExtractionResult(clean_data={}, updates={}, meta_flags={})

        clean_data, updates, meta_flags = self._apply_bulletproof_logic(raw_extractions, user_text, current_step)
        
        if clean_data:
            unresolved_by_field = await self._persist_updates(
                usuario_id,
                clean_data,
                session,
                source_text=user_text,
                current_step=current_step,
            )
            self._inject_unresolved_alerts(unresolved_by_field, meta_flags)
            
        return ExtractionResult(clean_data=clean_data, updates=updates, meta_flags=meta_flags)

    async def save_clean_data(
        self,
        usuario_id: int,
        clean_data: Dict[str, Any],
        session: AsyncSession,
        *,
        source_text: str = "",
        current_step: Optional[str] = None,
    ):
        """Persiste directamente datos ya limpios."""
        if clean_data:
            await self._persist_updates(
                usuario_id,
                clean_data,
                session,
                source_text=source_text,
                current_step=current_step,
            )

    def _inject_unresolved_alerts(self, unresolved_by_field: dict[str, list[str]], meta_flags: dict):
        if not unresolved_by_field:
            return
            
        for field, claims in unresolved_by_field.items():
            if not claims:
                continue
            claims_str = ", ".join(claims)
            meta_flags["needs_health_clarification"] = True
            
            if field == "enfermedades":
                meta_flags["clarification_prompt"] = (
                    f"Mencionaste '{claims_str}', pero no reconozco esa condicion medica en mi registro clinico. "
                    "Podrias confirmarme si esta bien escrito o de que se trata exactamente?"
                )
            elif field in ("alergias", "restricciones_alimentarias"):
                meta_flags["clarification_prompt"] = (
                    f"Mencionaste '{claims_str}', pero no logro identificarlo en mi registro de alimentos/alergenos. "
                    "Podrias confirmarme si esta bien escrito?"
                )
            else:
                meta_flags["clarification_prompt"] = (
                    f"Mencionaste '{claims_str}', pero no logre reconocer ese termino. "
                    "Podrias aclararlo un poco?"
                )
            break # Ask for clarification on the first one we find

    async def _run_llm_extraction(self, user_text: str, current_step: Optional[str]) -> Dict[str, Any]:
        context_info = f"\nPASO ACTUAL: {current_step}" if current_step else ""
        try:
            resp = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT + context_info},
                    {"role": "user", "content": user_text}
                ],
                response_format={"type": "json_object"},
                max_tokens=250,
                temperature=0
            )
            data = json.loads(resp.choices[0].message.content)
            logger.info("ProfileExtractionService: LLM extracted: %s", data)
            return data
        except Exception as e:
            logger.warning("Error en extraccion sincrona por IA: %s", e)
            return {}

    def _apply_bulletproof_logic(self, raw_extractions: dict, user_text: str, current_step: Optional[str]) -> tuple[dict, dict, dict]:
        """Retorna (clean_data, updates, meta_flags).
        
        meta_flags puede contener:
        - needs_health_clarification: bool
        - clarification_hints: list[str]
        - warnings: list[str]
        """
        clean_data = {}
        updates = {}
        meta_flags = {"warnings": []}
        text_lower = user_text.lower()
        negation_words = ["no tengo", "ninguna", "nada", "ninguno", "sin", "no como", "no padezco", "no sufro", "no se"]
        expected_field = self._expected_field_for_step(current_step)
        inferred_field_from_text = self._guess_field_from_text(user_text) if expected_field else None
        expected_field_explicit = self._text_explicitly_mentions_field(user_text, expected_field) if expected_field else False
        
        for key, raw_val in raw_extractions.items():
            key_norm = str(key).lower()
            config = self.FIELD_CONFIG.get(key_norm)
            
            if not config and current_step and raw_val:
                config_step = self.FIELD_CONFIG.get(current_step.lower())
                if config_step:
                    config = config_step

            if not config or not raw_val:
                continue
                
            parser = config["parser"]
            clean_val = parser(str(raw_val))
            
            if clean_val is not None:
                col_name = config["col"]

                # Guardia de coherencia por paso:
                # cuando estamos en un paso concreto de onboarding, evitamos contaminar
                # otros campos salvo que el usuario lo haya mencionado explícitamente.
                if expected_field and col_name != expected_field:
                    meta_flags["warnings"].append(
                        f"Ignorado valor para '{col_name}' por no corresponder al paso '{expected_field}'."
                    )
                    continue

                # Guardia semántica cruzada:
                # si el texto parece corresponder a otro campo distinto al paso actual,
                # no persistimos el valor para evitar ensuciar el perfil.
                if (
                    expected_field
                    and col_name == expected_field
                    and inferred_field_from_text
                    and inferred_field_from_text != expected_field
                    and not expected_field_explicit
                ):
                    meta_flags["clarification_prompt"] = (
                        "Creo que ese dato parece corresponder a otro campo del perfil. "
                        "¿Me lo confirmas para registrarlo correctamente?"
                    )
                    meta_flags["warnings"].append(
                        f"Posible mismatch semántico: step={expected_field}, inferred={inferred_field_from_text}, value={clean_val}"
                    )
                    continue

                if (
                    col_name in {"alergias", "enfermedades", "restricciones_alimentarias"}
                    and current_step
                    and self._step_matches_field(current_step, col_name)
                ):
                    source_list = standardize_text_list(user_text)
                    if source_list and str(source_list).upper() != "NINGUNA":
                        source_items = self._split_values(str(source_list))
                        extracted_items = self._split_values(str(clean_val))
                        if len(source_items) > len(extracted_items):
                            clean_val = source_list

                # No bloquear por listas estaticas aqui.
                # La resolucion final se hace contra maestro/alias/fuzzy/vector y, al final, LLM.
                if col_name in {"alergias", "enfermedades", "restricciones_alimentarias"} and self.contains_absurd_claim(str(clean_val)):
                    logger.info(
                        "Semantic pipeline: valor de baja plausibilidad detectado para '%s': %s",
                        col_name,
                        clean_val,
                    )

                # Validacion clinica generica: detectar datos ambiguos
                if col_name in {"enfermedades", "alergias"}:
                    prompt = self._check_health_ambiguity(str(clean_val), user_text)
                    if prompt:
                        meta_flags["needs_health_clarification"] = True
                        meta_flags["clarification_target"] = col_name
                        meta_flags["clarification_prompt"] = prompt
                        meta_flags["clarification_reason"] = "ambiguous_health_condition"
                        # Persistir el dato de todas formas (es razonable pero incompleto)

                if isinstance(clean_val, str) and clean_val.upper() == "NINGUNA":
                    is_current_step = False
                    if current_step:
                        cs_low = current_step.lower()
                        is_current_step = (col_name == cs_low or cs_low in col_name or col_name in cs_low or 
                                          (cs_low == "peso" and col_name == "peso_kg") or
                                          (cs_low == "altura" and col_name == "altura_cm") or
                                          (cs_low == "restricciones" and col_name == "restricciones_alimentarias"))
                    
                    if not is_current_step:
                        field_mentions = [key, col_name, "alergia", "enfermedad", "restriccion", "dieta", "objetivo", "correo", "asegurado"]
                        is_explicit_negation = any(nw in text_lower for nw in negation_words)
                        is_field_mentioned = any(fm in text_lower for fm in field_mentions)
                            
                        if not (is_explicit_negation and is_field_mentioned):
                            logger.warning("Data Shield: blocked NINGUNA for field '%s'", key)
                            continue

                col = col_name
                clean_data[col] = clean_val
                updates[col] = clean_val
                
        return clean_data, updates, meta_flags

    async def _get_current_measurement(
        self,
        session: AsyncSession,
        perfil_id: int,
        measurement_type: str,
    ) -> Optional[dict[str, Any]]:
        row = (
            await session.execute(
                text(
                    """
                    SELECT id, valor_decimal, unidad, fecha_medicion
                    FROM perfil_nutricional_medicion
                    WHERE perfil_nutricional_id = :pid
                      AND tipo_medicion = :mtype
                      AND es_valor_actual = TRUE
                    ORDER BY fecha_medicion DESC, id DESC
                    LIMIT 1
                    """
                ),
                {"pid": perfil_id, "mtype": measurement_type},
            )
        ).mappings().first()
        return dict(row) if row else None

    async def _upsert_measurement_with_semantics(
        self,
        session: AsyncSession,
        perfil_id: int,
        measurement_type: str,
        numeric_value: float,
        unit: str,
        now_peru,
        *,
        correction_mode: bool,
    ) -> None:
        current_row = await self._get_current_measurement(session, perfil_id, measurement_type)

        if current_row:
            current_value = float(current_row["valor_decimal"])
            if abs(current_value - float(numeric_value)) < 1e-6:
                await session.execute(
                    text(
                        """
                        UPDATE perfil_nutricional_medicion
                        SET unidad = :unit,
                            fecha_medicion = :now,
                            origen = :origin
                        WHERE id = :mid
                        """
                    ),
                    {
                        "mid": current_row["id"],
                        "unit": unit,
                        "now": now_peru,
                        "origin": "SELF_REPORT_CORRECTION" if correction_mode else "SELF_REPORT",
                    },
                )
                return

        if correction_mode and current_row:
            await session.execute(
                text(
                    """
                    UPDATE perfil_nutricional_medicion
                    SET valor_decimal = :val,
                        unidad = :unit,
                        fecha_medicion = :now,
                        origen = 'SELF_REPORT_CORRECTION'
                    WHERE id = :mid
                    """
                ),
                {"mid": current_row["id"], "val": numeric_value, "unit": unit, "now": now_peru},
            )
            return

        await session.execute(
            text(
                """
                UPDATE perfil_nutricional_medicion
                SET es_valor_actual = FALSE
                WHERE perfil_nutricional_id = :pid
                  AND tipo_medicion = :mtype
                  AND es_valor_actual = TRUE
                """
            ),
            {"pid": perfil_id, "mtype": measurement_type},
        )
        await session.execute(
            text(
                """
                INSERT INTO perfil_nutricional_medicion
                    (perfil_nutricional_id, tipo_medicion, valor_decimal, unidad, fecha_medicion, es_valor_actual, origen, creado_en)
                VALUES
                    (:pid, :mtype, :val, :unit, :now, TRUE, :origin, :now)
                """
            ),
            {
                "pid": perfil_id,
                "mtype": measurement_type,
                "val": numeric_value,
                "unit": unit,
                "now": now_peru,
                "origin": "SELF_REPORT",
            },
        )

    async def _sync_enfermedades(
        self,
        session: AsyncSession,
        perfil_id: int,
        raw_value: str,
        now_peru,
        *,
        operation: str,
        usuario_id: Optional[int] = None,
        incoming_message_id: Optional[int] = None,
    ) -> list[str]:
        unresolved: list[str] = []
        today = now_peru.date()
        if operation == self.OP_CLEAR:
            await session.execute(
                text(
                    """
                    UPDATE perfil_nutricional_enfermedad
                    SET vigente = FALSE, fecha_fin = :today
                    WHERE perfil_nutricional_id = :pid
                      AND vigente = TRUE
                    """
                ),
                {"pid": perfil_id, "today": today},
            )
            return unresolved

        values = self._split_values(raw_value)
        if not values:
            return unresolved

        resolved_ids: list[int] = []
        for value in values:
            enfermedad_id = await self._resolve_master_id(
                session,
                "mae_enfermedad_cie10",
                value,
                code_column="codigo_cie10",
                minimum_score=0.88,
                usuario_id=usuario_id,
                incoming_message_id=incoming_message_id,
                semantic_field_code="enfermedades",
            )
            if not enfermedad_id:
                logger.info("No se encontro enfermedad en maestro para valor='%s'", value)
                unresolved.append(value)
                continue
            resolved_ids.append(enfermedad_id)

        resolved_ids = list(dict.fromkeys(resolved_ids))
        if not resolved_ids:
            return unresolved

        if operation == self.OP_REPLACE:
            await session.execute(
                text(
                    """
                    UPDATE perfil_nutricional_enfermedad
                    SET vigente = FALSE, fecha_fin = :today
                    WHERE perfil_nutricional_id = :pid
                      AND vigente = TRUE
                    """
                ),
                {"pid": perfil_id, "today": today},
            )
        elif operation == self.OP_REMOVE:
            await session.execute(
                text(
                    """
                    UPDATE perfil_nutricional_enfermedad
                    SET vigente = FALSE, fecha_fin = :today
                    WHERE perfil_nutricional_id = :pid
                      AND vigente = TRUE
                      AND enfermedad_id = ANY(CAST(:ids AS bigint[]))
                    """
                ),
                {"pid": perfil_id, "today": today, "ids": resolved_ids},
            )
            return unresolved

        for enfermedad_id in resolved_ids:
            await session.execute(
                text(
                    """
                    INSERT INTO perfil_nutricional_enfermedad
                        (perfil_nutricional_id, enfermedad_id, es_principal, origen, grado_confianza, validado, fecha_inicio, vigente, creado_en)
                    VALUES
                        (:pid, :eid, FALSE, 'SELF_REPORT', 1.0, FALSE, :today, TRUE, :now)
                    ON CONFLICT (perfil_nutricional_id, enfermedad_id)
                    DO UPDATE SET
                        vigente = TRUE,
                        fecha_fin = NULL,
                        origen = EXCLUDED.origen,
                        grado_confianza = EXCLUDED.grado_confianza
                    """
                ),
                {"pid": perfil_id, "eid": enfermedad_id, "today": today, "now": now_peru},
            )
        return unresolved

    async def _sync_restricciones(
        self,
        session: AsyncSession,
        perfil_id: int,
        raw_value: str,
        now_peru,
        *,
        only_alergenos: bool,
        operation: str,
        usuario_id: Optional[int] = None,
        incoming_message_id: Optional[int] = None,
    ) -> list[str]:
        unresolved: list[str] = []
        today = now_peru.date()
        where_master = "tipo IN ('ALERGENO','INTOLERANCIA')" if only_alergenos else "TRUE"

        deactivate_query = """
            UPDATE perfil_nutricional_restriccion pr
            SET vigente = FALSE, fecha_fin = :today
            WHERE pr.perfil_nutricional_id = :pid
              AND pr.vigente = TRUE
        """
        if only_alergenos:
            deactivate_query += """
              AND pr.restriccion_id IN (
                    SELECT id FROM mae_restriccion_alimentaria WHERE tipo IN ('ALERGENO','INTOLERANCIA')
              )
            """

        if operation == self.OP_CLEAR:
            await session.execute(text(deactivate_query), {"pid": perfil_id, "today": today})
            return unresolved

        values = self._split_values(raw_value)
        if not values:
            return unresolved

        resolved_ids: list[int] = []
        for value in values:
            restriccion_id: Optional[int] = None
            for candidate in self._restriction_resolution_candidates(value):
                restriccion_id = await self._resolve_master_id(
                    session,
                    "mae_restriccion_alimentaria",
                    candidate,
                    extra_where=where_master,
                    minimum_score=0.88,
                    usuario_id=usuario_id,
                    incoming_message_id=incoming_message_id,
                    semantic_field_code="alergias" if only_alergenos else "restricciones_alimentarias",
                )
                if restriccion_id:
                    break
            if not restriccion_id:
                logger.info("No se encontro restriccion en maestro para valor='%s'", value)
                unresolved.append(value)
                continue
            resolved_ids.append(restriccion_id)

        if not resolved_ids:
            return unresolved

        if operation == self.OP_REPLACE:
            await session.execute(text(deactivate_query), {"pid": perfil_id, "today": today})
        elif operation == self.OP_REMOVE:
            remove_query = """
                UPDATE perfil_nutricional_restriccion pr
                SET vigente = FALSE, fecha_fin = :today
                WHERE pr.perfil_nutricional_id = :pid
                  AND pr.vigente = TRUE
                  AND pr.restriccion_id = ANY(CAST(:ids AS bigint[]))
            """
            if only_alergenos:
                remove_query += """
                  AND pr.restriccion_id IN (
                        SELECT id FROM mae_restriccion_alimentaria WHERE tipo IN ('ALERGENO','INTOLERANCIA')
                  )
                """
            await session.execute(text(remove_query), {"pid": perfil_id, "today": today, "ids": resolved_ids})
            return unresolved

        for restriccion_id in resolved_ids:
            await session.execute(
                text(
                    """
                    INSERT INTO perfil_nutricional_restriccion
                        (perfil_nutricional_id, restriccion_id, obligatoria, severidad, validado, origen, grado_confianza, fecha_inicio, vigente, creado_en)
                    VALUES
                        (:pid, :rid, FALSE, NULL, FALSE, 'SELF_REPORT', 1.0, :today, TRUE, :now)
                    ON CONFLICT (perfil_nutricional_id, restriccion_id)
                    DO UPDATE SET
                        vigente = TRUE,
                        fecha_fin = NULL,
                        origen = EXCLUDED.origen,
                        grado_confianza = EXCLUDED.grado_confianza
                    """
                ),
                {"pid": perfil_id, "rid": restriccion_id, "today": today, "now": now_peru},
            )
        return unresolved

    async def _persist_updates(
        self,
        usuario_id: int,
        updates: dict,
        session: AsyncSession,
        *,
        source_text: str = "",
        current_step: Optional[str] = None,
    ) -> dict[str, list[str]]:
        unresolved_by_field: dict[str, list[str]] = {}
        now_peru = get_now_peru()

        profile_res = await session.execute(
            text("SELECT id FROM perfil_nutricional WHERE usuario_id = :uid"),
            {"uid": usuario_id},
        )
        row = profile_res.fetchone()

        if not row:
            insert_prof = await session.execute(
                text(
                    """
                    INSERT INTO perfil_nutricional (usuario_id, creado_en, actualizado_en)
                    VALUES (:uid, :now, :now)
                    RETURNING id
                    """
                ),
                {"uid": usuario_id, "now": now_peru},
            )
            perfil_id = insert_prof.fetchone().id
        else:
            perfil_id = row.id

        perfil_updates: dict[str, Any] = {}
        for field_code, raw_value in updates.items():
            if raw_value is None:
                continue

            value = str(raw_value).strip() if isinstance(raw_value, str) else raw_value
            if isinstance(value, str) and not value:
                continue

            if field_code == "edad":
                try:
                    perfil_updates["edad_reportada"] = int(float(value))
                    perfil_updates["fecha_referencia_edad"] = now_peru.date()
                except (TypeError, ValueError):
                    logger.info("Edad invalida para persistencia V3: %s", value)
                continue

            if field_code == "peso_kg":
                try:
                    current_row = await self._get_current_measurement(session, perfil_id, "PESO_KG")
                    measurement_intent = self._classify_measurement_update(
                        source_text=source_text,
                        field_code=field_code,
                        current_step=current_step,
                        current_measurement_row=current_row,
                        now_peru=now_peru,
                    )
                    correction_mode = measurement_intent == self.MEASUREMENT_CORRECTION
                    await self._upsert_measurement_with_semantics(
                        session,
                        perfil_id,
                        "PESO_KG",
                        float(value),
                        "kg",
                        now_peru,
                        correction_mode=correction_mode,
                    )
                except (TypeError, ValueError):
                    logger.info("Peso invalido para persistencia V3: %s", value)
                continue

            if field_code == "altura_cm":
                try:
                    current_row = await self._get_current_measurement(session, perfil_id, "ALTURA_CM")
                    measurement_intent = self._classify_measurement_update(
                        source_text=source_text,
                        field_code=field_code,
                        current_step=current_step,
                        current_measurement_row=current_row,
                        now_peru=now_peru,
                    )
                    correction_mode = measurement_intent == self.MEASUREMENT_CORRECTION
                    await self._upsert_measurement_with_semantics(
                        session,
                        perfil_id,
                        "ALTURA_CM",
                        float(value),
                        "cm",
                        now_peru,
                        correction_mode=correction_mode,
                    )
                except (TypeError, ValueError):
                    logger.info("Altura invalida para persistencia V3: %s", value)
                continue

            if field_code == "tipo_dieta":
                patron_id = await self._resolve_master_id(
                    session,
                    "mae_patron_alimentario",
                    str(value),
                    minimum_score=0.84,
                    usuario_id=usuario_id,
                    semantic_field_code="tipo_dieta",
                )
                cache_meta = await self._semantic_cache_peek(
                    session,
                    field_code="tipo_dieta",
                    raw_query=str(value),
                )
                if patron_id:
                    perfil_updates["patron_alimentario_id"] = patron_id
                else:
                    unresolved_by_field["tipo_dieta"] = [str(value)]
                await self._log_profile_extraction(
                    session,
                    usuario_id,
                    field_code,
                    value,
                    now_peru,
                    resolved_entity_type=(cache_meta or {}).get("entidad_tipo_resuelta"),
                    resolved_entity_code=(cache_meta or {}).get("entidad_codigo_resuelto"),
                    resolution_strategy=(cache_meta or {}).get("estrategia_usada"),
                    semantic_cache_hit=bool(
                        (cache_meta or {}).get("estrategia_usada")
                        and str((cache_meta or {}).get("estrategia_usada")).upper().startswith("CACHE")
                    ),
                )
                continue

            if field_code == "objetivo_nutricional":
                objetivo_id = await self._resolve_master_id(
                    session,
                    "mae_objetivo_nutricional",
                    str(value),
                    minimum_score=0.82,
                    usuario_id=usuario_id,
                    semantic_field_code="objetivo_nutricional",
                )
                cache_meta = await self._semantic_cache_peek(
                    session,
                    field_code="objetivo_nutricional",
                    raw_query=str(value),
                )
                if objetivo_id:
                    perfil_updates["objetivo_nutricional_id"] = objetivo_id
                else:
                    unresolved_by_field["objetivo_nutricional"] = [str(value)]
                await self._log_profile_extraction(
                    session,
                    usuario_id,
                    field_code,
                    value,
                    now_peru,
                    resolved_entity_type=(cache_meta or {}).get("entidad_tipo_resuelta"),
                    resolved_entity_code=(cache_meta or {}).get("entidad_codigo_resuelto"),
                    resolution_strategy=(cache_meta or {}).get("estrategia_usada"),
                    semantic_cache_hit=bool(
                        (cache_meta or {}).get("estrategia_usada")
                        and str((cache_meta or {}).get("estrategia_usada")).upper().startswith("CACHE")
                    ),
                )
                continue

            if field_code == "distrito":
                distrito_id = await self._resolve_master_id(
                    session,
                    "mae_distrito",
                    str(value),
                    code_column="ubigeo",
                    minimum_score=0.88,
                    usuario_id=usuario_id,
                    semantic_field_code="distrito",
                )
                cache_meta = await self._semantic_cache_peek(
                    session,
                    field_code="distrito",
                    raw_query=str(value),
                )
                if distrito_id:
                    perfil_updates["distrito_id"] = distrito_id
                    perfil_updates["fuente_ubicacion"] = "usuario"
                else:
                    unresolved_by_field["distrito"] = [str(value)]
                await self._log_profile_extraction(
                    session,
                    usuario_id,
                    field_code,
                    value,
                    now_peru,
                    status="confirmed" if distrito_id else "unresolved",
                    resolved_entity_type=(cache_meta or {}).get("entidad_tipo_resuelta"),
                    resolved_entity_code=(cache_meta or {}).get("entidad_codigo_resuelto"),
                    resolution_strategy=(cache_meta or {}).get("estrategia_usada"),
                    semantic_cache_hit=bool(
                        (cache_meta or {}).get("estrategia_usada")
                        and str((cache_meta or {}).get("estrategia_usada")).upper().startswith("CACHE")
                    ),
                )
                continue


            if field_code == "enfermedades":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                unresolved = await self._sync_enfermedades(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    operation=op,
                    usuario_id=usuario_id,
                )
                if unresolved:
                    unresolved_by_field["enfermedades"] = unresolved
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "alergias":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                unresolved = await self._sync_restricciones(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    only_alergenos=True,
                    operation=op,
                    usuario_id=usuario_id,
                )
                if unresolved:
                    unresolved_by_field["alergias"] = unresolved
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "restricciones_alimentarias":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                unresolved = await self._sync_restricciones(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    only_alergenos=False,
                    operation=op,
                    usuario_id=usuario_id,
                )
                if unresolved:
                    unresolved_by_field["restricciones_alimentarias"] = unresolved
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "provincia":
                provincia_id = await self._resolve_master_id(
                    session,
                    "mae_provincia",
                    str(value),
                    code_column="codigo",
                    minimum_score=0.88,
                    usuario_id=usuario_id,
                    semantic_field_code="provincia",
                )
                cache_meta = await self._semantic_cache_peek(
                    session,
                    field_code="provincia",
                    raw_query=str(value),
                )
                canonical_value = str(value)
                if provincia_id:
                    prov_row = await session.execute(
                        text("SELECT nombre FROM mae_provincia WHERE id = :pid"),
                        {"pid": provincia_id},
                    )
                    canonical_value = str(prov_row.scalar() or value)
                    perfil_updates["provincia_id"] = provincia_id
                else:
                    unresolved_by_field["provincia"] = [str(value)]
                await self._log_profile_extraction(
                    session,
                    usuario_id,
                    field_code,
                    canonical_value,
                    now_peru,
                    status="confirmed" if provincia_id else "unresolved",
                    resolved_entity_type=(cache_meta or {}).get("entidad_tipo_resuelta"),
                    resolved_entity_code=(cache_meta or {}).get("entidad_codigo_resuelto"),
                    resolution_strategy=(cache_meta or {}).get("estrategia_usada"),
                    semantic_cache_hit=bool(
                        (cache_meta or {}).get("estrategia_usada")
                        and str((cache_meta or {}).get("estrategia_usada")).upper().startswith("CACHE")
                    ),
                )
                continue

            if field_code == "region":
                region_id = await self._resolve_master_id(
                    session,
                    "mae_departamento",
                    str(value),
                    code_column="codigo",
                    minimum_score=0.88,
                    usuario_id=usuario_id,
                    semantic_field_code="region",
                )
                cache_meta = await self._semantic_cache_peek(
                    session,
                    field_code="region",
                    raw_query=str(value),
                )
                canonical_value = str(value)
                if region_id:
                    dep_row = await session.execute(
                        text("SELECT nombre FROM mae_departamento WHERE id = :rid"),
                        {"rid": region_id},
                    )
                    canonical_value = str(dep_row.scalar() or value)
                    perfil_updates["departamento_id"] = region_id
                else:
                    unresolved_by_field["region"] = [str(value)]
                await self._log_profile_extraction(
                    session,
                    usuario_id,
                    field_code,
                    canonical_value,
                    now_peru,
                    status="confirmed" if region_id else "unresolved",
                    resolved_entity_type=(cache_meta or {}).get("entidad_tipo_resuelta"),
                    resolved_entity_code=(cache_meta or {}).get("entidad_codigo_resuelto"),
                    resolution_strategy=(cache_meta or {}).get("estrategia_usada"),
                    semantic_cache_hit=bool(
                        (cache_meta or {}).get("estrategia_usada")
                        and str((cache_meta or {}).get("estrategia_usada")).upper().startswith("CACHE")
                    ),
                )
                continue

            await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)

        if perfil_updates:
            set_clauses = ", ".join(f"{k} = :{k}" for k in perfil_updates.keys())
            stmt = f"UPDATE perfil_nutricional SET {set_clauses}, actualizado_en = :now WHERE id = :pid"
            params = {**perfil_updates, "now": now_peru, "pid": perfil_id}
            await session.execute(text(stmt), params)
        else:
            await session.execute(
                text("UPDATE perfil_nutricional SET actualizado_en = :now WHERE id = :pid"),
                {"now": now_peru, "pid": perfil_id},
            )

        logger.info("ProfileExtractionService: Data staged for user=%s in V3 schema", usuario_id)

        # â”€â”€ Trigger: Generar/actualizar orden dietÃ©tica basada en reglas â”€â”€
        # Solo si se actualizaron enfermedades o restricciones y el servicio estÃ¡ disponible.
        _disease_or_restriction_fields = {"enfermedades", "alergias", "restricciones_alimentarias"}
        if self._nutritional_rules and _disease_or_restriction_fields.intersection(updates.keys()):
            try:
                # Esta generacion es aditiva; si falla no debe invalidar el turno principal.
                async with session.begin_nested():
                    await self._nutritional_rules.generate_or_update_dietary_order(session, usuario_id)
            except Exception as e:
                logger.error(
                    "ProfileExtractionService: Non-critical error generating dietary order for user=%s: %s",
                    usuario_id, e,
                )

