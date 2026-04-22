"""
Nutribot Backend — ProfileExtractionService
Servicio de Aplicación Orientado a Objetos para extraer perfil.
"""
import json
import logging
import re
import unicodedata
from typing import Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from sqlalchemy import text
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
        "me equivoqué",
        "corrijo",
        "correccion",
        "corrección",
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
        "subí a",
        "baje a",
        "bajé a",
        "ultimo peso",
        "último peso",
        "mi ultimo peso",
        "mi último peso",
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
        "añade",
        "suma",
        "tambien",
        "también",
        "ademas",
        "además",
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

    EXTRACTION_SYSTEM_PROMPT = """Eres un Analista de Datos experto en COMPRENDER la intención del usuario para Nutribot.
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
        ("diabetes", "¿Te refieres a tipo 1, tipo 2 o gestacional?"),
        ("anemia", "¿Te refieres a algún tipo específico de anemia?"),
        ("tiroides", "¿Te refieres a hipotiroidismo o hipertiroidismo?"),
        ("problemas hormonales", "¿Te refieres a algún tipo de problema hormonal específico?"),
        ("colon", "¿Te refieres a colitis, colon irritable u otra condición similar?"),
        ("presion", "¿Te refieres a hipertensión o hipotensión?"),
        ("gastritis", "¿Te refieres a gastritis aguda o crónica?"),
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

    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self._openai_client = openai_client
        self._model = model

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
        }
        valid = aliases.get(field, {field})
        return step in valid

    def _infer_measurement_correction_mode(
        self,
        *,
        source_text: str,
        field_code: str,
        current_step: Optional[str],
        current_measurement_row: Optional[dict[str, Any]],
        now_peru,
    ) -> bool:
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

        if has_correction_marker and not has_time_change_marker:
            return True
        if matches_step:
            return True
        if recent_current and has_correction_marker:
            return True
        return False

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
            candidate = re.sub(
                r"^(?:ya no|no|tengo|padezco|sufro de|soy alergic[oa] a|alergia a|evito|no como|no consumo)\s+",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip(" .:-")
            candidate = re.sub(r"\b(?:por favor|gracias)$", "", candidate, flags=re.IGNORECASE).strip(" .:-")
            if not candidate:
                continue
            key = cls._normalize_text(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

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
    ) -> Optional[int]:
        if not raw_value:
            return None
        target_norm = self._normalize_text(raw_value)
        if not target_norm or self._is_negative_value(raw_value):
            return None

        query = text(
            f"""
            SELECT id, {code_column} AS codigo, {name_column} AS nombre
            FROM {table_name}
            WHERE activo = TRUE
              AND {extra_where}
            """
        )
        rows = (await session.execute(query)).mappings().all()
        best_id: Optional[int] = None
        best_score = 0.0
        for row in rows:
            score = self._score_master_candidate(target_norm, row.get("codigo"), row.get("nombre"))
            if score > best_score:
                best_score = score
                best_id = row.get("id")
        if best_score >= minimum_score:
            return best_id
        return None

    async def _log_profile_extraction(
        self,
        session: AsyncSession,
        usuario_id: int,
        field_code: str,
        raw_value: Any,
        now_peru,
    ) -> None:
        if raw_value is None:
            return
        raw_text = str(raw_value).strip()
        if not raw_text:
            return
        await session.execute(
            text(
                """
                INSERT INTO profile_extractions
                    (usuario_id, field_code, raw_value, normalized_value, confidence, evidence_text, status, extracted_at)
                VALUES
                    (:uid, :field, :raw, :normalized, 1.0, 'Automatic extraction', 'confirmed', :now)
                """
            ),
            {
                "uid": usuario_id,
                "field": field_code,
                "raw": raw_text,
                "normalized": self._normalize_text(raw_text)[:500],
                "now": now_peru,
            },
        )

    async def _validate_semantic(
        self, 
        session: AsyncSession, 
        categoria: str, 
        value: str, 
        threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """Busca similitud semántica en el catálogo maestro."""
        if not value or value.upper() == "NINGUNA":
            return None

        # Generar embedding del valor extraído
        try:
            resp = await self._openai_client.embeddings.create(
                input=[value],
                model="text-embedding-3-small"
            )
            embedding = resp.data[0].embedding
        except Exception as e:
            logger.error(f"Error generando embedding para validación: {e}")
            return None

        # Búsqueda vectorial en BD
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
        # El escudo de términos absurdos sigue siendo útil para filtrado rápido pre-BD
        return any(term in txt for term in cls.ABSURD_TERMS)

    def _check_health_ambiguity(self, value: str) -> Optional[str]:
        """Retorna un prompt de aclaracion si el valor es ambiguo, None si esta completo."""
        norm = value.lower().strip()
        for pattern, prompt in self._AMBIGUOUS_CONDITIONS:
            if pattern in norm:
                has_specificity = any(marker in norm for marker in self._SPECIFICITY_MARKERS)
                if not has_specificity:
                    return f"¡Entendido! Lo anoté provisionalmente de este lado. Para ser más certeros, {prompt}"
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
            await self._persist_updates(
                usuario_id,
                clean_data,
                session,
                source_text=user_text,
                current_step=current_step,
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
            await self._persist_updates(
                usuario_id,
                clean_data,
                session,
                source_text=user_text,
                current_step=current_step,
            )
            
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

                # Blindaje anti-datos absurdos en campos criticos de salud/perfil.
                if col_name in {"alergias", "enfermedades", "restricciones_alimentarias"}:
                    if self.contains_absurd_claim(str(clean_val)):
                        logger.warning(
                            "Data Shield: blocked implausible value for field '%s': %s",
                            col_name,
                            clean_val,
                        )
                        meta_flags["warnings"].append(
                            f"Valor implausible bloqueado para {col_name}: {clean_val}"
                        )
                        continue

                # Validacion clinica generica: detectar datos ambiguos
                if col_name in {"enfermedades", "alergias"}:
                    prompt = self._check_health_ambiguity(str(clean_val))
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
    ) -> None:
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
            return

        values = self._split_values(raw_value)
        if not values:
            return

        resolved_ids: list[int] = []
        for value in values:
            enfermedad_id = await self._resolve_master_id(
                session,
                "mae_enfermedad_cie10",
                value,
                code_column="codigo_cie10",
                minimum_score=0.88,
            )
            if not enfermedad_id:
                logger.info("No se encontro enfermedad en maestro para valor='%s'", value)
                continue
            resolved_ids.append(enfermedad_id)

        if not resolved_ids:
            return

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
            return

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

    async def _sync_restricciones(
        self,
        session: AsyncSession,
        perfil_id: int,
        raw_value: str,
        now_peru,
        *,
        only_alergenos: bool,
        operation: str,
    ) -> None:
        today = now_peru.date()
        where_master = "tipo = 'ALERGENO'" if only_alergenos else "TRUE"

        deactivate_query = """
            UPDATE perfil_nutricional_restriccion pr
            SET vigente = FALSE, fecha_fin = :today
            WHERE pr.perfil_nutricional_id = :pid
              AND pr.vigente = TRUE
        """
        if only_alergenos:
            deactivate_query += """
              AND pr.restriccion_id IN (
                    SELECT id FROM mae_restriccion_alimentaria WHERE tipo = 'ALERGENO'
              )
            """

        if operation == self.OP_CLEAR:
            await session.execute(text(deactivate_query), {"pid": perfil_id, "today": today})
            return

        values = self._split_values(raw_value)
        if not values:
            return

        resolved_ids: list[int] = []
        for value in values:
            restriccion_id = await self._resolve_master_id(
                session,
                "mae_restriccion_alimentaria",
                value,
                extra_where=where_master,
                minimum_score=0.88,
            )
            if not restriccion_id:
                logger.info("No se encontro restriccion en maestro para valor='%s'", value)
                continue
            resolved_ids.append(restriccion_id)

        if not resolved_ids:
            return

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
                        SELECT id FROM mae_restriccion_alimentaria WHERE tipo = 'ALERGENO'
                  )
                """
            await session.execute(text(remove_query), {"pid": perfil_id, "today": today, "ids": resolved_ids})
            return

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

    async def _persist_updates(
        self,
        usuario_id: int,
        updates: dict,
        session: AsyncSession,
        *,
        source_text: str = "",
        current_step: Optional[str] = None,
    ):
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
                    correction_mode = self._infer_measurement_correction_mode(
                        source_text=source_text,
                        field_code=field_code,
                        current_step=current_step,
                        current_measurement_row=current_row,
                        now_peru=now_peru,
                    )
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
                    correction_mode = self._infer_measurement_correction_mode(
                        source_text=source_text,
                        field_code=field_code,
                        current_step=current_step,
                        current_measurement_row=current_row,
                        now_peru=now_peru,
                    )
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
                patron_id = await self._resolve_master_id(session, "mae_patron_alimentario", str(value))
                if patron_id:
                    perfil_updates["patron_alimentario_id"] = patron_id
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "objetivo_nutricional":
                objetivo_id = await self._resolve_master_id(session, "mae_objetivo_nutricional", str(value))
                if objetivo_id:
                    perfil_updates["objetivo_nutricional_id"] = objetivo_id
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "distrito":
                distrito_id = await self._resolve_master_id(
                    session,
                    "mae_distrito",
                    str(value),
                    code_column="ubigeo",
                    minimum_score=0.88,
                )
                if distrito_id:
                    perfil_updates["distrito_id"] = distrito_id
                    perfil_updates["fuente_ubicacion"] = "usuario"
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "enfermedades":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                await self._sync_enfermedades(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    operation=op,
                )
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "alergias":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                await self._sync_restricciones(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    only_alergenos=True,
                    operation=op,
                )
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "restricciones_alimentarias":
                op = self._infer_list_operation(
                    raw_value=str(value),
                    source_text=source_text,
                    field_code=field_code,
                    current_step=current_step,
                )
                await self._sync_restricciones(
                    session,
                    perfil_id,
                    str(value),
                    now_peru,
                    only_alergenos=False,
                    operation=op,
                )
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code in {"provincia", "region"}:
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
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
