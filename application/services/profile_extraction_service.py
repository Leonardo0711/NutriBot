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
            await self._persist_updates(usuario_id, clean_data, session)
            
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
            await self._persist_updates(usuario_id, clean_data, session)
            
        return ExtractionResult(clean_data=clean_data, updates=updates, meta_flags=meta_flags)

    async def save_clean_data(self, usuario_id: int, clean_data: Dict[str, Any], session: AsyncSession):
        """Persiste directamente datos ya limpios."""
        if clean_data:
            await self._persist_updates(usuario_id, clean_data, session)

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

    async def _set_current_measurement(
        self,
        session: AsyncSession,
        perfil_id: int,
        measurement_type: str,
        numeric_value: float,
        unit: str,
        now_peru,
    ) -> None:
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
                    (:pid, :mtype, :val, :unit, :now, TRUE, 'SELF_REPORT', :now)
                """
            ),
            {"pid": perfil_id, "mtype": measurement_type, "val": numeric_value, "unit": unit, "now": now_peru},
        )

    async def _sync_enfermedades(
        self,
        session: AsyncSession,
        perfil_id: int,
        raw_value: str,
        now_peru,
    ) -> None:
        today = now_peru.date()
        if self._is_negative_value(raw_value):
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

        if self._is_negative_value(raw_value):
            await session.execute(text(deactivate_query), {"pid": perfil_id, "today": today})
            return

        values = self._split_values(raw_value)
        if not values:
            return

        await session.execute(text(deactivate_query), {"pid": perfil_id, "today": today})

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

    async def _persist_updates(self, usuario_id: int, updates: dict, session: AsyncSession):
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
                    await self._set_current_measurement(
                        session,
                        perfil_id,
                        "PESO_KG",
                        float(value),
                        "kg",
                        now_peru,
                    )
                except (TypeError, ValueError):
                    logger.info("Peso invalido para persistencia V3: %s", value)
                continue

            if field_code == "altura_cm":
                try:
                    await self._set_current_measurement(
                        session,
                        perfil_id,
                        "ALTURA_CM",
                        float(value),
                        "cm",
                        now_peru,
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
                await self._sync_enfermedades(session, perfil_id, str(value), now_peru)
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "alergias":
                await self._sync_restricciones(session, perfil_id, str(value), now_peru, only_alergenos=True)
                await self._log_profile_extraction(session, usuario_id, field_code, value, now_peru)
                continue

            if field_code == "restricciones_alimentarias":
                await self._sync_restricciones(session, perfil_id, str(value), now_peru, only_alergenos=False)
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
