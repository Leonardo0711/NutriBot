"""
Nutribot Backend — ProfileExtractionService
Servicio de Aplicación Orientado a Objetos para extraer perfil.
"""
import json
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.parsers import parse_weight, parse_height, parse_age, standardize_text_list
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

class ProfileExtractionService:
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
REGLAS CRÍTICAS DE ROBUSTEZ:
1. PRIORIDAD ABSOLUTA: Si el usuario menciona un dato (ej: 'mido 1.71', 'mi peso es 80'), EXTRAELO siempre.
2. ESCUDO CONTRA DUDAS: Si el usuario hace una PREGUNTA o expresa confusión (ej: '¿Cómo?', '¿Por qué?', '¿Qué es?', 'no entiendo', '¿?', 'qué alergias?'), NO extraigas nada. Devuelve un objeto vacío {}.
3. PESIMISMO OPERATIVO: Ante la menor duda de si el texto es un dato o una consulta, NO extraigas. Es mejor preguntar de nuevo que anotar basura.
4. RESTRICCIONES ALIMENTARIAS: Solo extrae si hay una negación explícita (ej: 'no me gusta el X', 'no como Y', 'no soporto Z', 'evito el W'). Extrae solo el alimento (X, Y, Z, W).
5. ALERGIAS vs RESTRICCIONES: Si el usuario dice 'soy alérgico a X', es una alergia. Si dice 'no me gusta X', es una restricción_alimentaria.
6. PETICIONES vs DATOS (REGLA DE ORO): Si el usuario PIDE algo (ej: 'dame una receta de pescado'), NO extraigas ese alimento como dato de perfil.
7. NEGACIÓN TOTAL: Si el usuario dice 'ninguna', 'nada', 'no tengo' a una pregunta sobre salud/alergia/restricción, extrae 'NINGUNA'.
8. FORMATO: Responde SOLO un objeto JSON PLANO. Si no hay datos claros, responde {}.
10. PROHIBICIÓN DE PREGUNTAS: Si el texto termina en '?' o comienza con palabras interrogativas ('cómo', 'qué', 'por qué', 'para qué', 'cuál', 'cuanto'), responde SIEMPRE con un objeto vacío {}. NO intentes salvar datos de una pregunta.
"""




    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self._openai_client = openai_client
        self._model = model

    async def extract_and_save(
        self,
        user_text: str,
        usuario_id: int,
        session: AsyncSession,
        current_step: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extrae información de perfil de forma síncrona usando IA y persiste en BD.
        Retorna el diccionario de datos limpiados.
        """
        raw_extractions = await self._run_llm_extraction(user_text, current_step)
        return await self.apply_cleaning_and_save(raw_extractions, user_text, usuario_id, session, current_step)
    async def apply_cleaning_and_save(
        self,
        raw_extractions: Dict[str, Any],
        user_text: str,
        usuario_id: int,
        session: AsyncSession,
        current_step: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Limpia datos crudos (extraídos previamente) y los persiste en BD.
        Útil para cuando el Switchboard ya hizo la extracción.
        """
        if not raw_extractions:
            return {}

        clean_data, updates = self._apply_bulletproof_logic(raw_extractions, user_text, current_step)
        
        if updates:
            await self._persist_updates(usuario_id, updates, session)
            
        return clean_data

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
            logger.warning("Error en extracción síncrona por IA: %s", e)
            return {}

    def _apply_bulletproof_logic(self, raw_extractions: dict, user_text: str, current_step: Optional[str]) -> tuple[dict, dict]:
        clean_data = {}
        updates = {}
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
                if isinstance(clean_val, str) and clean_val.upper() == "NINGUNA":
                    # If the current field matches the step we are asking about, ALWAYS allow NINGUNA
                    col_name = config["col"]
                    is_current_step = False
                    if current_step:
                        cs_low = current_step.lower()
                        # Allow partial matches and common aliases
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

                
                col = config["col"]
                clean_data[col] = clean_val
                updates[col] = clean_val
                
        return clean_data, updates

    async def _persist_updates(self, usuario_id: int, updates: dict, session: AsyncSession):
        now_peru = get_now_peru()
        cols_sql = ", ".join(updates.keys())
        placeholders = ", ".join([f":{k}" for k in updates.keys()])
        update_stmt = ", ".join([f"{k} = :{k}" for k in updates.keys()])

        sql = f"""
            INSERT INTO perfil_nutricional (usuario_id, {cols_sql}, actualizado_en)
            VALUES (:uid, {placeholders}, :now)
            ON CONFLICT (usuario_id) DO UPDATE SET 
                {update_stmt},
                actualizado_en = :now
        """
        params = {**updates, "uid": usuario_id, "now": now_peru}
        
        await session.execute(text(sql), params)
        logger.info("ProfileExtractionService: Data staged for user=%s", usuario_id)

