"""
Nutribot Backend — SyncProfileProcessor
Extractor universal y síncrono que protege la base de datos de errores humanos.
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

# Mapeo de campos del LLM a columnas y parsers
FIELD_CONFIG = {
    "edad": {"col": "edad", "parser": parse_age},
    "peso": {"col": "peso_kg", "parser": parse_weight},
    "talla": {"col": "altura_cm", "parser": parse_height},
    "alergias": {"col": "alergias", "parser": standardize_text_list},
    "enfermedades": {"col": "enfermedades", "parser": standardize_text_list},
    "tipo_dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
    "objetivo": {"col": "objetivo_nutricional", "parser": standardize_text_list},
    "region": {"col": "region", "parser": lambda x: x.upper() if x else None},
    "distrito": {"col": "distrito", "parser": lambda x: x.upper() if x else None},
}

EXTRACTION_SYSTEM_PROMPT = """Eres un Analista de Datos experto en extraer información de perfiles nutricionales desde mensajes informales (WhatsApp).
REGLAS:
1. Extrae datos que el usuario mencione sobre SÍ MISMO.
2. Identifica: edad, peso, talla, alergias, enfermedades, tipo_dieta, objetivo, region, distrito.
3. Responde SOLO en formato JSON. 
4. Si el usuario dice que NO tiene alergias o enfermedades, pon "NINGUNA".
5. Extrae el valor textual tal cual lo escribió el usuario (incluyendo errores, ej: "70 quilos", "pemse 80").

EJEMPLO:
Usuario: "ayer me pemse y estab en 70 quilos, ademas soy alergico al mani"
Respuesta: {"peso": "70 quilos", "alergias": "mani"}
"""

async def process_profile_sync(
    user_text: str, 
    usuario_id: int, 
    session: AsyncSession, 
    openai_client: AsyncOpenAI, 
    model: str
) -> Dict[str, Any]:
    """
    Extrae y guarda datos de perfil de forma síncrona.
    Retorna los datos 'limpios' encontrados para feedback opcional del bot.
    """
    # 1. Extracción rápida con LLM (identificar fragmentos)
    try:
        resp = await openai_client.chat.completions.create(
            model=model, # Recomendado gpt-4o-mini por velocidad
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0
        )
        raw_extractions = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.warning("Error en extracción síncrona para user %s: %s", usuario_id, e)
        return {}

    if not raw_extractions:
        return {}

    # 2. Refinamiento determinístico y Persistencia
    clean_data = {}
    now_peru = get_now_peru()
    
    updates = {}
    for key, raw_val in raw_extractions.items():
        config = FIELD_CONFIG.get(key)
        if not config or not raw_val:
            continue
            
        parser = config["parser"]
        clean_val = parser(str(raw_val))
        
        if clean_val is not None:
            col = config["col"]
            clean_data[col] = clean_val
            updates[col] = clean_val

    if updates:
        logger.info("SyncProfileProcessor: Updating user=%s with %s", usuario_id, updates)
        
        # Construir SQL dinámico para la actualización (N para las columnas, M para los valores)
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
        
    return clean_data
