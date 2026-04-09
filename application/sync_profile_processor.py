"""
Nutribot Backend — SyncProfileProcessor
Extractor universal y síncrono que protege la base de datos de errores humanos.
"""
import json
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.parsers import parse_weight, parse_height, parse_age, standardize_text_list
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

# Mapeo de campos del LLM a columnas y parsers
# Ahora incluimos ALIASES para que el bot no sea "tonto" con los nombres de campos.
FIELD_CONFIG = {
    # EDAD
    "edad": {"col": "edad", "parser": parse_age},
    "years": {"col": "edad", "parser": parse_age},
    
    # PESO
    "peso": {"col": "peso_kg", "parser": parse_weight},
    "peso_kg": {"col": "peso_kg", "parser": parse_weight},
    "kg": {"col": "peso_kg", "parser": parse_weight},
    
    # TALLA
    "talla": {"col": "altura_cm", "parser": parse_height},
    "altura": {"col": "altura_cm", "parser": parse_height},
    "cm": {"col": "altura_cm", "parser": parse_height},
    
    # ALERGIAS
    "alergias": {"col": "alergias", "parser": standardize_text_list},
    "alergia": {"col": "alergias", "parser": standardize_text_list},
    "allergies": {"col": "alergias", "parser": standardize_text_list},
    
    # ENFERMEDADES
    "enfermedades": {"col": "enfermedades", "parser": standardize_text_list},
    "enfermedad": {"col": "enfermedades", "parser": standardize_text_list},
    "condicion": {"col": "enfermedades", "parser": standardize_text_list},
    "patologia": {"col": "enfermedades", "parser": standardize_text_list},
    
    # DIETA
    "tipo_dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
    "dieta": {"col": "tipo_dieta", "parser": standardize_text_list},
    
    # OBJETIVO
    "objetivo": {"col": "objetivo_nutricional", "parser": standardize_text_list},
    "meta": {"col": "objetivo_nutricional", "parser": standardize_text_list},
    
    # RESTRICCIONES
    "restricciones": {"col": "restricciones_alimentarias", "parser": standardize_text_list},
    "restriccion": {"col": "restricciones_alimentarias", "parser": standardize_text_list},
    
    # UBICACIÓN
    "provincia": {"col": "provincia", "parser": lambda x: x.upper() if x else None},
    "distrito": {"col": "distrito", "parser": lambda x: x.upper() if x else None},
}

EXTRACTION_SYSTEM_PROMPT = """Eres un Analista de Datos experto en COMPRENDER la intención del usuario.
REGLAS DE ORO:
1. SOLO extrae información si el usuario la menciona EXPLÍCITAMENTE. 
2. NUNCA asumas "NINGUNA" para campos ausentes. Solo usa "NINGUNA" si el usuario lo NIEGA (ej: "no tengo nada").
3. Identifica: edad, peso, talla, alergias, enfermedades, restricciones, tipo_dieta, objetivo, provincia, distrito.
4. Responde SOLO en formato JSON.
5. CONTEXTO SITUACIONAL: El usuario está en medio de un proceso de perfil. Te pasaré el 'Paso Actual'. Si la respuesta del usuario es corta o ambigua (ej: solo dice 'mariscos', '70', 'Callao'), ASUME que se refiere al campo indicado en el paso actual.
6. Si dice algo que no tiene nada que ver con el perfil, devuelve un JSON vacío {{}}.
"""

async def process_profile_sync(
    user_text: str, 
    usuario_id: int, 
    session: AsyncSession, 
    openai_client: AsyncOpenAI, 
    model: str,
    current_step: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extrae y guarda datos de perfil de forma síncrona.
    Soporta 'current_step' para desambiguar respuestas cortas (ej: solo 'mariscos').
    """
    # 1. Extracción con LLM
    context_info = f"\nPASO ACTUAL: {current_step}" if current_step else ""
    try:
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT + context_info},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0
        )
        raw_extractions = json.loads(resp.choices[0].message.content)
        logger.info("SyncProfileProcessor: LLM extracted for user=%s: %s", usuario_id, raw_extractions)
    except Exception as e:
        logger.warning("Error en extracción síncrona para user %s: %s", usuario_id, e)
        return {}

    if not raw_extractions:
        return {}

    # 2. Refinamiento determinístico y Persistencia
    clean_data = {}
    now_peru = get_now_peru()
    
    # (Filtro de modo encuesta eliminado para asegurar fluidez en las extracciones de perfil)

    # Palabras de negación explícita
    negation_words = ["no tengo", "ninguna", "nada", "ninguno", "sin", "no como", "no padezco", "no sufro", "no se"]
    text_lower = user_text.lower()
    
    updates = {}
    for key, raw_val in raw_extractions.items():
        # NORMALIZACIÓN: Buscamos en FIELD_CONFIG (que ahora tiene alias)
        key_norm = str(key).lower()
        config = FIELD_CONFIG.get(key_norm)
        
        # --- LÓGICA BULLETPROOF (A prueba de torpezas) ---
        # Si la IA usó una clave que NO conocemos, pero estamos en un paso específico (current_step),
        # y solo hay una extracción o es claramente una respuesta, la forzamos.
        if not config and current_step and raw_val:
            # Buscamos el config del paso actual para ver si podemos "pescar" el dato
            config_step = FIELD_CONFIG.get(current_step.lower())
            if config_step:
                logger.info("Universal Shield: Force-mapping unrecognized key '%s' to current_step '%s'", key, current_step)
                config = config_step

        if not config or not raw_val:
            continue
            
        parser = config["parser"]
        clean_val = parser(str(raw_val))
        
        if clean_val is not None:
            # ESCUDO ANT-AMNESIA: Solo permitir "NINGUNA" si hay mención explícita o negación
            if isinstance(clean_val, str) and clean_val.upper() == "NINGUNA":
                # Si el valor es NINGUNA, verificamos si es coherente con el texto
                field_mentions = [key, config["col"], "alergia", "enfermedad", "restriccion", "dieta", "objetivo", "correo", "asegurado"]
                is_explicit_negation = any(nw in text_lower for nw in negation_words)
                is_field_mentioned = any(fm in text_lower for fm in field_mentions)
                
                if not (is_explicit_negation and is_field_mentioned):
                    logger.warning("Data Shield: blocked NINGUNA for field '%s' — not mentioned in text '%s'", key, user_text[:60])
                    continue
            
            col = config["col"]
            clean_data[col] = clean_val
            updates[col] = clean_val
    
    if updates:
        logger.info("SyncProfileProcessor: Final updates for user=%s: %s", usuario_id, updates)
        
        # Construir SQL dinámico para la actualización
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
