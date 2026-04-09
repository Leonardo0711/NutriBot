"""
Nutribot Backend — ExtractProfileUseCase (Worker)
Reclama extraction_jobs y extrae datos del perfil usando el LLM.
"""
from __future__ import annotations

import json
import logging

import re
from openai import AsyncOpenAI
from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analiza el siguiente mensaje de un usuario de WhatsApp y extrae datos personales SI están presentes.
Solo extrae datos que el usuario mencione sobre SÍ MISMO (no sobre terceros).

Campos a buscar:
- edad: edad en años
- peso: peso en kg
- altura: altura en cm
- alergias: alergias o intolerancias alimentarias (ej: maní, mariscos, gluten, lactosa)
- enfermedades: condiciones de salud relevantes (ej: diabetes, hipertensión)
- tipo_dieta: tipo de dieta que sigue (ej: vegano, vegetariano, keto)
- objetivo_nutricional: qué busca lograr (ej: bajar peso, ganar masa, comer sano)
- region: departamento/región de Perú (ej: Lima, Callao, Arequipa)
- provincia: provincia (ej: Lima, Callao, Cusco)
- distrito: distrito (ej: Miraflores, San Borja, Los Olivos)

Para cada dato encontrado, indica:
- field_code: el código del campo
- raw_value: el valor exacto extraído
- confidence: float entre 0 y 1 (qué tan seguro estás de que se refiere a sí mismo y el dato es claro)
- evidence_text: el fragmento textual exacto que lo sustenta

Responde en JSON puro, sin markdown. Si no se encontró ningún dato, responde con un array vacío.
Ejemplo: [{"field_code": "alergias", "raw_value": "Mani", "confidence": 0.95, "evidence_text": "soy alergico al mani"}]

Mensaje del usuario:
"""

# Mapeo de field_code a columnas de la tabla perfil_nutricional
FIELD_TO_COL = {
    "edad": "edad",
    "peso": "peso_kg",
    "altura": "altura_cm",
    "alergias": "alergias",
    "enfermedades": "enfermedades",
    "tipo_dieta": "tipo_dieta",
    "objetivo_nutricional": "objetivo_nutricional",
    "region": "region",
    "provincia": "provincia",
    "distrito": "distrito"
}

async def process_extractions() -> int:
    """
    Worker de extracción: reclama jobs pendientes y extrae perfil.
    Retorna la cantidad de jobs procesados.
    """
    settings = get_settings()
    factory = get_session_factory()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # ─── Transacción Corta: Reclamar jobs ───
    async with factory() as session:
        async with session.begin():
            result = await session.execute(
                text("""
                    UPDATE extraction_jobs
                    SET status = 'processing',
                        locked_at = NOW(),
                        updated_at = NOW()
                    WHERE id IN (
                        SELECT id FROM extraction_jobs
                        WHERE status IN ('pending', 'failed')
                          AND retry_count < :max_retry
                        ORDER BY created_at ASC
                        LIMIT 10
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, raw_text, usuario_id
                """),
                {"max_retry": settings.max_retry_count},
            )
            jobs = result.fetchall()

    if not jobs:
        return 0

    processed = 0
    for job in jobs:
        try:
            await _extract_single(job, factory, client, settings.openai_model)
            processed += 1
        except Exception as e:
            logger.exception("Error extrayendo perfil job id=%s: %s", job.id, e)
            await _mark_job_failed(factory, job.id)

    return processed


async def _extract_single(job, factory, client: AsyncOpenAI, model: str) -> None:
    """Extrae datos de perfil de un texto individual."""
    prompt = EXTRACTION_PROMPT + job.raw_text

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
    )

    raw_output = response.choices[0].message.content.strip()

    # Parsear JSON de la respuesta
    try:
        # Limpiar posibles backticks
        if raw_output.startswith("```"):
            if "\n" in raw_output:
                raw_output = raw_output.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            else:
                raw_output = raw_output.strip("`")
        extractions = json.loads(raw_output)
    except (json.JSONDecodeError, IndexError):
        logger.warning("LLM retornó JSON inválido para job %s: %s", job.id, raw_output[:200])
        extractions = []

    if not isinstance(extractions, list):
        extractions = []

    # Persistir y SINCRONIZAR directamente con perfil_nutricional si hay alta confianza
    async with factory() as session:
        async with session.begin():
            for ext in extractions:
                field_code = ext.get("field_code", "").lower()
                raw_value = str(ext.get("raw_value", ""))
                confidence = float(ext.get("confidence", 0))
                evidence = ext.get("evidence_text", "")
                
                if confidence < 0.7:
                    continue  # Descarte total

                # 1. Guardar log de extracción
                status = "confirmed" if confidence >= 0.85 else "tentative"
                await session.execute(
                    text("""
                        INSERT INTO profile_extractions
                            (usuario_id, field_code, raw_value, confidence, evidence_text, status)
                        VALUES (:uid, :fc, :rv, :conf, :ev, :st)
                    """),
                    {
                        "uid": job.usuario_id,
                        "fc": field_code,
                        "rv": raw_value,
                        "conf": confidence,
                        "ev": evidence,
                        "st": status,
                    },
                )

                # 3. La sincronización directa a Perfil ahora es SÍNCRONA
                # en handle_incoming_message (via SyncProfileProcessor).
                # Ya no es necesario que el worker lo haga para evitar race conditions.
                logger.debug("extract_profile: extraction logged, bypassing direct sink to perfil_nutricional (handled by sync layer)")

            # Marcar job como done
            await session.execute(
                text("UPDATE extraction_jobs SET status='done', updated_at=NOW() WHERE id=:id"),
                {"id": job.id},
            )

    logger.debug(
        "Extracción completada job=%s: %d campos procesados",
        job.id, len(extractions)
    )


async def _mark_job_failed(factory, job_id: int) -> None:
    async with factory() as session:
        async with session.begin():
            await session.execute(
                text("""
                    UPDATE extraction_jobs
                    SET status = 'failed',
                        retry_count = retry_count + 1,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {"id": job_id},
            )
