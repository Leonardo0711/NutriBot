"""
Nutribot Backend — ExtractProfileUseCase (Worker)
Reclama extraction_jobs y extrae datos del perfil usando el LLM.
"""
from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analiza el siguiente mensaje de un usuario de WhatsApp y extrae datos personales SI están presentes.
Solo extrae datos que el usuario mencione sobre SÍ MISMO (no sobre terceros).

Campos a buscar:
- correo: dirección de email
- edad: edad en años  
- region: departamento/región de Perú
- peso: peso en kg
- asegurado: si es asegurado de EsSalud (si/no)
- autorizo: si autoriza uso de datos para investigación (si/no)

Para cada dato encontrado, indica:
- field_code: el código del campo
- raw_value: el valor exacto extraído
- confidence: float entre 0 y 1 (qué tan seguro estás de que se refiere a sí mismo)
- evidence_text: el fragmento textual exacto que lo sustenta

Responde en JSON puro, sin markdown. Si no se encontró ningún dato, responde con un array vacío.
Ejemplo: [{"field_code": "edad", "raw_value": "32", "confidence": 0.95, "evidence_text": "tengo 32 años"}]

Mensaje del usuario:
"""


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

    response = await client.responses.create(
        model=model,
        input=prompt,
        instructions="Responde SOLO con un array JSON válido, sin markdown ni explicaciones.",
    )

    raw_output = response.output_text.strip()

    # Parsear JSON de la respuesta
    try:
        # Limpiar posibles backticks
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        extractions = json.loads(raw_output)
    except (json.JSONDecodeError, IndexError):
        logger.warning("LLM retornó JSON inválido para job %s: %s", job.id, raw_output[:200])
        extractions = []

    if not isinstance(extractions, list):
        extractions = []

    # Persistir extracciones con confianza suficiente
    async with factory() as session:
        async with session.begin():
            for ext in extractions:
                confidence = float(ext.get("confidence", 0))
                if confidence < 0.7:
                    continue  # Descarte por baja confianza

                status = "confirmed" if confidence >= 0.85 else "tentative"
                await session.execute(
                    text("""
                        INSERT INTO profile_extractions
                            (usuario_id, field_code, raw_value, confidence, evidence_text, status)
                        VALUES (:uid, :fc, :rv, :conf, :ev, :st)
                    """),
                    {
                        "uid": job.usuario_id,
                        "fc": ext.get("field_code", ""),
                        "rv": ext.get("raw_value", ""),
                        "conf": confidence,
                        "ev": ext.get("evidence_text", ""),
                        "st": status,
                    },
                )

            # Marcar job como done
            await session.execute(
                text("UPDATE extraction_jobs SET status='done', updated_at=NOW() WHERE id=:id"),
                {"id": job.id},
            )

    logger.debug(
        "Extracción completada job=%s: %d campos extraídos",
        job.id,
        len([e for e in extractions if float(e.get("confidence", 0)) >= 0.7]),
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
