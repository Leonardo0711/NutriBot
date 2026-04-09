"""
Nutribot Backend — ExtractionWorker
Worker de extracción: reclama jobs pendientes y extrae perfil en background.
"""
from __future__ import annotations

import json
import logging
from openai import AsyncOpenAI
from sqlalchemy import text

from config import get_settings

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

class ExtractionWorker:
    def __init__(self, session_factory, openai_client: AsyncOpenAI, model: str):
        self.session_factory = session_factory
        self.openai_client = openai_client
        self.model = model

    async def process_extractions(self) -> int:
        settings = get_settings()

        async with self.session_factory() as session:
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
                await self._extract_single(job)
                processed += 1
            except Exception as e:
                logger.exception("Error extrayendo perfil job id=%s: %s", job.id, e)
                await self._mark_job_failed(job.id)

        return processed

    async def _extract_single(self, job) -> None:
        prompt = EXTRACTION_PROMPT + job.raw_text

        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )

        raw_output = response.choices[0].message.content.strip()

        try:
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

        async with self.session_factory() as session:
            async with session.begin():
                for ext in extractions:
                    field_code = ext.get("field_code", "").lower()
                    raw_value = str(ext.get("raw_value", ""))
                    confidence = float(ext.get("confidence", 0))
                    evidence = ext.get("evidence_text", "")
                    
                    if confidence < 0.7:
                        continue

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

                await session.execute(
                    text("UPDATE extraction_jobs SET status='done', updated_at=NOW() WHERE id=:id"),
                    {"id": job.id},
                )

    async def _mark_job_failed(self, job_id: int) -> None:
        async with self.session_factory() as session:
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
