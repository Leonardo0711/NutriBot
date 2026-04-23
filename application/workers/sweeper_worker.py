"""
Nutribot Backend - SweeperWorker
Procesos periodicos que recuperan jobs/mensajes zombie.
Reinyecta IDs recuperados en Redis para reprocesamiento inmediato.
"""
from __future__ import annotations

import logging
import math
from sqlalchemy import text
from config import get_settings
from infrastructure.redis.client import enqueue, INBOX_QUEUE, OUTBOX_QUEUE
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

class SweeperWorker:
    def __init__(self, session_factory, openai_client=None, embedding_model: str = "text-embedding-3-small"):
        self.session_factory = session_factory
        self.openai_client = openai_client
        self.embedding_model = embedding_model

    @staticmethod
    def _build_embedding_literal(values: list[float]) -> str | None:
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

    async def process_embedding_jobs(self, max_jobs: int = 8) -> int:
        if not self.openai_client:
            return 0
        processed = 0

        for _ in range(max(1, max_jobs)):
            job = None
            async with self.session_factory() as session:
                async with session.begin():
                    claimed = await session.execute(
                        text(
                            """
                            WITH next_job AS (
                                SELECT id
                                FROM embedding_jobs
                                WHERE estado IN ('PENDING', 'FAILED')
                                  AND retry_count < 5
                                ORDER BY creado_en ASC NULLS LAST, id ASC
                                LIMIT 1
                                FOR UPDATE SKIP LOCKED
                            )
                            UPDATE embedding_jobs ej
                            SET estado = 'PROCESSING',
                                error_detail = NULL,
                                actualizado_en = :now
                            FROM next_job
                            WHERE ej.id = next_job.id
                            RETURNING ej.id, ej.source_table, ej.source_id, ej.texto_fuente, ej.modelo
                            """
                        ),
                        {"now": get_now_peru()},
                    )
                    row = claimed.mappings().first()
                    if row:
                        job = dict(row)

            if not job:
                break

            try:
                model = str(job.get("modelo") or self.embedding_model)
                text_to_embed = str(job.get("texto_fuente") or "").strip()
                if not text_to_embed:
                    raise ValueError("texto_fuente vacio")

                emb_resp = await self.openai_client.embeddings.create(
                    input=[text_to_embed],
                    model=model,
                )
                literal = self._build_embedding_literal(emb_resp.data[0].embedding)
                if not literal:
                    raise ValueError("embedding invalido")

                async with self.session_factory() as session:
                    async with session.begin():
                        if str(job.get("source_table")) == "semantic_catalog":
                            try:
                                await session.execute(
                                    text(
                                        """
                                        UPDATE semantic_catalog
                                        SET embedding = CAST(:emb AS vector),
                                            embedding_model = :model,
                                            actualizado_en = :now
                                        WHERE id = :sid
                                        """
                                    ),
                                    {
                                        "emb": literal,
                                        "model": model[:100],
                                        "sid": int(job.get("source_id")),
                                        "now": get_now_peru(),
                                    },
                                )
                            except Exception:
                                await session.execute(
                                    text(
                                        """
                                        UPDATE semantic_catalog
                                        SET embedding = :emb,
                                            embedding_model = :model,
                                            actualizado_en = :now
                                        WHERE id = :sid
                                        """
                                    ),
                                    {
                                        "emb": literal,
                                        "model": model[:100],
                                        "sid": int(job.get("source_id")),
                                        "now": get_now_peru(),
                                    },
                                )
                        else:
                            raise ValueError(f"source_table no soportada: {job.get('source_table')}")

                        await session.execute(
                            text(
                                """
                                UPDATE embedding_jobs
                                SET estado = 'DONE',
                                    procesado_en = :now,
                                    actualizado_en = :now,
                                    error_detail = NULL
                                WHERE id = :id
                                """
                            ),
                            {"id": int(job.get("id")), "now": get_now_peru()},
                        )

                processed += 1
            except Exception as exc:
                async with self.session_factory() as session:
                    async with session.begin():
                        await session.execute(
                            text(
                                """
                                UPDATE embedding_jobs
                                SET estado = 'FAILED',
                                    retry_count = COALESCE(retry_count, 0) + 1,
                                    error_detail = :err,
                                    actualizado_en = :now
                                WHERE id = :id
                                """
                            ),
                            {
                                "id": int(job.get("id")),
                                "err": f"{type(exc).__name__}: {exc}"[:2000],
                                "now": get_now_peru(),
                            },
                        )
                logger.debug("embedding_jobs fallo id=%s", job.get("id"), exc_info=True)

        return processed

    async def sweep_zombies(self) -> int:
        settings = get_settings()
        timeout = settings.zombie_timeout_minutes

        async with self.session_factory() as session:
            async with session.begin():
                # Rescatar incoming zombies y obtener IDs
                r1 = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'pending', updated_at = TIMEZONE('America/Lima', NOW()), locked_at = NULL
                        WHERE status = 'processing'
                          AND locked_at < TIMEZONE('America/Lima', NOW()) - MAKE_INTERVAL(mins => :timeout)
                        RETURNING id
                    """),
                    {"timeout": timeout},
                )
                inbox_ids = [row.id for row in r1.fetchall()]

                # Rescatar outgoing zombies y obtener IDs
                r2 = await session.execute(
                    text("""
                        UPDATE outgoing_messages
                        SET status = 'pending', updated_at = TIMEZONE('America/Lima', NOW()), locked_at = NULL
                        WHERE status IN ('processing', 'sending')
                          AND locked_at < TIMEZONE('America/Lima', NOW()) - MAKE_INTERVAL(mins => :timeout)
                        RETURNING id
                    """),
                    {"timeout": timeout},
                )
                outbox_ids = [row.id for row in r2.fetchall()]

        total = len(inbox_ids) + len(outbox_ids)
        if total > 0:
            logger.warning(
                "Sweeper recupero %d zombies (incoming=%d, outgoing=%d)",
                total,
                len(inbox_ids),
                len(outbox_ids),
            )

            # Reinyectar en Redis para reprocesamiento inmediato
            for mid in inbox_ids:
                try:
                    await enqueue(INBOX_QUEUE, mid)
                except Exception:
                    pass
            for mid in outbox_ids:
                try:
                    await enqueue(OUTBOX_QUEUE, mid)
                except Exception:
                    pass

        embedding_processed = await self.process_embedding_jobs(max_jobs=8)
        if embedding_processed > 0:
            logger.info("Sweeper proceso %d embedding_jobs", embedding_processed)

        return total + embedding_processed
