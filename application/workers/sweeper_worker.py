"""
Nutribot Backend — SweeperWorker
Procesos periódicos que recuperan jobs/mensajes zombie.
Reinyecta IDs recuperados en Redis para reprocesamiento inmediato.
"""
from __future__ import annotations

import logging
from sqlalchemy import text
from config import get_settings
from infrastructure.redis.client import enqueue, INBOX_QUEUE, OUTBOX_QUEUE

logger = logging.getLogger(__name__)

class SweeperWorker:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def sweep_zombies(self) -> None:
        settings = get_settings()
        timeout = settings.zombie_timeout_minutes

        async with self.session_factory() as session:
            async with session.begin():
                # Rescatar incoming zombies y obtener IDs
                r1 = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'pending', updated_at = NOW(), locked_at = NULL
                        WHERE status = 'processing'
                          AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                        RETURNING id
                    """),
                    {"timeout": timeout},
                )
                inbox_ids = [row.id for row in r1.fetchall()]

                # Rescatar outgoing zombies y obtener IDs
                r2 = await session.execute(
                    text("""
                        UPDATE outgoing_messages
                        SET status = 'pending', updated_at = NOW(), locked_at = NULL
                        WHERE status IN ('processing', 'sending')
                          AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                        RETURNING id
                    """),
                    {"timeout": timeout},
                )
                outbox_ids = [row.id for row in r2.fetchall()]

        total = len(inbox_ids) + len(outbox_ids)
        if total > 0:
            logger.warning(
                "Sweeper recuperó %d zombies (incoming=%d, outgoing=%d)",
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
