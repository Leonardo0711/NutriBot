"""
Nutribot Backend — SweeperWorker
Procesos periódicos que recuperan jobs/mensajes zombie.
"""
from __future__ import annotations

import logging
from sqlalchemy import text
from config import get_settings

logger = logging.getLogger(__name__)

class SweeperWorker:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def sweep_zombies(self) -> None:
        settings = get_settings()
        timeout = settings.zombie_timeout_minutes

        async with self.session_factory() as session:
            async with session.begin():
                r1 = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'failed', updated_at = NOW()
                        WHERE status = 'processing'
                          AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                    """),
                    {"timeout": timeout},
                )

                r2 = await session.execute(
                    text("""
                        UPDATE outgoing_messages
                        SET status = 'failed', updated_at = NOW()
                        WHERE status IN ('processing', 'sending')
                          AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                    """),
                    {"timeout": timeout},
                )

                r3 = await session.execute(
                    text("""
                        UPDATE extraction_jobs
                        SET status = 'failed', updated_at = NOW()
                        WHERE status = 'processing'
                          AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                    """),
                    {"timeout": timeout},
                )

        total = (r1.rowcount or 0) + (r2.rowcount or 0) + (r3.rowcount or 0)
        if total > 0:
            logger.warning(
                "Sweeper recuperó %d zombies (incoming=%d, outgoing=%d, extraction=%d)",
                total,
                r1.rowcount or 0,
                r2.rowcount or 0,
                r3.rowcount or 0,
            )
