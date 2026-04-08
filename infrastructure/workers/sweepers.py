"""
Nutribot Backend — Sweepers
Procesos periódicos que recuperan jobs/mensajes zombie.
Se ejecutan via APScheduler dentro del lifespan de FastAPI.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)


async def sweep_zombies() -> None:
    """
    Marca como 'failed' los registros que llevan demasiado tiempo en 'processing'.
    Timeout conservador de 10 minutos: suficiente para STT + LLM + TTS en picos de red.
    """
    settings = get_settings()
    timeout = settings.zombie_timeout_minutes
    factory = get_session_factory()

    async with factory() as session:
        async with session.begin():
            # 1. Incoming messages zombie
            r1 = await session.execute(
                text("""
                    UPDATE incoming_messages
                    SET status = 'failed', updated_at = NOW()
                    WHERE status = 'processing'
                      AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                """),
                {"timeout": timeout},
            )

            # 2. Outgoing messages zombie
            r2 = await session.execute(
                text("""
                    UPDATE outgoing_messages
                    SET status = 'failed', updated_at = NOW()
                    WHERE status IN ('processing', 'sending')
                      AND locked_at < NOW() - MAKE_INTERVAL(mins => :timeout)
                """),
                {"timeout": timeout},
            )

            # 3. Extraction jobs zombie
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
