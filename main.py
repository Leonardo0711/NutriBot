"""
Nutribot Backend — Main Application
====================================
FastAPI application con dos modos de operación:

  NUTRIBOT_MODE=monolith  → API + workers (backward-compatible, default)
  NUTRIBOT_MODE=api       → Solo API (webhook + health), los workers corren aparte

En producción separada:
  - Este proceso sirve HTTP (uvicorn main:app)
  - Los workers corren con: python run_worker.py inbox|outbox|sweeper
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.db.connection import dispose_engine
from infrastructure.redis.client import close_redis
from interface.webhook_controller import router as webhook_router
from di import container

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NUTRIBOT_MODE = os.environ.get("NUTRIBOT_MODE", "monolith").lower()


async def _periodic_task(coro, interval: float, name: str) -> None:
    """Ejecuta una coroutine periódicamente con manejo de errores."""
    while True:
        try:
            await coro()
        except asyncio.CancelledError:
            logger.info("Worker %s detenido.", name)
            return
        except Exception:
            logger.exception("Error en worker %s", name)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan: en modo monolith arranca workers; en modo api solo sirve HTTP.
    """
    settings = container.settings
    logger.info("Nutribot Backend iniciando (mode=%s)...", NUTRIBOT_MODE)

    tasks: list[asyncio.Task] = []

    if NUTRIBOT_MODE == "monolith":
        # Backward-compatible: API + workers en el mismo proceso
        logger.info("Workers embebidos: inbox=%.1fs, outbox=%.1fs, sweeper=%.1fs",
                    settings.inbox_poll_interval_seconds,
                    settings.outbox_poll_interval_seconds,
                    settings.sweeper_interval_seconds)

        tasks.append(asyncio.create_task(
            _periodic_task(container.sweeper_worker.sweep_zombies, settings.sweeper_interval_seconds, "sweeper")
        ))
        tasks.append(asyncio.create_task(
            _periodic_task(container.inbox_worker.process_inbox, settings.inbox_poll_interval_seconds, "inbox")
        ))
        tasks.append(asyncio.create_task(
            _periodic_task(container.outbox_worker.deliver_pending_messages, settings.outbox_poll_interval_seconds, "outbox")
        ))

        logger.info("Workers arrancados: %d", len(tasks))
    else:
        logger.info("Modo API-only: workers desactivados (corren en procesos separados).")

    yield

    # Shutdown
    if tasks:
        logger.info("Deteniendo %d workers...", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Cierre de clientes HTTP
    try:
        await container.openai_client.close()
    except Exception:
        pass
    try:
        await container.evolution_client.close()
    except Exception:
        pass

    await close_redis()
    await dispose_engine()
    logger.info("Nutribot Backend detenido.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Nutribot Backend",
    version="2.3.0",
    lifespan=lifespan,
)

app.include_router(webhook_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.3.0",
        "mode": NUTRIBOT_MODE,
        "refactor": "oop+redis+router+decouple",
    }


@app.get("/health/queues")
async def health_queues():
    """Monitoreo en vivo de las colas Redis."""
    from infrastructure.redis.client import queue_depth, INBOX_QUEUE, OUTBOX_QUEUE
    try:
        return {
            "status": "ok",
            "queues": {
                "inbox": await queue_depth(INBOX_QUEUE),
                "outbox": await queue_depth(OUTBOX_QUEUE),
            }
        }
    except Exception as e:
        return {"status": "redis_unavailable", "error": str(e)}
