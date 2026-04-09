"""
Nutribot Backend — Main Application
FastAPI con lifespan para arrancar/detener workers de background usando OOP.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.db.connection import dispose_engine
from interface.webhook_controller import router as webhook_router
from di import container

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
    Lifespan: arranca workers de background al iniciar y los detiene al cerrar.
    """
    settings = container.settings
    logger.info("Nutribot Backend iniciando (OOP Refactor)...")
    logger.info("Workers: inbox=%.1fs, outbox=%.1fs, extraction=%.1fs, sweeper=%.1fs",
                settings.inbox_poll_interval_seconds,
                settings.outbox_poll_interval_seconds,
                settings.extraction_poll_interval_seconds,
                settings.sweeper_interval_seconds)

    tasks: list[asyncio.Task] = []

    # Sweeper de zombies
    tasks.append(
        asyncio.create_task(
            _periodic_task(container.sweeper_worker.sweep_zombies, settings.sweeper_interval_seconds, "sweeper")
        )
    )

    # Worker: Inbox (procesa webhooks pendientes)
    tasks.append(
        asyncio.create_task(
            _periodic_task(container.inbox_worker.process_inbox, settings.inbox_poll_interval_seconds, "inbox")
        )
    )

    # Worker: Outbox (envía respuestas + TTS)
    tasks.append(
        asyncio.create_task(
            _periodic_task(container.outbox_worker.deliver_pending_messages, settings.outbox_poll_interval_seconds, "outbox")
        )
    )

    # Worker: Extraction (extrae perfil en background)
    tasks.append(
        asyncio.create_task(
            _periodic_task(container.extraction_worker.process_extractions, settings.extraction_poll_interval_seconds, "extraction")
        )
    )

    logger.info("Workers arrancados: %d", len(tasks))

    yield

    # Shutdown: cancelar todos los workers
    logger.info("Deteniendo %d workers...", len(tasks))
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cerrar pool de conexiones
    await dispose_engine()
    logger.info("Nutribot Backend detenido.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Nutribot Backend",
    version="2.1.0",
    lifespan=lifespan,
)

app.include_router(webhook_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0", "refactor": "oop"}
