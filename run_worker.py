"""
Nutribot Backend — Worker Runner
=================================
Entry point independiente para correr workers sin FastAPI.
Permite escalar API y workers por separado.

Modos soportados:
  python run_worker.py inbox
  python run_worker.py outbox
  python run_worker.py sweeper
  python run_worker.py all          ← equivalente al monolito actual
"""
from __future__ import annotations

import asyncio
import logging
import sys
import signal

from config import get_settings
from infrastructure.db.connection import dispose_engine
from infrastructure.redis.client import close_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Señal de shutdown
_shutdown = asyncio.Event()


def _signal_handler():
    logger.info("Señal de shutdown recibida.")
    _shutdown.set()


async def _periodic_task(coro, interval: float, name: str) -> None:
    """Ejecuta una coroutine periódicamente hasta que se reciba shutdown."""
    while not _shutdown.is_set():
        try:
            await coro()
        except asyncio.CancelledError:
            logger.info("Worker %s detenido.", name)
            return
        except Exception:
            logger.exception("Error en worker %s", name)
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=interval)
            break  # shutdown señalado
        except asyncio.TimeoutError:
            pass  # seguir con el siguiente ciclo


async def run_workers(mode: str) -> None:
    """Arranca los workers según el modo solicitado."""
    # Importar container aquí para que las settings estén cargadas
    from di import container

    settings = container.settings
    tasks: list[asyncio.Task] = []

    if mode in ("inbox", "all"):
        tasks.append(asyncio.create_task(
            _periodic_task(container.inbox_worker.process_inbox, settings.inbox_poll_interval_seconds, "inbox")
        ))
        logger.info("Worker INBOX arrancado (interval=%.1fs)", settings.inbox_poll_interval_seconds)

    if mode in ("outbox", "all"):
        tasks.append(asyncio.create_task(
            _periodic_task(container.outbox_worker.deliver_pending_messages, settings.outbox_poll_interval_seconds, "outbox")
        ))
        logger.info("Worker OUTBOX arrancado (interval=%.1fs)", settings.outbox_poll_interval_seconds)


    if mode in ("sweeper", "all"):
        tasks.append(asyncio.create_task(
            _periodic_task(container.sweeper_worker.sweep_zombies, settings.sweeper_interval_seconds, "sweeper")
        ))
        logger.info("Worker SWEEPER arrancado (interval=%.1fs)", settings.sweeper_interval_seconds)

    if not tasks:
        logger.error("Modo inválido: %s. Use: inbox, outbox, sweeper, all", mode)
        return

    logger.info("Worker runner [%s] activo con %d tasks.", mode, len(tasks))

    # Esperar shutdown
    await _shutdown.wait()

    # Cancelar tareas
    logger.info("Deteniendo %d workers...", len(tasks))
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup
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
    logger.info("Worker runner [%s] detenido.", mode)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    mode = mode.lower()

    loop = asyncio.new_event_loop()

    # Registrar señales de shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows no soporta add_signal_handler
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        loop.run_until_complete(run_workers(mode))
    finally:
        loop.close()


if __name__ == "__main__":
    main()
