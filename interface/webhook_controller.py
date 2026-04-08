"""
Nutribot Backend — Webhook Controller
Fase A del Inbox: recibe webhooks de Evolution y los almacena en BD.
No procesa nada — solo ingestión ultrarrápida.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook")
async def receive_webhook(request: Request):
    """
    Ingestión de Buzón Ultra Rápida.
    - Extrae provider_message_id del payload.
    - Si no hay ID (eventos de estado, ack, etc.), ignora.
    - Inserta en incoming_messages con ON CONFLICT DO NOTHING.
    - Retorna 200 inmediato.
    """
    payload = await request.json()

    event = payload.get("event", "unknown")

    # Extraer el ID del mensaje del proveedor
    data = payload.get("data", {})
    provider_message_id = None

    # Evolution API pone el ID en data.key.id
    if isinstance(data.get("key"), dict):
        provider_message_id = data["key"].get("id")
    elif data.get("id"):
        provider_message_id = data.get("id")

    if not provider_message_id:
        return {"status": "ignored"}

    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            await session.execute(
                text("""
                    INSERT INTO incoming_messages (provider_message_id, webhook_payload)
                    VALUES (:mid, :pay)
                    ON CONFLICT (provider_message_id) DO NOTHING
                """),
                {"mid": provider_message_id, "pay": json.dumps(payload)},
            )

    logger.debug("Webhook ingestado: %s", provider_message_id)
    return {"status": "ok"}
