"""
Nutribot Backend - Webhook Controller
Ingestion rapida con autenticacion, validacion y deduplicacion defensiva.
"""
from __future__ import annotations

import logging
from hmac import compare_digest
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from config import get_settings
from infrastructure.db.connection import get_session_factory
from infrastructure.redis.client import (
    INBOX_QUEUE,
    check_rate_limit,
    enqueue,
    mark_seen,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _extract_message_fields(payload: dict[str, Any]) -> dict[str, Any]:
    event = payload.get("event")
    data = payload.get("data")
    if not isinstance(data, dict):
        return {
            "event": str(event or ""),
            "provider_message_id": "",
            "phone": "",
            "message_type": "",
            "from_me": False,
        }

    key = data.get("key") if isinstance(data.get("key"), dict) else {}
    provider_message_id = str(key.get("id") or data.get("id") or "")
    remote_jid = str(key.get("remoteJid") or "")
    phone = remote_jid.split("@")[0] if remote_jid else ""
    message_type = str(data.get("messageType") or "")
    return {
        "event": str(event or ""),
        "provider_message_id": provider_message_id,
        "phone": phone,
        "message_type": message_type,
        "from_me": bool(key.get("fromMe", False)),
    }


@router.post("/webhook")
async def receive_webhook(request: Request):
    """
    Ingestion ultra-rapida:
    - autentica origen por secreto compartido
    - valida forma minima del payload
    - rate limit por IP y telefono
    - persiste en incoming_messages con barrera UNIQUE final
    """
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"

    if settings.webhook_secret:
        incoming_secret = request.headers.get("X-Webhook-Secret", "")
        if not compare_digest(incoming_secret, settings.webhook_secret):
            logger.warning("Webhook rechazado por secret invalido ip=%s", client_ip)
            raise HTTPException(status_code=401, detail="unauthorized")
    else:
        logger.warning(
            "WEBHOOK_SECRET vacio: aceptando webhook sin autenticacion (modo temporal) ip=%s",
            client_ip,
        )

    try:
        payload = await request.json()
    except Exception:
        logger.warning("Webhook rechazado: JSON invalido ip=%s", client_ip)
        raise HTTPException(status_code=400, detail="invalid_json")

    if not isinstance(payload, dict):
        logger.warning("Webhook rechazado: payload no es objeto ip=%s", client_ip)
        raise HTTPException(status_code=400, detail="invalid_payload")

    fields = _extract_message_fields(payload)
    event = fields["event"]
    provider_message_id = fields["provider_message_id"]
    phone = fields["phone"]
    message_type = fields["message_type"]
    from_me = bool(fields["from_me"])

    allowed_events = {e.strip() for e in settings.webhook_allowed_events.split(",") if e.strip()}
    if event not in allowed_events:
        logger.info("Webhook ignorado por evento no permitido ip=%s event=%s", client_ip, event)
        return {"status": "ignored_event"}

    if not provider_message_id or not phone or not message_type:
        logger.warning(
            "Webhook ignorado por payload incompleto ip=%s event=%s mid=%s phone=%s type=%s",
            client_ip,
            event,
            provider_message_id or "-",
            phone or "-",
            message_type or "-",
        )
        return {"status": "ignored_invalid_payload"}

    if from_me:
        logger.info("Webhook ignorado (fromMe=true) ip=%s mid=%s", client_ip, provider_message_id)
        return {"status": "ignored_outbound"}

    rate_limited = False
    try:
        ip_allowed = True
        if settings.webhook_rate_limit_max_ip > 0:
            ip_allowed = await check_rate_limit(
                subject=client_ip,
                category="webhook_ip",
                max_count=settings.webhook_rate_limit_max_ip,
                window_seconds=settings.webhook_rate_limit_window_seconds,
            )
        phone_allowed = await check_rate_limit(
            subject=phone,
            category="webhook_phone",
            max_count=settings.webhook_rate_limit_max_phone,
            window_seconds=settings.webhook_rate_limit_window_seconds,
        )
        rate_limited = not (ip_allowed and phone_allowed)
        if rate_limited:
            logger.warning("Webhook rate_limited ip=%s phone=%s mid=%s", client_ip, phone, provider_message_id)
    except Exception:
        logger.exception("Rate limiting no disponible ip=%s phone=%s", client_ip, phone)

    row = None
    insert_stmt = text(
        """
        INSERT INTO incoming_messages (provider_message_id, webhook_payload)
        VALUES (:mid, :pay)
        ON CONFLICT (provider_message_id) DO NOTHING
        RETURNING id
        """
    ).bindparams(bindparam("pay", type_=JSONB))

    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            result = await session.execute(
                insert_stmt,
                {"mid": provider_message_id, "pay": payload},
            )
            row = result.fetchone()

            if row and rate_limited:
                await session.execute(
                    text(
                        """
                        UPDATE incoming_messages
                        SET status = 'done',
                            error_detail = :detail,
                            updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {"id": row.id, "detail": f"rate_limited ip={client_ip} phone={phone}"[:500]},
                )

    if not row:
        # Duplicado definitivo detectado por UNIQUE en DB.
        logger.info("Webhook duplicado en DB ip=%s mid=%s", client_ip, provider_message_id)
        return {"status": "duplicate"}

    # Best effort cache de deduplicacion: no debe bloquear ni decidir flujo.
    try:
        await mark_seen(provider_message_id, ttl_seconds=settings.webhook_dedup_ttl_seconds)
    except Exception:
        logger.exception("No se pudo marcar dedup cache ip=%s mid=%s", client_ip, provider_message_id)

    if not rate_limited:
        try:
            await enqueue(INBOX_QUEUE, row.id)
        except Exception:
            logger.exception("No se pudo encolar inbox id=%s (fallback SQL activo)", row.id)

    logger.info(
        "Webhook ingestado ip=%s event=%s mid=%s inbox_id=%s rate_limited=%s",
        client_ip,
        event,
        provider_message_id,
        row.id,
        rate_limited,
    )
    return {"status": "rate_limited" if rate_limited else "ok"}
