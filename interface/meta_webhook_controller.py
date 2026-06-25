"""Meta WhatsApp Cloud API webhook verification and ingestion."""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from config import get_settings
from interface.webhook_controller import ingest_evolution_payload

logger = logging.getLogger(__name__)
router = APIRouter()


def normalize_meta_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Meta message notifications into the established inbox shape."""
    normalized: list[dict[str, Any]] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for message in value.get("messages", []) or []:
                message_id = str(message.get("id") or "")
                phone = str(message.get("from") or "")
                message_type = str(message.get("type") or "")
                if not message_id or not phone or not message_type:
                    continue

                message_obj: dict[str, Any] = {}
                evolution_type = ""
                if message_type == "text":
                    evolution_type = "conversation"
                    message_obj["conversation"] = str(
                        (message.get("text") or {}).get("body") or ""
                    )
                elif message_type == "interactive":
                    interactive = message.get("interactive") or {}
                    if interactive.get("type") == "button_reply":
                        reply = interactive.get("button_reply") or {}
                        evolution_type = "buttonsResponseMessage"
                        message_obj["buttonsResponseMessage"] = {
                            "selectedButtonId": reply.get("id"),
                            "selectedDisplayText": reply.get("title"),
                        }
                    elif interactive.get("type") == "list_reply":
                        reply = interactive.get("list_reply") or {}
                        evolution_type = "listResponseMessage"
                        message_obj["listResponseMessage"] = {
                            "singleSelectReply": {
                                "selectedRowId": reply.get("id"),
                                "title": reply.get("title"),
                            }
                        }
                elif message_type in {"audio", "image"}:
                    media = message.get(message_type) or {}
                    evolution_type = (
                        "audioMessage" if message_type == "audio" else "imageMessage"
                    )
                    media_key = (
                        "audioMessage" if message_type == "audio" else "imageMessage"
                    )
                    message_obj[media_key] = {
                        "mimetype": media.get("mime_type"),
                        "metaMediaId": media.get("id"),
                        "caption": media.get("caption"),
                    }

                if not evolution_type:
                    continue

                normalized.append(
                    {
                        "event": "messages.upsert",
                        "provider": "meta",
                        "data": {
                            "key": {
                                "id": message_id,
                                "remoteJid": f"{phone}@s.whatsapp.net",
                                "fromMe": False,
                            },
                            "messageType": evolution_type,
                            "message": message_obj,
                            "meta": {
                                "metadata": value.get("metadata") or {},
                                "contacts": value.get("contacts") or [],
                                "originalMessage": message,
                            },
                        },
                    }
                )
    return normalized


@router.get("/webhook/meta")
async def verify_meta_webhook(
    mode: str = Query(alias="hub.mode"),
    verify_token: str = Query(alias="hub.verify_token"),
    challenge: str = Query(alias="hub.challenge"),
) -> Response:
    settings = get_settings()
    if (
        mode != "subscribe"
        or not settings.meta_verify_token
        or not hmac.compare_digest(verify_token, settings.meta_verify_token)
    ):
        raise HTTPException(status_code=403, detail="verification_failed")
    return Response(content=challenge, media_type="text/plain")


@router.post("/webhook/meta")
async def receive_meta_webhook(request: Request) -> dict[str, Any]:
    settings = get_settings()
    if not settings.meta_app_secret:
        raise HTTPException(status_code=503, detail="meta_not_configured")

    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    expected = "sha256=" + hmac.new(
        settings.meta_app_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="invalid_signature")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_json") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_payload")

    client_ip = request.client.host if request.client else "unknown"
    messages = normalize_meta_messages(payload)
    results = [
        await ingest_evolution_payload(message, client_ip)
        for message in messages
    ]
    logger.info("Meta webhook recibido ip=%s mensajes=%d", client_ip, len(messages))
    return {"status": "accepted", "messages": len(messages), "results": results}
