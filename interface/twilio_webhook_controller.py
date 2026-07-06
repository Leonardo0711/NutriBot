"""Twilio WhatsApp webhook ingestion."""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from config import get_settings
from interface.webhook_controller import ingest_evolution_payload

logger = logging.getLogger(__name__)
router = APIRouter()


def _single_values(params: dict[str, list[str]]) -> dict[str, str]:
    return {key: values[0] if values else "" for key, values in params.items()}


def _digits(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _message_type_from_media(content_type: str) -> str:
    content_type = (content_type or "").lower()
    if content_type.startswith("image/"):
        return "imageMessage"
    if content_type.startswith("audio/"):
        return "audioMessage"
    return "conversation"


def normalize_twilio_message(form: dict[str, str]) -> dict[str, Any] | None:
    """Convert Twilio form fields to NutriBot's internal webhook shape."""
    message_sid = (form.get("MessageSid") or form.get("SmsMessageSid") or form.get("SmsSid") or "").strip()
    from_value = form.get("WaId") or form.get("From") or ""
    phone = _digits(from_value)
    body = form.get("Body") or ""
    button_payload = (form.get("ButtonPayload") or "").strip()
    button_text = (form.get("ButtonText") or body or "").strip()

    if not message_sid or not phone:
        return None

    num_media = int(form.get("NumMedia") or "0")
    media_url = form.get("MediaUrl0") or ""
    media_content_type = form.get("MediaContentType0") or ""
    message_type = "conversation"
    message: dict[str, Any] = {"conversation": body}

    if button_payload:
        message = {
            "buttonsResponseMessage": {
                "selectedButtonId": button_payload,
                "selectedDisplayText": button_text,
                "text": button_text,
            }
        }

    if num_media > 0 and media_url:
        message_type = _message_type_from_media(media_content_type)
        if message_type == "imageMessage":
            message = {
                "imageMessage": {
                    "caption": body,
                    "mimetype": media_content_type or "image/jpeg",
                    "twilioMediaUrl": media_url,
                }
            }
        elif message_type == "audioMessage":
            message = {
                "audioMessage": {
                    "mimetype": media_content_type or "audio/ogg",
                    "twilioMediaUrl": media_url,
                }
            }
        else:
            message = {"conversation": body or "[archivo recibido]"}

    return {
        "event": "messages.upsert",
        "provider": "twilio",
        "data": {
            "key": {
                "id": message_sid,
                "remoteJid": f"{phone}@s.whatsapp.net",
                "fromMe": False,
            },
            "messageType": message_type,
            "message": message,
            "twilio": form,
        },
    }


def _validate_twilio_signature(url: str, form: dict[str, str], signature: str, auth_token: str) -> bool:
    data = url + "".join(key + form[key] for key in sorted(form))
    digest = hmac.new(auth_token.encode("utf-8"), data.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


@router.post("/webhook/twilio")
async def receive_twilio_webhook(request: Request):
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    raw_body = await request.body()
    params = parse_qs(raw_body.decode("utf-8", errors="replace"), keep_blank_values=True)
    form = _single_values(params)

    if settings.twilio_validate_signature:
        signature = request.headers.get("X-Twilio-Signature", "")
        if not settings.twilio_auth_token or not signature:
            logger.warning("Twilio webhook rechazado por firma ausente ip=%s", client_ip)
            raise HTTPException(status_code=401, detail="unauthorized")
        if not _validate_twilio_signature(str(request.url), form, signature, settings.twilio_auth_token):
            logger.warning("Twilio webhook rechazado por firma invalida ip=%s", client_ip)
            raise HTTPException(status_code=401, detail="unauthorized")

    payload = normalize_twilio_message(form)
    if not payload:
        logger.warning("Twilio webhook ignorado por payload incompleto ip=%s", client_ip)
        return {"status": "ignored_invalid_payload"}

    return await ingest_evolution_payload(payload, client_ip)


@router.post("/webhook/twilio/status")
async def receive_twilio_status_webhook(request: Request):
    # Por ahora no mutamos outgoing_messages porque Twilio envia estados con
    # MessageSid del proveedor y el outbox ya guarda provider_delivery_id.
    raw_body = await request.body()
    params = parse_qs(raw_body.decode("utf-8", errors="replace"), keep_blank_values=True)
    form = _single_values(params)
    logger.info(
        "Twilio status callback sid=%s status=%s",
        form.get("MessageSid") or form.get("SmsSid") or "-",
        form.get("MessageStatus") or form.get("SmsStatus") or "-",
    )
    return {"status": "ok"}
