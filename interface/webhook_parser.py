"""
Nutribot Backend — Webhook Parser
Convierte el payload JSON crudo de Evolution en una entidad de dominio.
"""
from __future__ import annotations

import logging
from typing import Optional

from domain.entities import IncomingWebhookMessage
from domain.value_objects import MessageType

logger = logging.getLogger(__name__)

# Mapeo de keys de Evolution a nuestro enum
_EVOLUTION_TYPE_MAP: dict[str, MessageType] = {
    "conversation": MessageType.TEXT,
    "extendedTextMessage": MessageType.TEXT,
    "textMessage": MessageType.TEXT,
    "audioMessage": MessageType.AUDIO,
    "pttMessage": MessageType.PTT,
    "imageMessage": MessageType.IMAGE,
}


def parse_evolution_webhook(payload: dict) -> Optional[IncomingWebhookMessage]:
    """
    Parsea un webhook de Evolution API y retorna un IncomingWebhookMessage,
    o None si no es un tipo de mensaje procesable.

    Estructura esperada de Evolution (v2):
    {
      "event": "messages.upsert",
      "data": {
        "key": { "remoteJid": "51999888777@s.whatsapp.net", "fromMe": false, "id": "..." },
        "message": { "conversation": "hola" | "audioMessage": {...} | ... },
        "messageType": "conversation" | "audioMessage" | ...
      }
    }
    """
    # Solo procesar eventos de mensaje entrante
    event = payload.get("event", "")
    if event != "messages.upsert":
        return None

    data = payload.get("data", {})
    key = data.get("key", {})

    # Ignorar mensajes enviados por nosotros
    if key.get("fromMe", False):
        return None

    message_type_raw = data.get("messageType", "")
    content_type = _EVOLUTION_TYPE_MAP.get(message_type_raw)

    if content_type is None:
        # Tipo no soportado (sticker, location, etc.)
        logger.debug("Tipo de mensaje no soportado: %s", message_type_raw)
        return None

    # Extraer número de teléfono
    remote_jid = key.get("remoteJid", "")
    phone = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid

    if not phone:
        return None

    provider_message_id = key.get("id", "") or data.get("message", {}).get("id", "")
    if not provider_message_id:
        return None

    # Extraer contenido según tipo
    message_obj = data.get("message", {})
    text_body: Optional[str] = None
    media_url: Optional[str] = None
    media_mimetype: Optional[str] = None

    if content_type == MessageType.TEXT:
        # El texto puede estar en "conversation" o en "extendedTextMessage.text"
        text_body = message_obj.get("conversation") or message_obj.get(
            "extendedTextMessage", {}
        ).get("text", "")

    elif content_type in (MessageType.AUDIO, MessageType.PTT):
        import json
        audio_data = message_obj.get("audioMessage") or message_obj.get("pttMessage", {})
        # Evolution necesita TODO el nodo "data" para descifrar la media
        media_url = json.dumps(data)
        media_mimetype = audio_data.get("mimetype", "audio/ogg")

    elif content_type == MessageType.IMAGE:
        import json
        image_data = message_obj.get("imageMessage", {})
        # Evolution necesita TODO el nodo "data" para descifrar la media
        media_url = json.dumps(data)
        media_mimetype = image_data.get("mimetype", "image/jpeg")
        # Las imágenes pueden tener caption
        text_body = image_data.get("caption")

    return IncomingWebhookMessage(
        provider_message_id=provider_message_id,
        phone=phone,
        content_type=content_type,
        text_body=text_body,
        media_url=media_url,
        media_mimetype=media_mimetype,
    )
