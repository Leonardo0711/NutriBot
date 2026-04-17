"""
Nutribot Backend - Webhook Parser
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from domain.entities import IncomingWebhookMessage
from domain.value_objects import MessageType

logger = logging.getLogger(__name__)

_EVOLUTION_TYPE_MAP: dict[str, MessageType] = {
    "conversation": MessageType.TEXT,
    "extendedTextMessage": MessageType.TEXT,
    "textMessage": MessageType.TEXT,
    "audioMessage": MessageType.AUDIO,
    "pttMessage": MessageType.PTT,
    "imageMessage": MessageType.IMAGE,
}


def _extract_interactive_response(message_obj: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Extract stable IDs for button/list replies from known WhatsApp payload shapes.
    """
    try:
        buttons = message_obj.get("buttonsResponseMessage", {}) or {}
        if buttons.get("selectedButtonId"):
            return str(buttons["selectedButtonId"]), buttons.get("selectedDisplayText") or buttons.get("text")

        template = message_obj.get("templateButtonReplyMessage", {}) or {}
        if template.get("selectedId"):
            return str(template["selectedId"]), template.get("selectedDisplayText")

        list_msg = message_obj.get("listResponseMessage", {}) or {}
        single = list_msg.get("singleSelectReply", {}) or {}
        if single.get("selectedRowId"):
            return str(single["selectedRowId"]), single.get("title") or list_msg.get("title")

        interactive = message_obj.get("interactiveResponseMessage", {}) or {}
        native_flow = interactive.get("nativeFlowResponseMessage", {}) or {}
        params_json = native_flow.get("paramsJson")
        if params_json:
            parsed = json.loads(params_json)
            selected_id = parsed.get("id") or parsed.get("selectedId") or parsed.get("selectedRowId")
            if selected_id:
                selected_text = parsed.get("title") or parsed.get("text") or parsed.get("selectedText")
                return str(selected_id), selected_text
    except Exception:
        logger.exception("Failed parsing interactive response payload")
    return None, None


def parse_evolution_webhook(payload: dict) -> Optional[IncomingWebhookMessage]:
    event = payload.get("event", "")
    if event != "messages.upsert":
        return None

    data = payload.get("data", {})
    key = data.get("key", {})

    if key.get("fromMe", False):
        return None

    message_obj = data.get("message", {}) or {}
    interactive_id, interactive_text = _extract_interactive_response(message_obj)

    message_type_raw = data.get("messageType", "")
    if not message_type_raw:
        if interactive_id:
            message_type_raw = "conversation"
        elif "conversation" in message_obj or "extendedTextMessage" in message_obj:
            message_type_raw = "conversation"
        elif "audioMessage" in message_obj:
            message_type_raw = "audioMessage"
        elif "pttMessage" in message_obj:
            message_type_raw = "pttMessage"
        elif "imageMessage" in message_obj:
            message_type_raw = "imageMessage"

    content_type = _EVOLUTION_TYPE_MAP.get(message_type_raw)
    if content_type is None and interactive_id:
        # Algunos proveedores reportan messageType interactivo propio
        # (ej. buttonsResponseMessage/listResponseMessage). Lo tratamos como
        # texto estructurado para procesarlo de forma deterministica.
        content_type = MessageType.TEXT
    if content_type is None:
        logger.debug("Unsupported message type: %s", message_type_raw)
        return None

    remote_jid = key.get("remoteJid", "")
    phone = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid
    if not phone:
        return None

    provider_message_id = key.get("id", "") or data.get("message", {}).get("id", "")
    if not provider_message_id:
        return None

    text_body: Optional[str] = None
    media_url: Optional[str] = None
    media_mimetype: Optional[str] = None

    if content_type == MessageType.TEXT:
        text_body = (
            interactive_text
            or message_obj.get("conversation")
            or message_obj.get("extendedTextMessage", {}).get("text", "")
        )
    elif content_type in (MessageType.AUDIO, MessageType.PTT):
        audio_data = message_obj.get("audioMessage") or message_obj.get("pttMessage", {})
        media_url = json.dumps(data)
        media_mimetype = audio_data.get("mimetype", "audio/ogg")
    elif content_type == MessageType.IMAGE:
        image_data = message_obj.get("imageMessage", {})
        media_url = json.dumps(data)
        media_mimetype = image_data.get("mimetype", "image/jpeg")
        text_body = image_data.get("caption")

    return IncomingWebhookMessage(
        provider_message_id=provider_message_id,
        phone=phone,
        content_type=content_type,
        text_body=text_body,
        media_url=media_url,
        media_mimetype=media_mimetype,
        interactive_id=interactive_id,
        interactive_text=interactive_text,
    )
