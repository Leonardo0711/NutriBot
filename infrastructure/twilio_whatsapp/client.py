"""Async client for Twilio WhatsApp messages."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from typing import Any, Optional

import httpx

from config import get_settings
from infrastructure.evolution.client import DeliveryResult

logger = logging.getLogger(__name__)

_TWILIO_BSUID_RE = re.compile(r"^[A-Za-z]{2}\.[A-Za-z0-9]{1,128}$")


class TwilioWhatsAppClient:
    """Twilio Programmable Messaging adapter with the outbox result shape."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None) -> None:
        settings = get_settings()
        self._account_sid = settings.twilio_account_sid.strip()
        self._auth_token = settings.twilio_auth_token.strip()
        self._from = self._normalize_sender(settings.twilio_whatsapp_from)
        self._messaging_service_sid = settings.twilio_messaging_service_sid.strip()
        self._base_url = "https://api.twilio.com/2010-04-01"
        self._content_url = "https://content.twilio.com/v1/Content"
        self._typing_url = "https://messaging.twilio.com/v3/Indicators/Typing.json"
        self._client = http_client
        self._content_sid_cache: dict[str, str] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ready_error(self) -> Optional[str]:
        if not self._account_sid:
            return "twilio_account_sid_missing"
        if not self._auth_token:
            return "twilio_auth_token_missing"
        if not self._from and not self._messaging_service_sid:
            return "twilio_sender_missing"
        return None

    @staticmethod
    def _digits(value: str) -> str:
        return re.sub(r"\D+", "", str(value or ""))

    @classmethod
    def _normalize_to(cls, phone: str) -> str:
        raw = str(phone or "").strip()
        if raw.lower().startswith("whatsapp:"):
            raw = raw.split(":", 1)[1].strip()
        if _TWILIO_BSUID_RE.fullmatch(raw):
            return f"whatsapp:{raw}"
        digits = cls._digits(raw)
        return f"whatsapp:+{digits}" if digits else ""

    @classmethod
    def _normalize_sender(cls, sender: str) -> str:
        raw = str(sender or "").strip()
        if not raw:
            return ""
        if raw.startswith("whatsapp:"):
            return raw
        digits = cls._digits(raw)
        return f"whatsapp:+{digits}" if digits else raw

    async def send_text(self, phone: str, text: str) -> bool:
        result = await self.send_text_with_result(phone, text)
        return result.success

    async def send_text_with_result(
        self, phone: str, text: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        payload: dict[str, Any] = {
            "To": self._normalize_to(phone),
            "Body": str(text or ""),
        }
        if self._messaging_service_sid:
            payload["MessagingServiceSid"] = self._messaging_service_sid
        else:
            payload["From"] = self._from
        if idempotency_key:
            # Twilio no expone idempotency-key para este endpoint; lo guardamos
            # como metadata local en la respuesta registrada.
            payload["_nutribot_idempotency_key"] = idempotency_key
        return await self._post_message(payload)

    async def send_buttons_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        quick_reply = self._quick_reply_content_from_payload(payload)
        if quick_reply:
            content_result = await self._send_quick_reply_content(
                phone,
                quick_reply,
                idempotency_key=idempotency_key,
            )
            if content_result.success:
                return content_result
            logger.warning(
                "Twilio quick-reply failed; falling back to text phone=%s error=%s",
                phone,
                content_result.error,
            )
        return await self.send_text_with_result(
            phone,
            self._interactive_payload_to_text(payload),
            idempotency_key=idempotency_key,
        )

    async def send_list_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        list_picker = self._list_picker_content_from_payload(payload)
        if list_picker:
            content_result = await self._send_list_picker_content(
                phone,
                list_picker,
                idempotency_key=idempotency_key,
            )
            if content_result.success:
                return content_result
            logger.warning(
                "Twilio list-picker failed; falling back to text phone=%s error=%s",
                phone,
                content_result.error,
            )
        return await self.send_text_with_result(
            phone,
            self._interactive_payload_to_text(payload),
            idempotency_key=idempotency_key,
        )

    async def send_audio_base64(self, phone: str, audio_base64: str) -> bool:
        result = await self.send_audio_base64_with_result(phone, audio_base64)
        return result.success

    async def send_audio_base64_with_result(
        self, phone: str, audio_base64: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        return DeliveryResult(
            success=False,
            error="twilio_audio_url_required",
            response_body={"reason": "Twilio WhatsApp requires a public MediaUrl for outbound media."},
            retryable=False,
        )

    async def send_audio_url_with_result(
        self, phone: str, media_url: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        payload: dict[str, Any] = {
            "To": self._normalize_to(phone),
            "MediaUrl": str(media_url or ""),
        }
        if self._messaging_service_sid:
            payload["MessagingServiceSid"] = self._messaging_service_sid
        else:
            payload["From"] = self._from
        if idempotency_key:
            payload["_nutribot_idempotency_key"] = idempotency_key
        return await self._post_message(payload)

    async def send_presence(self, phone: str, presence: str = "composing") -> bool:
        return False

    async def send_typing_indicator(self, message_id: str) -> bool:
        """Show WhatsApp typing indicator for the inbound Twilio Message SID."""
        ready_error = self._ready_error()
        if ready_error or not message_id:
            return False
        if not re.match(r"^(SM|MM)[A-Za-z0-9]{10,}$", str(message_id)):
            return False
        try:
            client = self._get_client()
            resp = await client.post(
                self._typing_url,
                json={"channel": "WHATSAPP", "messageId": str(message_id)},
                auth=(self._account_sid, self._auth_token),
                timeout=5.0,
            )
            if not resp.is_success:
                logger.info(
                    "Twilio typing indicator ignored status=%s body=%s",
                    resp.status_code,
                    resp.text[:300],
                )
            return resp.is_success
        except Exception:
            logger.info("Twilio typing indicator failed", exc_info=True)
            return False

    async def download_media(self, media_ref: str) -> Optional[bytes]:
        url = self._extract_media_url(media_ref)
        if not url or not self._account_sid or not self._auth_token:
            return None
        try:
            client = self._get_client()
            resp = await client.get(
                url,
                auth=(self._account_sid, self._auth_token),
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception:
            logger.exception("Error downloading media from Twilio")
            return None

    async def _post_message(self, payload: dict[str, Any]) -> DeliveryResult:
        ready_error = self._ready_error()
        if ready_error:
            return DeliveryResult(success=False, error=ready_error, retryable=True)

        payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self._base_url}/Accounts/{self._account_sid}/Messages.json",
                data=payload,
                auth=(self._account_sid, self._auth_token),
            )
            body = self._safe_json(resp)
            provider_message_id = body.get("sid") or body.get("message_sid")
            if resp.is_success:
                return DeliveryResult(
                    success=True,
                    status_code=resp.status_code,
                    provider_message_id=provider_message_id,
                    response_body=body,
                    retryable=False,
                )

            retryable = resp.status_code >= 500 or resp.status_code == 429
            err = str(body.get("message") or body.get("code") or f"HTTP {resp.status_code}")
            logger.error("Twilio WhatsApp API failure %s: %s", resp.status_code, err)
            return DeliveryResult(
                success=False,
                status_code=resp.status_code,
                provider_message_id=provider_message_id,
                response_body=body,
                error=err,
                retryable=retryable,
            )
        except Exception:
            logger.exception("Twilio WhatsApp API connection error")
            return DeliveryResult(success=False, error="connection_error", retryable=True)

    async def _send_quick_reply_content(
        self,
        phone: str,
        content: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> DeliveryResult:
        content_sid = await self._get_or_create_quick_reply_content_sid(content)
        if not content_sid:
            return DeliveryResult(success=False, error="twilio_content_sid_missing", retryable=True)

        payload: dict[str, Any] = {
            "To": self._normalize_to(phone),
            "ContentSid": content_sid,
        }
        if self._messaging_service_sid:
            payload["MessagingServiceSid"] = self._messaging_service_sid
        else:
            payload["From"] = self._from
        if idempotency_key:
            payload["_nutribot_idempotency_key"] = idempotency_key
        return await self._post_message(payload)

    async def _send_list_picker_content(
        self,
        phone: str,
        content: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> DeliveryResult:
        content_sid = await self._get_or_create_list_picker_content_sid(content)
        if not content_sid:
            return DeliveryResult(success=False, error="twilio_content_sid_missing", retryable=True)

        payload: dict[str, Any] = {
            "To": self._normalize_to(phone),
            "ContentSid": content_sid,
        }
        if self._messaging_service_sid:
            payload["MessagingServiceSid"] = self._messaging_service_sid
        else:
            payload["From"] = self._from
        if idempotency_key:
            payload["_nutribot_idempotency_key"] = idempotency_key
        return await self._post_message(payload)

    async def _get_or_create_quick_reply_content_sid(self, content: dict[str, Any]) -> Optional[str]:
        cache_key = self._content_cache_key(content)
        if cache_key in self._content_sid_cache:
            return self._content_sid_cache[cache_key]

        body = content["body"]
        actions = content["actions"]
        friendly_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
        payload = {
            "friendly_name": f"nutribot_qr_{friendly_hash}",
            "language": "es",
            "types": {
                "twilio/text": {"body": self._quick_reply_text_fallback(body, actions)},
                "twilio/quick-reply": {
                    "body": body,
                    "actions": actions,
                },
            },
        }
        try:
            client = self._get_client()
            resp = await client.post(
                self._content_url,
                json=payload,
                auth=(self._account_sid, self._auth_token),
                timeout=10.0,
            )
            data = self._safe_json(resp)
            sid = data.get("sid")
            if resp.is_success and sid:
                self._content_sid_cache[cache_key] = sid
                return sid
            logger.error("Twilio content create failed %s: %s", resp.status_code, data)
            return None
        except Exception:
            logger.exception("Twilio content create connection error")
            return None

    async def _get_or_create_list_picker_content_sid(self, content: dict[str, Any]) -> Optional[str]:
        cache_key = self._content_cache_key(content)
        if cache_key in self._content_sid_cache:
            return self._content_sid_cache[cache_key]

        body = content["body"]
        button = content["button"]
        items = content["items"]
        friendly_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
        payload = {
            "friendly_name": f"nutribot_list_{friendly_hash}",
            "language": "es",
            "types": {
                "twilio/text": {"body": self._list_picker_text_fallback(body, items)},
                "twilio/list-picker": {
                    "body": body,
                    "button": button,
                    "items": items,
                },
            },
        }
        try:
            client = self._get_client()
            resp = await client.post(
                self._content_url,
                json=payload,
                auth=(self._account_sid, self._auth_token),
                timeout=10.0,
            )
            data = self._safe_json(resp)
            sid = data.get("sid")
            if resp.is_success and sid:
                self._content_sid_cache[cache_key] = sid
                return sid
            logger.error("Twilio list-picker content create failed %s: %s", resp.status_code, data)
            return None
        except Exception:
            logger.exception("Twilio list-picker content create connection error")
            return None

    @classmethod
    def _quick_reply_content_from_payload(cls, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        body = str(payload.get("body") or payload.get("description") or "").strip()
        actions: list[dict[str, str]] = []

        for button in payload.get("buttons", []) or []:
            if not isinstance(button, dict):
                continue
            title = str(
                button.get("text")
                or button.get("displayText")
                or (button.get("buttonText") or {}).get("displayText")
                or button.get("title")
                or button.get("id")
                or ""
            ).strip()
            action_id = str(button.get("id") or title).strip()
            if title and action_id:
                actions.append({"title": title[:20], "id": action_id[:200]})

        if not actions:
            for section in payload.get("sections", []) or []:
                if not isinstance(section, dict):
                    continue
                for row in section.get("rows", []) or []:
                    if not isinstance(row, dict):
                        continue
                    title = str(row.get("title") or row.get("text") or row.get("id") or "").strip()
                    action_id = str(row.get("id") or title).strip()
                    if title and action_id:
                        actions.append({"title": title[:20], "id": action_id[:200]})

        if not body or not 1 <= len(actions) <= 3:
            return None
        return {"body": body[:1024], "actions": actions[:3]}

    @classmethod
    def _list_picker_content_from_payload(cls, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        body = str(payload.get("body") or payload.get("description") or "").strip()
        button = str(payload.get("buttonText") or "Elegir").strip()[:20] or "Elegir"
        items: list[dict[str, str]] = []

        for section in payload.get("sections", []) or []:
            if not isinstance(section, dict):
                continue
            for row in section.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or row.get("text") or row.get("id") or "").strip()
                action_id = str(row.get("id") or title).strip()
                if title and action_id:
                    item = {"item": title[:24], "id": action_id[:200]}
                    description = str(row.get("description") or "").strip()
                    if description:
                        item["description"] = description[:72]
                    items.append(item)

        if not body or not 1 <= len(items) <= 10:
            return None
        return {"body": body[:1024], "button": button, "items": items[:10]}

    @staticmethod
    def _quick_reply_text_fallback(body: str, actions: list[dict[str, str]]) -> str:
        options = "\n".join(f"- {item.get('title', '')}" for item in actions if item.get("title"))
        return f"{body}\n\n{options}".strip()

    @staticmethod
    def _list_picker_text_fallback(body: str, items: list[dict[str, str]]) -> str:
        options = "\n".join(f"- {item.get('item', '')}" for item in items if item.get("item"))
        return f"{body}\n\n{options}".strip()

    @staticmethod
    def _content_cache_key(content: dict[str, Any]) -> str:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _safe_json(resp: httpx.Response) -> dict[str, Any]:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text[:2000]}

    @staticmethod
    def _interactive_payload_to_text(payload: dict[str, Any]) -> str:
        body = str(payload.get("body") or payload.get("description") or "").strip()
        lines: list[str] = [body] if body else []

        for button in payload.get("buttons", []) or []:
            if not isinstance(button, dict):
                continue
            label = str(
                button.get("text")
                or button.get("displayText")
                or (button.get("buttonText") or {}).get("displayText")
                or button.get("id")
                or ""
            ).strip()
            if label:
                lines.append(f"- {label}")

        for section in payload.get("sections", []) or []:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            if title:
                lines.append(title)
            for row in section.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                row_title = str(row.get("title") or row.get("text") or row.get("id") or "").strip()
                row_desc = str(row.get("description") or "").strip()
                if row_title and row_desc:
                    lines.append(f"- {row_title}: {row_desc}")
                elif row_title:
                    lines.append(f"- {row_title}")

        return "\n".join(lines).strip() or "Por favor, elige una opcion."

    @staticmethod
    def _extract_media_url(media_ref: str) -> Optional[str]:
        try:
            data = json.loads(media_ref)
        except Exception:
            return None
        message = data.get("message") or {}
        for key in ("audioMessage", "pttMessage", "imageMessage"):
            media = message.get(key) or {}
            url = media.get("twilioMediaUrl") or media.get("url")
            if url:
                return str(url)
        return None
