"""Async client for Meta WhatsApp Cloud API."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Optional

import httpx

from config import get_settings
from infrastructure.evolution.client import DeliveryResult

logger = logging.getLogger(__name__)


class MetaWhatsAppClient:
    """WhatsApp Cloud API adapter with the same result shape used by outbox."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None) -> None:
        settings = get_settings()
        self._version = settings.meta_graph_api_version.strip() or "v25.0"
        self._token = settings.meta_whatsapp_token.strip()
        self._phone_number_id = settings.meta_phone_number_id.strip()
        self._base_url = f"https://graph.facebook.com/{self._version}"
        self._client = http_client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def normalize_phone(phone: str) -> str:
        """Return WhatsApp Cloud API recipient format: digits only, no plus sign."""
        return re.sub(r"\D+", "", str(phone or ""))

    def _messages_url(self) -> str:
        return f"{self._base_url}/{self._phone_number_id}/messages"

    def _ready_error(self) -> Optional[str]:
        if not self._token:
            return "meta_whatsapp_token_missing"
        if not self._phone_number_id:
            return "meta_phone_number_id_missing"
        return None

    async def send_text(self, to: str, text: str) -> dict[str, Any]:
        result = await self.send_text_with_result(to, text)
        if not result.success:
            raise RuntimeError(result.error or "meta_send_text_failed")
        return result.response_body or {}

    async def send_text_with_result(
        self, phone: str, text: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": self.normalize_phone(phone),
            "type": "text",
            "text": {
                "preview_url": False,
                "body": str(text or ""),
            },
        }
        return await self._post_message(payload)

    async def send_template(
        self,
        to: str,
        template_name: str,
        language_code: str = "es",
        components: list | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": self.normalize_phone(to),
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }
        if components:
            payload["template"]["components"] = components
        result = await self._post_message(payload)
        if not result.success:
            raise RuntimeError(result.error or "meta_send_template_failed")
        return result.response_body or {}

    async def mark_as_read(self, message_id: str) -> dict[str, Any]:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        result = await self._post_message(payload)
        if not result.success:
            raise RuntimeError(result.error or "meta_mark_as_read_failed")
        return result.response_body or {}

    async def send_buttons_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        text = self._interactive_payload_to_text(payload)
        return await self.send_text_with_result(phone, text, idempotency_key=idempotency_key)

    async def send_list_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        text = self._interactive_payload_to_text(payload)
        return await self.send_text_with_result(phone, text, idempotency_key=idempotency_key)

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

    async def send_audio_base64_with_result(
        self, phone: str, audio_base64: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        ready_error = self._ready_error()
        if ready_error:
            return DeliveryResult(success=False, error=ready_error, retryable=True)

        try:
            audio_bytes = base64.b64decode(audio_base64)
            media_id = await self._upload_media(audio_bytes, "audio/ogg", "nutribot.opus")
            if not media_id:
                return DeliveryResult(success=False, error="meta_media_upload_failed", retryable=True)
            payload = {
                "messaging_product": "whatsapp",
                "to": self.normalize_phone(phone),
                "type": "audio",
                "audio": {"id": media_id},
            }
            return await self._post_message(payload)
        except Exception:
            logger.exception("Meta WhatsApp audio send failed")
            return DeliveryResult(success=False, error="meta_audio_send_failed", retryable=True)

    async def download_media(self, media_ref: str) -> Optional[bytes]:
        media_id = self._extract_media_id(media_ref)
        if not media_id:
            logger.error("Meta media id not found in incoming payload")
            return None
        media_url = await self.get_media_url(media_id)
        if not media_url:
            return None
        try:
            client = self._get_client()
            resp = await client.get(media_url, headers={"Authorization": f"Bearer {self._token}"})
            resp.raise_for_status()
            return resp.content
        except Exception:
            logger.exception("Error downloading media from Meta WhatsApp")
            return None

    async def get_media_url(self, media_id: str) -> Optional[str]:
        ready_error = self._ready_error()
        if ready_error:
            logger.error("Meta media URL unavailable: %s", ready_error)
            return None
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self._base_url}/{media_id}",
                headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
            )
            resp.raise_for_status()
            body = resp.json()
            return body.get("url")
        except Exception:
            logger.exception("Error requesting Meta media URL")
            return None

    async def _upload_media(self, media_bytes: bytes, mime_type: str, filename: str) -> Optional[str]:
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/{self._phone_number_id}/media",
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
            data={"messaging_product": "whatsapp", "type": mime_type},
            files={"file": (filename, media_bytes, mime_type)},
        )
        body = self._safe_json(resp)
        if resp.is_success:
            return body.get("id")
        logger.error("Meta media upload failure %s: %s", resp.status_code, body)
        return None

    async def _post_message(self, payload: dict[str, Any]) -> DeliveryResult:
        ready_error = self._ready_error()
        if ready_error:
            return DeliveryResult(success=False, error=ready_error, retryable=True)

        try:
            client = self._get_client()
            resp = await client.post(self._messages_url(), json=payload, headers=self._headers())
            body = self._safe_json(resp)
            provider_message_id = self._extract_provider_message_id(body)
            if resp.is_success:
                return DeliveryResult(
                    success=True,
                    status_code=resp.status_code,
                    provider_message_id=provider_message_id,
                    response_body=body,
                    retryable=False,
                )

            retryable = resp.status_code >= 500 or resp.status_code == 429
            err = self._extract_error(body) or f"HTTP {resp.status_code}"
            logger.error("Meta WhatsApp API failure %s: %s", resp.status_code, err)
            return DeliveryResult(
                success=False,
                status_code=resp.status_code,
                provider_message_id=provider_message_id,
                response_body=body,
                error=err,
                retryable=retryable,
            )
        except Exception:
            logger.exception("Meta WhatsApp API connection error")
            return DeliveryResult(success=False, error="connection_error", retryable=True)

    @staticmethod
    def _safe_json(resp: httpx.Response) -> dict[str, Any]:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text[:2000]}

    @staticmethod
    def _extract_provider_message_id(body: dict[str, Any]) -> Optional[str]:
        messages = body.get("messages") or []
        if messages and isinstance(messages[0], dict):
            return messages[0].get("id")
        return body.get("id")

    @staticmethod
    def _extract_error(body: dict[str, Any]) -> Optional[str]:
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)
        return str(err) if err else None

    @staticmethod
    def _extract_media_id(media_ref: str) -> Optional[str]:
        try:
            data = json.loads(media_ref)
        except Exception:
            return None
        message = data.get("message") or {}
        for key in ("audioMessage", "pttMessage", "imageMessage"):
            media = message.get(key) or {}
            if media.get("metaMediaId"):
                return str(media["metaMediaId"])
        original = ((data.get("meta") or {}).get("originalMessage") or {})
        message_type = original.get("type")
        if message_type and isinstance(original.get(message_type), dict):
            media_id = original[message_type].get("id")
            if media_id:
                return str(media_id)
        return None
