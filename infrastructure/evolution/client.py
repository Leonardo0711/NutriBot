"""
Nutribot Backend - Evolution API Client
Async HTTP adapter for WhatsApp sends via Evolution API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from config import get_settings
from domain.ports import EvolutionClient as EvolutionClientPort

logger = logging.getLogger(__name__)


@dataclass
class DeliveryResult:
    success: bool
    status_code: Optional[int] = None
    provider_message_id: Optional[str] = None
    response_body: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    retryable: bool = True


class EvolutionApiClient(EvolutionClientPort):
    """Async HTTP client for Evolution API."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None) -> None:
        settings = get_settings()
        self._base_url = settings.evolution_api_url.rstrip("/")
        self._api_key = settings.evolution_api_key
        self._instance = settings.evolution_instance
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
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "apikey": self._api_key,
        }

    @staticmethod
    def _json_bytes(payload: dict[str, Any]) -> bytes:
        """
        Serialize payload preserving Unicode characters and forcing UTF-8 bytes.
        This avoids mojibake in providers that mishandle default charset inference.
        """
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    async def send_text(self, phone: str, text: str) -> bool:
        result = await self.send_text_with_result(phone, text)
        return result.success

    async def send_text_with_result(
        self, phone: str, text: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        url = f"{self._base_url}/message/sendText/{self._instance}"
        payload = {
            "number": phone,
            "text": text,
        }
        if idempotency_key:
            payload["messageId"] = idempotency_key
            payload["externalId"] = idempotency_key
            payload["clientRef"] = idempotency_key
        return await self._post_detailed(url, payload)

    async def send_audio_base64(self, phone: str, audio_base64: str) -> bool:
        result = await self.send_audio_base64_with_result(phone, audio_base64)
        return result.success

    async def send_audio_base64_with_result(
        self, phone: str, audio_base64: str, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        url = f"{self._base_url}/message/sendWhatsAppAudio/{self._instance}"
        payload = {
            "number": phone,
            "audio": audio_base64,
            "encoding": True,
        }
        if idempotency_key:
            payload["messageId"] = idempotency_key
            payload["externalId"] = idempotency_key
            payload["clientRef"] = idempotency_key
        return await self._post_detailed(url, payload)

    async def send_buttons_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        """
        Send interactive buttons.

        We normalize into a dual-shape payload (canonical + legacy aliases)
        because Evolution variants differ in expected key names.
        """
        url = f"{self._base_url}/message/sendButtons/{self._instance}"
        req_payload = self._normalize_buttons_payload(phone, payload or {})
        if idempotency_key:
            req_payload["messageId"] = idempotency_key
            req_payload["externalId"] = idempotency_key
            req_payload["clientRef"] = idempotency_key
        return await self._post_detailed(url, req_payload)

    async def send_list_with_result(
        self, phone: str, payload: dict, idempotency_key: Optional[str] = None
    ) -> DeliveryResult:
        """
        Send interactive list.

        We normalize into a dual-shape payload (canonical + legacy aliases)
        because Evolution variants differ in expected key names.
        """
        url = f"{self._base_url}/message/sendList/{self._instance}"
        req_payload = self._normalize_list_payload(phone, payload or {})
        if idempotency_key:
            req_payload["messageId"] = idempotency_key
            req_payload["externalId"] = idempotency_key
            req_payload["clientRef"] = idempotency_key
        return await self._post_detailed(url, req_payload)

    @staticmethod
    def _normalize_buttons_payload(phone: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = str(payload.get("body") or payload.get("description") or "").strip()
        title = str(payload.get("title") or "").strip()
        footer = str(payload.get("footer") or "").strip()

        normalized_buttons: list[dict[str, Any]] = []
        for item in payload.get("buttons", []) or []:
            if not isinstance(item, dict):
                continue
            button_id = str(item.get("id") or item.get("buttonId") or "").strip()
            label = str(
                item.get("text")
                or item.get("displayText")
                or (item.get("buttonText") or {}).get("displayText")
                or button_id
            ).strip()
            if not button_id:
                continue
            normalized_buttons.append(
                {
                    "id": button_id,
                    "text": label,
                    "type": str(item.get("type") or "reply"),
                    "buttonId": button_id,
                    "buttonText": {"displayText": label},
                }
            )

        return {
            "number": phone,
            "type": str(payload.get("type") or "buttons"),
            "body": body,
            "title": title,
            "description": str(payload.get("description") or body),
            "footer": footer,
            "buttons": normalized_buttons,
        }

    @staticmethod
    def _normalize_list_payload(phone: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = str(payload.get("body") or payload.get("description") or "").strip()
        title = str(payload.get("title") or "").strip()
        footer = str(payload.get("footer") or "").strip()
        button_text = str(payload.get("buttonText") or "Elegir").strip()

        normalized_sections: list[dict[str, Any]] = []
        for section in payload.get("sections", []) or []:
            if not isinstance(section, dict):
                continue
            section_title = str(section.get("title") or "").strip()
            rows_out: list[dict[str, Any]] = []
            for row in section.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                row_id = str(row.get("id") or row.get("rowId") or "").strip()
                row_title = str(row.get("title") or row.get("text") or row_id).strip()
                row_description = str(row.get("description") or "").strip()
                if not row_id:
                    continue
                rows_out.append(
                    {
                        "id": row_id,
                        "title": row_title,
                        "description": row_description,
                        "rowId": row_id,
                    }
                )
            if rows_out:
                normalized_sections.append({"title": section_title, "rows": rows_out})

        out = {
            "number": phone,
            "type": str(payload.get("type") or "list"),
            "body": body,
            "sections": normalized_sections,
        }
        if title:
            out["title"] = title
        if str(payload.get("description") or body).strip():
            out["description"] = str(payload.get("description") or body).strip()
        if footer:
            out["footer"] = footer
        if button_text:
            out["buttonText"] = button_text
        return out

    async def download_media(self, media_ref: str) -> Optional[bytes]:
        """Download and decrypt media bytes using Evolution API."""
        import base64
        import json

        try:
            message_obj = json.loads(media_ref)
            url = f"{self._base_url}/chat/getBase64FromMediaMessage/{self._instance}"
            payload = {"message": message_obj}

            client = self._get_client()
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

            b64_str = data.get("base64", "")
            if not b64_str:
                logger.error("Evolution returned empty media base64")
                return None

            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            return base64.b64decode(b64_str)
        except Exception:
            logger.exception("Error downloading media from Evolution /getBase64")
            return None

    async def send_presence(self, phone: str, presence: str = "composing") -> bool:
        """Send typing presence with short timeout."""
        url = f"{self._base_url}/chat/sendPresence/{self._instance}"
        payload = {"number": phone, "presence": presence}
        try:
            client = self._get_client()
            resp = await client.post(url, json=payload, headers=self._headers(), timeout=5.0)
            return resp.is_success
        except Exception:
            return False

    async def _post(self, url: str, payload: dict) -> bool:
        result = await self._post_detailed(url, payload)
        return result.success

    async def _post_detailed(self, url: str, payload: dict) -> DeliveryResult:
        try:
            client = self._get_client()
            resp = await client.post(url, content=self._json_bytes(payload), headers=self._headers())
            body = None
            provider_message_id = None
            try:
                body = resp.json()
                provider_message_id = (
                    body.get("key", {}).get("id")
                    or body.get("id")
                    or body.get("messageId")
                )
            except Exception:
                body = {"raw": resp.text[:2000]}

            if resp.is_success:
                logger.debug("Evolution response: %s", resp.status_code)
                return DeliveryResult(
                    success=True,
                    status_code=resp.status_code,
                    provider_message_id=provider_message_id,
                    response_body=body,
                    retryable=False,
                )

            logger.error("Evolution API failure %s: %s", resp.status_code, resp.text)
            retryable = resp.status_code >= 500 or resp.status_code == 429
            return DeliveryResult(
                success=False,
                status_code=resp.status_code,
                provider_message_id=provider_message_id,
                response_body=body,
                error=f"HTTP {resp.status_code}",
                retryable=retryable,
            )
        except httpx.HTTPStatusError as e:
            logger.error("Evolution API HTTP error %s: %s", e.response.status_code, e.response.text)
            retryable = e.response.status_code >= 500 or e.response.status_code == 429
            return DeliveryResult(
                success=False,
                status_code=e.response.status_code,
                response_body={"raw": e.response.text[:2000]},
                error=f"HTTP {e.response.status_code}",
                retryable=retryable,
            )
        except Exception:
            logger.exception("Evolution API connection error: %s", url)
            return DeliveryResult(
                success=False,
                error="connection_error",
                retryable=True,
            )
