"""
Nutribot Backend — Evolution API Client
Adaptador HTTP para enviar mensajes a WhatsApp vía Evolution API.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from config import get_settings
from domain.ports import EvolutionClient as EvolutionClientPort

logger = logging.getLogger(__name__)


class EvolutionApiClient(EvolutionClientPort):
    """Cliente HTTP async para la Evolution API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.evolution_api_url.rstrip("/")
        self._api_key = settings.evolution_api_key
        self._instance = settings.evolution_instance

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "apikey": self._api_key,
        }

    async def send_text(self, phone: str, text: str) -> bool:
        """Envía un mensaje de texto plano."""
        url = f"{self._base_url}/message/sendText/{self._instance}"
        payload = {
            "number": phone,
            "text": text,
        }
        return await self._post(url, payload)

    async def send_audio_base64(self, phone: str, audio_base64: str) -> bool:
        """Envía audio como base64 embebido en el payload.
        Evolution API acepta audio base64 vía /message/sendWhatsAppAudio."""
        url = f"{self._base_url}/message/sendWhatsAppAudio/{self._instance}"
        payload = {
            "number": phone,
            "audio": audio_base64,
            "encoding": True,
        }
        return await self._post(url, payload)

    async def download_media(self, media_url: str) -> Optional[bytes]:
        """Descarga y descifra un archivo de media usando Evolution API."""
        import json
        import base64
        try:
            message_obj = json.loads(media_url)
            url = f"{self._base_url}/chat/getBase64FromMediaMessage/{self._instance}"
            payload = {"message": message_obj}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                
                b64_str = data.get("base64", "")
                if not b64_str:
                    logger.error("Evolution no devolvió base64 para la media.")
                    return None
                    
                # Quitar el prefijo data:...;base64, si lo tiene
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                    
                return base64.b64decode(b64_str)
        except Exception:
            logger.exception("Error descargando media desde Evolution /getBase64")
            return None

    async def send_presence(self, phone: str, presence: str = "composing") -> bool:
        """Envía el estado de presencia con timeout ultra corto (5s)."""
        url = f"{self._base_url}/chat/sendPresence/{self._instance}"
        payload = {"number": phone, "presence": presence}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                return resp.is_success
        except Exception:
            return False

    async def _post(self, url: str, payload: dict) -> bool:
        """POST genérico con manejo de errores."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                logger.debug("Evolution response: %s", resp.status_code)
                return True
        except httpx.HTTPStatusError as e:
            logger.error(
                "Evolution API HTTP error %s: %s", e.response.status_code, e.response.text
            )
            return False
        except Exception:
            logger.exception("Evolution API connection error: %s", url)
            return False
