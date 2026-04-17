"""
Nutribot Backend — Media Service
Normaliza mensajes multimedia: descarga, convierte y transcribe.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from domain.entities import IncomingWebhookMessage, NormalizedMessage
from domain.ports import MediaService
from domain.value_objects import MessageType
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.stt_adapter import OpenAISpeechToTextAdapter

logger = logging.getLogger(__name__)


class DefaultMediaService(MediaService):
    """Implementación concreta de MediaService."""

    def __init__(
        self,
        stt_service: Optional[OpenAISpeechToTextAdapter] = None,
        evolution_client: Optional[EvolutionApiClient] = None,
    ) -> None:
        self._stt = stt_service or OpenAISpeechToTextAdapter()
        self._evolution = evolution_client or EvolutionApiClient()

    async def normalize(self, msg: IncomingWebhookMessage) -> NormalizedMessage:
        """
        Convierte un IncomingWebhookMessage en un NormalizedMessage listo para el LLM.

        - TEXT: pasa directo.
        - AUDIO/PTT: descarga ogg desde Evolution, convierte a mp3, envía a STT.
        - IMAGE: descarga imagen, codifica base64 para Vision API.
        """
        text = msg.interactive_id or msg.text_body or ""
        image_base64: Optional[str] = None
        image_mimetype: Optional[str] = None
        used_audio = False

        if msg.content_type == MessageType.TEXT:
            # Texto plano, nada que procesar
            pass

        elif msg.content_type in (MessageType.AUDIO, MessageType.PTT):
            # Nota de voz: descargar + transcribir
            used_audio = True
            if msg.media_url:
                audio_bytes = await self._evolution.download_media(msg.media_url)
                if audio_bytes:
                    transcription = await self._stt.transcribe(
                        audio_bytes, msg.media_mimetype or "audio/ogg"
                    )
                    if transcription:
                        text = transcription
                    else:
                        text = "[Audio no pudo ser transcrito]"
                else:
                    text = "[Audio no pudo ser descargado]"
            else:
                text = "[Audio sin URL de descarga]"

        elif msg.content_type == MessageType.IMAGE:
            # Imagen: descargar + codificar base64 para Vision
            if msg.media_url:
                image_bytes = await self._evolution.download_media(msg.media_url)
                if image_bytes:
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_mimetype = msg.media_mimetype or "image/jpeg"
                    # Si la imagen tiene caption, lo usamos como texto
                    if not text:
                        text = "El usuario envió una imagen."
                else:
                    text = text or "[Imagen no pudo ser descargada]"
            else:
                text = text or "[Imagen sin URL de descarga]"

        return NormalizedMessage(
            provider_message_id=msg.provider_message_id,
            phone=msg.phone,
            content_type=msg.content_type,
            text=text,
            image_base64=image_base64,
            image_mimetype=image_mimetype,
            used_audio=used_audio,
            interactive_id=msg.interactive_id,
        )
