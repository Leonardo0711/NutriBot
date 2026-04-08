"""
Nutribot Backend — TTS Adapter (Text-to-Speech)
Genera audio de voz en memoria como base64 para envío stateless.
"""
from __future__ import annotations

import base64
import io
import logging

from openai import AsyncOpenAI

from config import get_settings
from domain.ports import TTSService

logger = logging.getLogger(__name__)


class OpenAITextToSpeechAdapter(TTSService):
    """Genera audio TTS usando OpenAI y lo retorna como base64 en memoria."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_tts_model
        self._voice = settings.openai_tts_voice

    async def generate_audio_base64(self, text: str) -> str:
        """
        Genera audio desde texto y retorna string base64.
        
        - Formato: opus (óptimo para WhatsApp/comunicación).
        - Sin persistencia local — 100% en memoria.
        - Evolution API acepta audio base64 embebido.
        """
        try:
            response = await self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="opus",
            )

            # The async client returns an HttpxBinaryResponseContent object
            audio_bytes = response.content
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            logger.debug(
                "TTS generado: %d bytes → %d chars base64",
                len(audio_bytes),
                len(audio_b64),
            )
            return audio_b64

        except Exception:
            logger.exception("Error generando TTS")
            raise
