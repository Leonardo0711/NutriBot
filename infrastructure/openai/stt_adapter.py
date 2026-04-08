"""
Nutribot Backend — STT Adapter (Speech-to-Text)
Convierte audio ogg/opus de WhatsApp a texto usando OpenAI Whisper/Transcribe.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from pydub import AudioSegment
from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


class OpenAISpeechToTextAdapter:
    """Transcribe audio usando la API de OpenAI Audio Transcriptions."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_stt_model

    async def transcribe(self, audio_bytes: bytes, mimetype: str = "audio/ogg") -> Optional[str]:
        """
        Transcribe audio a texto.
        
        1. Recibe bytes crudos (ogg/opus de WhatsApp).
        2. Convierte a mp3 con pydub/ffmpeg (formato soportado por OpenAI).
        3. Envía a la API de transcripción.
        4. Retorna el texto transcrito.

        OpenAI STT soporta: mp3, mp4, mpeg, mpga, m4a, wav, webm (hasta 25 MB).
        ogg/opus NO está soportado, por eso la conversión es obligatoria.
        """
        try:
            # Conversión ogg/opus → mp3 en memoria (sin tocar disco)
            mp3_buffer = self._convert_to_mp3(audio_bytes, mimetype)
            if mp3_buffer is None:
                return None

            # Enviar a OpenAI
            mp3_buffer.name = "audio.mp3"  # OpenAI necesita un nombre con extensión
            transcript = await self._client.audio.transcriptions.create(
                model=self._model,
                file=mp3_buffer,
                language="es",
            )

            text = transcript.text.strip()
            logger.debug("STT transcripción (%d chars): %s...", len(text), text[:80])
            return text

        except Exception:
            logger.exception("Error en transcripción STT")
            return None

    @staticmethod
    def _convert_to_mp3(audio_bytes: bytes, mimetype: str) -> Optional[io.BytesIO]:
        """Convierte audio crudo a mp3 en memoria usando pydub + ffmpeg."""
        try:
            # Determinar formato de entrada
            if "opus" in mimetype or "ogg" in mimetype:
                fmt = "ogg"
            elif "webm" in mimetype:
                fmt = "webm"
            elif "mp4" in mimetype or "m4a" in mimetype:
                fmt = "mp4"
            else:
                fmt = "ogg"  # Default para WhatsApp

            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)

            # Exportar a mp3 en memoria
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3", bitrate="64k")
            mp3_buffer.seek(0)
            return mp3_buffer

        except Exception:
            logger.exception("Error convirtiendo audio a mp3 (mimetype=%s)", mimetype)
            return None
