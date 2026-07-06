"""ElevenLabs Text-to-Speech adapter."""
from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from config import get_settings
from domain.ports import TTSService

logger = logging.getLogger(__name__)


class ElevenLabsTextToSpeechAdapter(TTSService):
    """Generate TTS audio with ElevenLabs and return it as base64."""

    _BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model: str | None = None,
        output_format: str | None = None,
        speed: float | None = None,
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        use_speaker_boost: bool | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.elevenlabs_tts_api_key
        self._voice_id = voice_id or settings.elevenlabs_tts_voice_id
        self._model = model or settings.elevenlabs_tts_model
        self._output_format = output_format or settings.elevenlabs_tts_output_format
        self._speed = settings.elevenlabs_tts_speed if speed is None else speed
        self._stability = settings.elevenlabs_tts_stability if stability is None else stability
        self._similarity_boost = (
            settings.elevenlabs_tts_similarity_boost
            if similarity_boost is None
            else similarity_boost
        )
        self._style = settings.elevenlabs_tts_style if style is None else style
        self._use_speaker_boost = (
            settings.elevenlabs_tts_use_speaker_boost
            if use_speaker_boost is None
            else use_speaker_boost
        )
        self._http_client = http_client

        if not self._api_key:
            raise ValueError("ELEVENLABS_TTS_API_KEY is required when TTS_PROVIDER=elevenlabs")
        if not self._voice_id:
            raise ValueError("ELEVENLABS_TTS_VOICE_ID is required when TTS_PROVIDER=elevenlabs")

    @property
    def audio_content_type(self) -> str:
        return self._content_type_for_format(self._output_format)

    async def generate_audio_base64(self, text: str) -> str:
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": self._stability,
                "similarity_boost": self._similarity_boost,
                "style": self._style,
                "speed": self._speed,
                "use_speaker_boost": self._use_speaker_boost,
            },
        }
        url = f"{self._BASE_URL}/{self._voice_id}/stream"
        params = {
            "output_format": self._output_format,
            "optimize_streaming_latency": "3",
        }
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        url,
                        params=params,
                        headers=headers,
                        json=payload,
                    )
            response.raise_for_status()
            audio_bytes = response.content
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            logger.debug(
                "ElevenLabs TTS generated: %d bytes -> %d chars base64 (%s)",
                len(audio_bytes),
                len(audio_b64),
                self.audio_content_type,
            )
            return audio_b64
        except Exception:
            logger.exception("Error generating ElevenLabs TTS")
            raise

    @staticmethod
    def _content_type_for_format(output_format: str) -> str:
        fmt = (output_format or "").lower()
        if fmt.startswith("mp3"):
            return "audio/mpeg"
        if fmt.startswith("opus") or fmt.startswith("ogg"):
            return "audio/ogg"
        if fmt.startswith("ulaw"):
            return "audio/basic"
        return "application/octet-stream"
