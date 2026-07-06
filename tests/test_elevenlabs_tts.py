import base64

import httpx
import pytest

from infrastructure.elevenlabs.tts_adapter import ElevenLabsTextToSpeechAdapter


class FakeHttpClient:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls = []

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_elevenlabs_tts_returns_base64_and_uses_mp3_format():
    fake_audio = b"fake-mp3"
    fake_client = FakeHttpClient(
        httpx.Response(
            200,
            content=fake_audio,
            request=httpx.Request("POST", "https://example.test"),
        )
    )
    adapter = ElevenLabsTextToSpeechAdapter(
        api_key="test-key",
        voice_id="voice-id",
        model="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        speed=0.92,
        http_client=fake_client,
    )

    result = await adapter.generate_audio_base64("Hola, soy NutriBot.")

    assert result == base64.b64encode(fake_audio).decode("utf-8")
    assert adapter.audio_content_type == "audio/mpeg"
    _, kwargs = fake_client.calls[0]
    assert kwargs["params"]["output_format"] == "mp3_44100_128"
    assert kwargs["json"]["model_id"] == "eleven_flash_v2_5"
    assert kwargs["json"]["voice_settings"]["speed"] == 0.92


def test_elevenlabs_tts_content_type_mapping():
    assert ElevenLabsTextToSpeechAdapter._content_type_for_format("mp3_22050_32") == "audio/mpeg"
    assert ElevenLabsTextToSpeechAdapter._content_type_for_format("opus_48000_32") == "audio/ogg"
    assert ElevenLabsTextToSpeechAdapter._content_type_for_format("pcm_16000") == "application/octet-stream"
