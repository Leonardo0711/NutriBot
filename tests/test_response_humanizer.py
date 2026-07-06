import json

import httpx
import pytest
from urllib.parse import parse_qs

from application.services.response_humanizer import ResponseHumanizer
from config import get_settings
from infrastructure.twilio_whatsapp.client import TwilioWhatsAppClient


def test_humanizer_variants_delay_and_typing_id(monkeypatch):
    monkeypatch.setenv("HUMANIZE_OUTBOX_ENABLED", "true")
    monkeypatch.setenv("HUMANIZE_TYPING_ENABLED", "true")
    monkeypatch.setenv("HUMANIZE_DELAY_MAX_SECONDS", "2.0")
    get_settings.cache_clear()

    humanizer = ResponseHumanizer()
    prepared = humanizer.prepare(
        text="Entendido, seguimos conversando sin problema.",
        content_type="text",
        idempotency_key="reply:SM1234567890abcdef1234567890abcdef",
        phone="51999999999",
    )

    assert prepared.text in {
        "Entendido 😊 seguimos conversando sin problema.",
        "Claro, seguimos conversando sin problema 😊",
        "De acuerdo, continuamos sin problema.",
        "Perfecto, seguimos normal 😊",
    }
    assert 0 < prepared.delay_seconds <= 2.0
    assert prepared.typing_message_id == "SM1234567890abcdef1234567890abcdef"

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_twilio_typing_indicator_posts_expected_payload(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+51912857367")
    get_settings.cache_clear()

    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.content.decode("utf-8")
        return httpx.Response(200, json={"success": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    twilio = TwilioWhatsAppClient(http_client=client)

    ok = await twilio.send_typing_indicator("SM1234567890abcdef1234567890abcdef")

    assert ok is True
    assert seen["url"] == "https://messaging.twilio.com/v3/Indicators/Typing.json"
    assert '"channel":"WHATSAPP"' in seen["json"]
    assert '"messageId":"SM1234567890abcdef1234567890abcdef"' in seen["json"]

    await twilio.close()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_twilio_download_media_follows_redirect(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+51912857367")
    get_settings.cache_clear()

    api_url = "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM1/Media/ME1"
    cdn_url = "https://mms.twiliocdn.com/AC123/file"

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == api_url:
            return httpx.Response(307, headers={"Location": cdn_url})
        if str(request.url) == cdn_url:
            return httpx.Response(200, content=b"image-bytes")
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    twilio = TwilioWhatsAppClient(http_client=client)

    payload = json.dumps(
        {
            "message": {
                "imageMessage": {
                    "twilioMediaUrl": api_url,
                }
            }
        }
    )

    media = await twilio.download_media(payload)

    assert media == b"image-bytes"

    await twilio.close()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_twilio_sends_audio_by_public_media_url(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+51912857367")
    monkeypatch.delenv("TWILIO_MESSAGING_SERVICE_SID", raising=False)
    get_settings.cache_clear()

    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("utf-8")
        return httpx.Response(201, json={"sid": "SM_AUDIO"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    twilio = TwilioWhatsAppClient(http_client=client)

    result = await twilio.send_audio_url_with_result(
        "51930502319",
        "https://api-nutribot.ietsidis.com/media/outgoing/token123",
    )

    assert result.success is True
    message_payload = parse_qs(seen["body"])
    assert message_payload["To"] == ["whatsapp:+51930502319"]
    assert message_payload["From"] == ["whatsapp:+51912857367"]
    assert message_payload["MediaUrl"] == ["https://api-nutribot.ietsidis.com/media/outgoing/token123"]

    await twilio.close()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_twilio_sends_interactive_buttons_as_quick_reply_content(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+51912857367")
    monkeypatch.delenv("TWILIO_MESSAGING_SERVICE_SID", raising=False)
    get_settings.cache_clear()

    seen = {"content": None, "message": None}

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://content.twilio.com/v1/Content":
            seen["content"] = request
            return httpx.Response(201, json={"sid": "HX123"})
        if str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json":
            seen["message"] = request
            return httpx.Response(201, json={"sid": "SM456"})
        return httpx.Response(404, json={"message": "unexpected url"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    twilio = TwilioWhatsAppClient(http_client=client)

    result = await twilio.send_buttons_with_result(
        "51930502319",
        {
            "body": "¿Quieres completar tu perfil?",
            "sections": [
                {
                    "rows": [
                        {"id": "profile:basic:yes", "title": "Sí"},
                        {"id": "profile:basic:no", "title": "No"},
                    ]
                }
            ],
        },
    )

    assert result.success is True
    assert result.provider_message_id == "SM456"

    content_payload = json.loads(seen["content"].content.decode("utf-8"))
    assert content_payload["types"]["twilio/quick-reply"]["actions"] == [
        {"title": "Sí", "id": "profile:basic:yes"},
        {"title": "No", "id": "profile:basic:no"},
    ]

    message_payload = parse_qs(seen["message"].content.decode("utf-8"))
    assert message_payload["To"] == ["whatsapp:+51930502319"]
    assert message_payload["From"] == ["whatsapp:+51912857367"]
    assert message_payload["ContentSid"] == ["HX123"]

    await twilio.close()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_twilio_sends_interactive_list_as_list_picker_content(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_WHATSAPP_FROM", "whatsapp:+51912857367")
    monkeypatch.delenv("TWILIO_MESSAGING_SERVICE_SID", raising=False)
    get_settings.cache_clear()

    seen = {"content": None, "message": None}

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://content.twilio.com/v1/Content":
            seen["content"] = request
            return httpx.Response(201, json={"sid": "HX789"})
        if str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json":
            seen["message"] = request
            return httpx.Response(201, json={"sid": "SM999"})
        return httpx.Response(404, json={"message": "unexpected url"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    twilio = TwilioWhatsAppClient(http_client=client)

    result = await twilio.send_list_with_result(
        "51930502319",
        {
            "body": "En una escala del 1 al 10, elige una opcion.",
            "buttonText": "Elegir",
            "sections": [
                {
                    "rows": [
                        {"id": f"survey:nps:{i}", "title": str(i), "description": ""}
                        for i in range(1, 11)
                    ]
                }
            ],
        },
    )

    assert result.success is True
    assert result.provider_message_id == "SM999"

    content_payload = json.loads(seen["content"].content.decode("utf-8"))
    list_picker = content_payload["types"]["twilio/list-picker"]
    assert list_picker["button"] == "Elegir"
    assert list_picker["items"][0] == {"item": "1", "id": "survey:nps:1"}
    assert "description" not in list_picker["items"][0]
    assert list_picker["items"][-1]["id"] == "survey:nps:10"
    assert len(list_picker["items"]) == 10

    message_payload = parse_qs(seen["message"].content.decode("utf-8"))
    assert message_payload["To"] == ["whatsapp:+51930502319"]
    assert message_payload["From"] == ["whatsapp:+51912857367"]
    assert message_payload["ContentSid"] == ["HX789"]

    await twilio.close()
    get_settings.cache_clear()
