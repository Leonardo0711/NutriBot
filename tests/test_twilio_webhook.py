from interface.twilio_webhook_controller import normalize_twilio_message
from infrastructure.twilio_whatsapp.client import TwilioWhatsAppClient


def test_normalize_twilio_text_message():
    payload = normalize_twilio_message(
        {
            "MessageSid": "SM123",
            "From": "whatsapp:+51999999999",
            "To": "whatsapp:+51912857367",
            "Body": "hola nutribot",
            "NumMedia": "0",
        }
    )

    assert payload is not None
    assert payload["provider"] == "twilio"
    assert payload["event"] == "messages.upsert"
    assert payload["data"]["key"]["id"] == "SM123"
    assert payload["data"]["key"]["remoteJid"] == "51999999999@s.whatsapp.net"
    assert payload["data"]["messageType"] == "conversation"
    assert payload["data"]["message"]["conversation"] == "hola nutribot"


def test_normalize_twilio_image_message():
    payload = normalize_twilio_message(
        {
            "MessageSid": "SM456",
            "WaId": "51988888888",
            "Body": "mi plato",
            "NumMedia": "1",
            "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC/Messages/MM/Media/ME",
            "MediaContentType0": "image/jpeg",
        }
    )

    assert payload is not None
    assert payload["data"]["key"]["remoteJid"] == "51988888888@s.whatsapp.net"
    assert payload["data"]["messageType"] == "imageMessage"
    image = payload["data"]["message"]["imageMessage"]
    assert image["caption"] == "mi plato"
    assert image["mimetype"] == "image/jpeg"
    assert image["twilioMediaUrl"].startswith("https://api.twilio.com/")


def test_normalize_twilio_button_reply():
    payload = normalize_twilio_message(
        {
            "MessageSid": "SM789",
            "WaId": "51977777777",
            "Body": "Sí",
            "ButtonText": "Sí",
            "ButtonPayload": "profile:basic:yes",
            "NumMedia": "0",
        }
    )

    assert payload is not None
    message = payload["data"]["message"]["buttonsResponseMessage"]
    assert message["selectedButtonId"] == "profile:basic:yes"
    assert message["selectedDisplayText"] == "Sí"


def test_normalize_twilio_message_with_private_bsuid():
    payload = normalize_twilio_message(
        {
            "MessageSid": "SM-BSUID",
            "From": "whatsapp:PE.1A2B3C4D5E6F",
            "ExternalUserId": "whatsapp:PE.1A2B3C4D5E6F",
            "Body": "hola desde un usuario privado",
            "NumMedia": "0",
        }
    )

    assert payload is not None
    assert payload["data"]["key"]["remoteJid"] == "PE.1A2B3C4D5E6F@s.whatsapp.net"


def test_twilio_recipient_keeps_private_bsuid():
    assert (
        TwilioWhatsAppClient._normalize_to("PE.1A2B3C4D5E6F")
        == "whatsapp:PE.1A2B3C4D5E6F"
    )


def test_twilio_recipient_keeps_regular_phone_behavior():
    assert TwilioWhatsAppClient._normalize_to("51930502319") == "whatsapp:+51930502319"
