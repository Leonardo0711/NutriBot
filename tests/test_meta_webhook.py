from interface.meta_webhook_controller import normalize_meta_messages


def test_normalize_meta_text_message():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "51999999999",
                                    "id": "wamid.123",
                                    "type": "text",
                                    "text": {"body": "hola nutribot"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    messages = normalize_meta_messages(payload)

    assert len(messages) == 1
    assert messages[0]["event"] == "messages.upsert"
    assert messages[0]["data"]["key"]["id"] == "wamid.123"
    assert messages[0]["data"]["key"]["remoteJid"] == "51999999999@s.whatsapp.net"
    assert messages[0]["data"]["messageType"] == "conversation"
    assert messages[0]["data"]["message"]["conversation"] == "hola nutribot"


def test_normalize_meta_button_reply():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "51999999999",
                                    "id": "wamid.456",
                                    "type": "interactive",
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {
                                            "id": "continue",
                                            "title": "Continuar",
                                        },
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    messages = normalize_meta_messages(payload)

    reply = messages[0]["data"]["message"]["buttonsResponseMessage"]
    assert reply["selectedButtonId"] == "continue"
    assert reply["selectedDisplayText"] == "Continuar"
