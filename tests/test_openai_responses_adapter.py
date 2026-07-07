from infrastructure.openai.responses_adapter import OpenAIResponsesAdapter


def test_short_contextual_followup_keeps_history_even_with_previous_response():
    history = [
        {
            "role": "assistant",
            "content": "¿Te gustaría saber cómo acompañarla con ensaladas o algún otro platillo?",
        }
    ]

    assert OpenAIResponsesAdapter._should_include_history_with_previous("Dale", history)


def test_non_contextual_short_reply_does_not_duplicate_history():
    history = [{"role": "assistant", "content": "Perfecto. Seguimos cuando quieras 😊"}]

    assert not OpenAIResponsesAdapter._should_include_history_with_previous("ok", history)
