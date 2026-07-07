from application.workers.outbox_worker import OutboxWorker, TWILIO_SAFE_TEXT_LIMIT


def test_split_text_for_whatsapp_keeps_short_text_unchanged():
    text = "Hola, aqui tienes una receta breve."

    assert OutboxWorker._split_text_for_whatsapp(text) == [text]


def test_split_text_for_whatsapp_chunks_long_recipe():
    paragraph = "Ingredientes: " + "pollo tomate espinaca queso integral " * 80
    text = f"{paragraph}\n\nPreparacion: " + "mezcla hornea sirve " * 90

    chunks = OutboxWorker._split_text_for_whatsapp(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= TWILIO_SAFE_TEXT_LIMIT for chunk in chunks)
    assert not any(chunk.startswith("(") for chunk in chunks)
    assert "Ingredientes" in chunks[0]
