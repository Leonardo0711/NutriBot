"""
Nutribot Backend — DeliverReplyUseCase (Outbox Worker)
Reclama mensajes salientes, genera TTS si corresponde, y envía vía Evolution.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.tts_adapter import OpenAITextToSpeechAdapter

logger = logging.getLogger(__name__)


async def deliver_pending_messages() -> int:
    """
    Worker de outbox: reclama mensajes pendientes y los envía.
    Retorna la cantidad de mensajes entregados.
    """
    settings = get_settings()
    factory = get_session_factory()
    evolution = EvolutionApiClient()
    tts = OpenAITextToSpeechAdapter()

    # ─── Transacción Corta: Reclamar mensajes ───
    async with factory() as session:
        async with session.begin():
            result = await session.execute(
                text("""
                    UPDATE outgoing_messages
                    SET status = 'processing',
                        locked_at = NOW(),
                        updated_at = NOW(),
                        attempt_count = attempt_count + 1
                    WHERE id IN (
                        SELECT id FROM outgoing_messages
                        WHERE status IN ('pending', 'failed')
                          AND attempt_count < :max_retry
                        ORDER BY created_at ASC
                        LIMIT 10
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                """),
                {"max_retry": settings.max_retry_count},
            )
            messages = result.fetchall()

    if not messages:
        return 0

    delivered = 0
    for msg in messages:
        try:
            success = await _deliver_single(msg, factory, evolution, tts)
            if success:
                delivered += 1
        except Exception as e:
            logger.exception("Error entregando mensaje id=%s: %s", msg.id, e)
            await _mark_failed(factory, msg.id)

    return delivered


async def _deliver_single(msg, factory, evolution: EvolutionApiClient, tts: OpenAITextToSpeechAdapter) -> bool:
    """Entrega un único mensaje, generando TTS si corresponde."""

    msg_content = msg.content
    send_as_audio = False

    # ─── TTS On-the-Fly (si es audio_tts) ───
    if msg.content_type == "audio_tts":
        try:
            audio_b64 = await tts.generate_audio_base64(msg.content)
        except Exception:
            logger.exception("TTS falló para mensaje id=%s", msg.id)
            await _mark_failed(factory, msg.id)
            return False

        # Actualizar: audio generado → marcar como 'sending' con el base64
        async with factory() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        UPDATE outgoing_messages
                        SET content_type = 'audio',
                            content = :b64,
                            status = 'sending',
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {"b64": audio_b64, "id": msg.id},
                )
        msg_content = audio_b64
        send_as_audio = True
    else:
        # Texto plano → marcar como 'sending'
        async with factory() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE outgoing_messages SET status='sending', updated_at=NOW() WHERE id=:id"),
                    {"id": msg.id},
                )

    # ─── Envío a Evolution ───
    if send_as_audio:
        success = await evolution.send_audio_base64(msg.phone, msg_content)
    else:
        success = await evolution.send_text(msg.phone, msg_content)

    if success:
        await _mark_sent(factory, msg.id)
        return True
    else:
        await _mark_failed(factory, msg.id)
        return False


async def _mark_sent(factory, msg_id: int) -> None:
    async with factory() as session:
        async with session.begin():
            await session.execute(
                text("UPDATE outgoing_messages SET status='sent', sent_at=NOW(), updated_at=NOW() WHERE id=:id"),
                {"id": msg_id},
            )


async def _mark_failed(factory, msg_id: int) -> None:
    async with factory() as session:
        async with session.begin():
            await session.execute(
                text("UPDATE outgoing_messages SET status='failed', updated_at=NOW() WHERE id=:id"),
                {"id": msg_id},
            )
