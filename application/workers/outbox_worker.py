"""
Nutribot Backend — OutboxWorker
Worker de outbox: reclama mensajes pendientes y los envía vía Evolution.
"""
from __future__ import annotations

import logging
from sqlalchemy import text

from config import get_settings
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.tts_adapter import OpenAITextToSpeechAdapter

logger = logging.getLogger(__name__)

class OutboxWorker:
    def __init__(self, session_factory, evolution_client: EvolutionApiClient, tts_adapter: OpenAITextToSpeechAdapter):
        self.session_factory = session_factory
        self.evolution_client = evolution_client
        self.tts_adapter = tts_adapter

    async def deliver_pending_messages(self) -> int:
        settings = get_settings()

        async with self.session_factory() as session:
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
                success = await self._deliver_single(msg)
                if success:
                    delivered += 1
            except Exception as e:
                logger.exception("Error entregando mensaje id=%s: %s", msg.id, e)
                await self._mark_failed(msg.id)

        return delivered

    async def _deliver_single(self, msg) -> bool:
        msg_content = msg.content
        send_as_audio = False

        if msg.content_type == "audio_tts":
            try:
                audio_b64 = await self.tts_adapter.generate_audio_base64(msg.content)
            except Exception:
                logger.exception("TTS falló para mensaje id=%s", msg.id)
                await self._mark_failed(msg.id)
                return False

            async with self.session_factory() as session:
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
            async with self.session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text("UPDATE outgoing_messages SET status='sending', updated_at=NOW() WHERE id=:id"),
                        {"id": msg.id},
                    )

        if send_as_audio:
            success = await self.evolution_client.send_audio_base64(msg.phone, msg_content)
        else:
            success = await self.evolution_client.send_text(msg.phone, msg_content)

        if success:
            await self._mark_sent(msg.id)
            return True
        else:
            await self._mark_failed(msg.id)
            return False

    async def _mark_sent(self, msg_id: int) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE outgoing_messages SET status='sent', sent_at=NOW(), updated_at=NOW() WHERE id=:id"),
                    {"id": msg_id},
                )

    async def _mark_failed(self, msg_id: int) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE outgoing_messages SET status='failed', updated_at=NOW() WHERE id=:id"),
                    {"id": msg_id},
                )
