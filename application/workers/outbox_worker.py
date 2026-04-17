"""
Nutribot Backend — OutboxWorker
Worker de outbox: consume IDs desde Redis (o SQL fallback)
y envía mensajes vía Evolution.
"""
from __future__ import annotations

import json
import logging
from sqlalchemy import bindparam, text
from sqlalchemy import Integer
from sqlalchemy.dialects.postgresql import JSONB

from config import get_settings
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.tts_adapter import OpenAITextToSpeechAdapter
from infrastructure.redis.client import dequeue, OUTBOX_QUEUE

logger = logging.getLogger(__name__)

class OutboxWorker:
    def __init__(self, session_factory, evolution_client: EvolutionApiClient, tts_adapter: OpenAITextToSpeechAdapter):
        self.session_factory = session_factory
        self.evolution_client = evolution_client
        self.tts_adapter = tts_adapter

    async def deliver_pending_messages(self) -> int:
        """Consume mensajes: primero de Redis, luego fallback SQL."""
        delivered = 0

        # --- Estrategia 1: Consumir de Redis ---
        try:
            msg_id = await dequeue(OUTBOX_QUEUE, timeout=0.5)
        except Exception:
            msg_id = None

        if msg_id:
            msg = await self._load_and_lock_by_id(int(msg_id))
            if msg:
                try:
                    success = await self._deliver_single(msg)
                    if success:
                        delivered += 1
                except Exception as e:
                    logger.exception("Error entregando mensaje id=%s: %s", msg.id, e)
                    await self._mark_failed(msg.id, error_detail=str(e))
            return delivered

        # --- Estrategia 2: Fallback SQL ---
        messages = await self._claim_from_sql()
        if not messages:
            return 0

        for msg in messages:
            try:
                success = await self._deliver_single(msg)
                if success:
                    delivered += 1
            except Exception as e:
                logger.exception("Error entregando mensaje id=%s: %s", msg.id, e)
                await self._mark_failed(msg.id, error_detail=str(e))

        return delivered

    async def _load_and_lock_by_id(self, msg_id: int):
        """Carga un mensaje por ID y lo marca como processing."""
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text("""
                        UPDATE outgoing_messages
                        SET status = 'processing',
                            locked_at = NOW(),
                            updated_at = NOW(),
                            attempt_count = attempt_count + 1
                        WHERE id = :id
                          AND status = 'pending'
                          AND scheduled_at <= NOW()
                        RETURNING *
                    """),
                    {"id": msg_id},
                )
                return result.fetchone()

    async def _claim_from_sql(self):
        """Fallback: reclama mensajes pendientes via SQL polling."""
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
                              AND scheduled_at <= NOW()
                            ORDER BY scheduled_at ASC, created_at ASC
                            LIMIT 10
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING *
                    """),
                    {"max_retry": settings.max_retry_count},
                )
                return result.fetchall()

    async def _deliver_single(self, msg) -> bool:
        msg_content = msg.content
        send_as_audio = False
        interactive_payload = getattr(msg, "payload_json", None) or {}
        if isinstance(interactive_payload, str):
            try:
                interactive_payload = json.loads(interactive_payload)
            except Exception:
                interactive_payload = {}
        ctype = msg.content_type

        if ctype == "audio_tts":
            try:
                # No mutamos content_type/content: conservar intencion logica
                # evita romper los retries si el envio falla.
                msg_content = await self.tts_adapter.generate_audio_base64(msg.content)
                send_as_audio = True
            except Exception:
                logger.exception("TTS fallo para mensaje id=%s", msg.id)
                await self._mark_failed(msg.id, error_detail="tts_failed")
                return False
        elif ctype == "audio":
            # Compatibilidad con registros legacy cuyo payload ya es base64 de audio.
            send_as_audio = True

        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        UPDATE outgoing_messages
                        SET status='sending',
                            last_attempt_at=NOW(),
                            error_detail=NULL,
                            updated_at=NOW()
                        WHERE id=:id
                        """
                    ),
                    {"id": msg.id},
                )

        if ctype == "interactive_buttons":
            result = await self.evolution_client.send_buttons_with_result(
                msg.phone,
                interactive_payload,
                idempotency_key=msg.idempotency_key,
            )
        elif ctype == "interactive_list":
            result = await self.evolution_client.send_list_with_result(
                msg.phone,
                interactive_payload,
                idempotency_key=msg.idempotency_key,
            )
        elif send_as_audio:
            result = await self.evolution_client.send_audio_base64_with_result(
                msg.phone,
                msg_content,
                idempotency_key=msg.idempotency_key,
            )
        else:
            result = await self.evolution_client.send_text_with_result(
                msg.phone,
                msg_content,
                idempotency_key=msg.idempotency_key,
            )

        if result.success:
            await self._mark_sent(
                msg.id,
                provider_delivery_id=result.provider_message_id,
                provider_response=result.response_body,
            )
            return True

        # Fallback: if interactive list/button fails, try sending as text to ensure user gets a response
        if not result.success and ctype in {"interactive_buttons", "interactive_list"}:
            logger.warning("Interactive message failed with 400. Falling back to plain text for phone %s", msg.phone)
            result = await self.evolution_client.send_text_with_result(
                msg.phone,
                msg_content,
                idempotency_key=f"fb_{msg.idempotency_key or msg.id}",
            )
            if result.success:
                await self._mark_sent(
                    msg.id,
                    provider_delivery_id=result.provider_message_id,
                    provider_response=result.response_body,
                )
                return True

        # Final failure handling
        await self._mark_failed(
            msg.id,
            error_detail=result.error or f"provider_http_{result.status_code or 'unknown'}",
            provider_response=result.response_body,
            non_retryable=not result.retryable,
        )
        return False

    async def _mark_sent(
        self,
        msg_id: int,
        provider_delivery_id: str | None = None,
        provider_response: dict | None = None,
    ) -> None:
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    mark_sent_stmt = text(
                        """
                        UPDATE outgoing_messages
                        SET status='sent',
                            sent_at=NOW(),
                            provider_delivery_id=:provider_delivery_id,
                            provider_response=:provider_response,
                            updated_at=NOW()
                        WHERE id=:id
                        """
                    ).bindparams(bindparam("provider_response", type_=JSONB))
                    await session.execute(
                        mark_sent_stmt,
                        {
                            "id": msg_id,
                            "provider_delivery_id": provider_delivery_id,
                            "provider_response": provider_response or {},
                        },
                    )
            return
        except Exception:
            logger.exception(
                "outbox _mark_sent detallado falló para id=%s, aplicando fallback",
                msg_id,
            )

        # Compatibilidad temporal en entornos sin columnas nuevas.
        async with self.session_factory() as fallback_session:
            async with fallback_session.begin():
                await fallback_session.execute(
                    text("UPDATE outgoing_messages SET status='sent', sent_at=NOW(), updated_at=NOW() WHERE id=:id"),
                    {"id": msg_id},
                )

    async def _mark_failed(
        self,
        msg_id: int,
        error_detail: str | None = None,
        provider_response: dict | None = None,
        non_retryable: bool = False,
    ) -> None:
        settings = get_settings()
        capped_attempt = settings.max_retry_count if non_retryable else None

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    mark_failed_stmt = text(
                        """
                        UPDATE outgoing_messages
                        SET status='failed',
                            error_detail=:err,
                            provider_response=:provider_response,
                            attempt_count=CASE
                                WHEN :cap_attempt IS NULL THEN attempt_count
                                ELSE GREATEST(attempt_count, :cap_attempt)
                            END,
                            updated_at=NOW()
                        WHERE id=:id
                        """
                    ).bindparams(
                        bindparam("provider_response", type_=JSONB),
                        bindparam("cap_attempt", type_=Integer),
                    )
                    await session.execute(
                        mark_failed_stmt,
                        {
                            "id": msg_id,
                            "err": (error_detail or "send_failed")[:1000],
                            "provider_response": provider_response or {},
                            "cap_attempt": capped_attempt,
                        },
                    )
            return
        except Exception:
            logger.exception(
                "outbox _mark_failed detallado falló para id=%s, aplicando fallback",
                msg_id,
            )

        # Compatibilidad temporal en entornos sin columnas nuevas.
        async with self.session_factory() as fallback_session:
            async with fallback_session.begin():
                await fallback_session.execute(
                    text("UPDATE outgoing_messages SET status='failed', updated_at=NOW() WHERE id=:id"),
                    {"id": msg_id},
                )

