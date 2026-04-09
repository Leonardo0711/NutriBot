"""
Nutribot Backend — InboxWorker
Consumidor del Inbox: reclama webhooks pendientes y delega
al MessageOrchestratorService.
"""
from __future__ import annotations

import asyncio
import logging
from sqlalchemy import text

from domain.value_objects import MessageType
from config import get_settings
from infrastructure.db.connection import get_session_factory
from application.services.message_orchestrator import MessageOrchestratorService
from interface.webhook_parser import parse_evolution_webhook

logger = logging.getLogger(__name__)

class InboxWorker:
    def __init__(
        self,
        session_factory,
        user_repo,
        conv_repo,
        media_service,
        embeddings,
        rag_repo,
        evolution_client,
        orchestrator: MessageOrchestratorService
    ):
        self.session_factory = session_factory
        self.user_repo = user_repo
        self.conv_repo = conv_repo
        self.media_service = media_service
        self.embeddings = embeddings
        self.rag_repo = rag_repo
        self.evolution_client = evolution_client
        self.orchestrator = orchestrator

    async def process_inbox(self) -> int:
        settings = get_settings()

        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'processing',
                            locked_at = NOW(),
                            retry_count = retry_count + 1,
                            updated_at = NOW()
                        WHERE id IN (
                            SELECT id FROM incoming_messages
                            WHERE status IN ('pending', 'failed')
                              AND retry_count < :max_retry
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

        processed = 0
        for inbox_msg in messages:
            try:
                await self._process_single_message(inbox_msg)
                processed += 1
            except Exception as e:
                logger.exception("Error procesando mensaje inbox id=%s: %s", inbox_msg.id, e)
                async with self.session_factory() as session:
                    async with session.begin():
                        await session.execute(
                            text("""
                                UPDATE incoming_messages
                                SET status = 'failed',
                                    error_detail = :err,
                                    updated_at = NOW()
                                WHERE id = :id
                            """),
                            {"err": str(e)[:500], "id": inbox_msg.id},
                        )

        return processed

    async def _process_single_message(self, inbox_msg) -> None:
        msg = parse_evolution_webhook(inbox_msg.webhook_payload)
        if not msg:
            async with self.session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                        {"id": inbox_msg.id},
                    )
            return

        user = await self.user_repo.get_or_create(msg.phone)
        normalized = await self.media_service.normalize(msg)
        asyncio.create_task(self.evolution_client.send_presence(normalized.phone, "composing"))

        rag_text = None
        try:
            query_embedding = await self.embeddings.embed(normalized.text)
            if query_embedding:
                rag_fragments = await self.rag_repo.search(query_embedding)
                if rag_fragments:
                    rag_text = "\n---\n".join(rag_fragments)
        except Exception:
            logger.exception("Error en RAG pipeline, continuando sin contexto")

        state_snapshot = await self.conv_repo.get_state_no_lock(user.id)

        async with self.session_factory() as session:
            async with session.begin():
                state = await self.conv_repo.get_state_for_update(session, user.id)

                if state.version > state_snapshot.version:
                    raise Exception(f"Estado cambió (v{state_snapshot.version} → v{state.version}) mientras se procesaba.")

                final_reply, new_response_id = await self.orchestrator.process_turn(
                    session=session,
                    state=state,
                    state_snapshot=state_snapshot,
                    user=user,
                    normalized=normalized,
                    rag_text=rag_text,
                    factory=self.session_factory
                )

                outbound_type = "audio_tts" if msg.content_type in (MessageType.AUDIO, MessageType.PTT) else "text"

                try:
                    await session.execute(
                        text("""
                            INSERT INTO outgoing_messages
                                (idempotency_key, usuario_id, phone, content_type, content)
                            VALUES (:ikey, :uid, :ph, :ctype, :txt)
                        """),
                        {
                            "ikey": f"reply:{msg.provider_message_id}:{outbound_type}",
                            "uid": user.id,
                            "ph": msg.phone,
                            "ctype": outbound_type,
                            "txt": final_reply,
                        },
                    )
                except Exception as e:
                    if "UniqueViolation" in str(e) or "duplicate key" in str(e).lower():
                        logger.warning("Idempotency hit for message %s, skipping insert.", msg.provider_message_id)
                        await session.rollback()
                        return 
                    else:
                        raise

                await session.execute(
                    text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                    {"id": inbox_msg.id},
                )

                state.last_openai_response_id = new_response_id
                state.last_provider_message_id = msg.provider_message_id
                state.turns_since_last_prompt += 1
                state.version += 1
                await self.conv_repo.save_state(session, state)
