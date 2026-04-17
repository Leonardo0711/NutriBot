"""
Nutribot Backend — InboxWorker
Consumidor del Inbox: consume IDs desde Redis (o SQL fallback)
y delega al MessageOrchestratorService.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from domain.exceptions import ConcurrentStateUpdateError
from domain.value_objects import MessageType
from config import get_settings
from infrastructure.db.connection import get_session_factory
from infrastructure.redis.client import (
    dequeue, enqueue, acquire_lock, release_lock,
    INBOX_QUEUE, OUTBOX_QUEUE,
)
from application.services.message_orchestrator import MessageOrchestratorService
from interface.webhook_parser import parse_evolution_webhook
from domain.router import classify_message, Intent

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
        """Consume mensajes: primero de Redis, luego fallback SQL."""
        processed = 0

        # --- Estrategia 1: Consumir de Redis (instantáneo) ---
        redis_available = True
        try:
            msg_id = await dequeue(INBOX_QUEUE, timeout=0.5)
        except Exception:
            msg_id = None
            redis_available = False

        if msg_id:
            inbox_msg = await self._load_and_lock_by_id(int(msg_id))
            if inbox_msg:
                try:
                    await self._process_single_message(inbox_msg)
                    processed += 1
                except ConcurrentStateUpdateError as e:
                    logger.info("Conflicto de concurrencia recuperable inbox id=%s: %s", inbox_msg.id, e)
                    await self._mark_pending_recoverable(inbox_msg.id, str(e))
                except Exception as e:
                    logger.exception("Error procesando mensaje inbox id=%s: %s", inbox_msg.id, e)
                    await self._mark_failed(inbox_msg.id, str(e))
            return processed

        # --- Estrategia 2: Fallback SQL (polling clásico) ---
        if not redis_available or not msg_id:
            messages = await self._claim_from_sql()
            if not messages:
                return 0

            for inbox_msg in messages:
                try:
                    await self._process_single_message(inbox_msg)
                    processed += 1
                except ConcurrentStateUpdateError as e:
                    logger.info("Conflicto de concurrencia recuperable inbox id=%s: %s", inbox_msg.id, e)
                    await self._mark_pending_recoverable(inbox_msg.id, str(e))
                except Exception as e:
                    logger.exception("Error procesando mensaje inbox id=%s: %s", inbox_msg.id, e)
                    await self._mark_failed(inbox_msg.id, str(e))

        return processed

    async def _load_and_lock_by_id(self, msg_id: int):
        """Carga un mensaje por ID desde Redis y lo marca como processing."""
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'processing',
                            locked_at = NOW(),
                            updated_at = NOW()
                        WHERE id = :id AND status = 'pending'
                        RETURNING *
                    """),
                    {"id": msg_id},
                )
                row = result.fetchone()
        return row

    async def _claim_from_sql(self):
        """Fallback: reclama mensajes pendientes via SQL polling."""
        settings = get_settings()
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text("""
                        UPDATE incoming_messages
                        SET status = 'processing',
                            locked_at = NOW(),
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
                return result.fetchall()

    # Intents que NO necesitan RAG ni embedding (ruta barata)
    _CHEAP_INTENTS = frozenset({
        Intent.GREETING, Intent.CONFIRMATION, Intent.DENIAL, Intent.SKIP,
        Intent.RESET, Intent.SMALL_TALK, Intent.ANSWER_CURRENT_STEP,
        Intent.PROFILE_UPDATE, Intent.CORRECTION_PAST_FIELD,
        Intent.PERSONALIZE_REQUEST, Intent.SURVEY_CONTINUE,
    })

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

        default_outbound_type = "audio_tts" if msg.content_type in (MessageType.AUDIO, MessageType.PTT) else "text"
        idempotency_key = f"reply:{msg.provider_message_id}"

        # Si ya existe respuesta para este provider_message_id, no reprocesar.
        if await self._outbox_exists(idempotency_key):
            async with self.session_factory() as session:
                async with session.begin():
                    await session.execute(
                        text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                        {"id": inbox_msg.id},
                    )
            return

        user = await self.user_repo.get_or_create(msg.phone)

        # Lock por usuario para evitar condiciones de carrera
        user_lock_key = f"user:{user.id}"
        lock_token = None
        redis_lock_unavailable = False
        settings = get_settings()
        try:
            lock_token = await acquire_lock(
                user_lock_key,
                ttl_seconds=settings.processing_lock_ttl_seconds,
            )
        except Exception:
            redis_lock_unavailable = True
            logger.exception("Redis lock no disponible, continuando con fallback DB user=%s", user.id)

        if not lock_token and not redis_lock_unavailable:
            await self._mark_pending_recoverable(inbox_msg.id, "lock_not_acquired")
            return

        try:
            # 1. NORMALIZAR (incluye STT para audio)
            normalized = await self.media_service.normalize(msg)

            # 2. RUTEAR POR TEXTO TRANSCRITO (no por content_type)
            #    Audio pasa por STT primero y se rutea como texto.
            #    used_audio es metadato, no razón para ruta cara.
            router_content_type = "text"
            if msg.content_type == MessageType.IMAGE:
                router_content_type = "image"
            # Audio ya fue transcrito a texto por normalize(), se rutea como texto

            state_snapshot = await self.conv_repo.get_state_no_lock(user.id)

            route = classify_message(
                raw_text=normalized.text,
                current_mode=state_snapshot.mode,
                onboarding_status=state_snapshot.onboarding_status,
                onboarding_step=state_snapshot.onboarding_step,
                content_type=router_content_type,
            )
            logger.info(
                "Router-first: user=%s intent=%s conf=%.2f reason='%s'",
                user.id, route.intent.value, route.confidence, route.reason,
            )

            # 3. CONDICIONAL: solo generar embedding + RAG si el intent lo requiere
            rag_text = None
            if route.intent not in self._CHEAP_INTENTS:
                try:
                    query_embedding = await self.embeddings.embed(normalized.text)
                    if query_embedding:
                        rag_fragments = await self.rag_repo.search(query_embedding)
                        if rag_fragments:
                            rag_text = "\n---\n".join(rag_fragments)
                except Exception:
                    logger.exception("Error en RAG pipeline, continuando sin contexto")
            else:
                logger.debug("Skipping RAG for cheap intent %s user=%s", route.intent.value, user.id)

            # 4. ORQUESTAR
            async with self.session_factory() as session:
                async with session.begin():
                    state = await self.conv_repo.get_state_for_update(session, user.id)

                    if state.version > state_snapshot.version:
                        raise ConcurrentStateUpdateError(
                            f"Estado cambió (v{state_snapshot.version} → v{state.version}) mientras se procesaba."
                        )

                    # INICIO DEL SAVEPOINT. Ninguna mutación sobrevivirá si gana la carrera de idempotencia.
                    nested_sp = await session.begin_nested()

                    bot_reply, new_response_id = await self.orchestrator.process_turn(
                        session=session,
                        state=state,
                        state_snapshot=state_snapshot,
                        user=user,
                        normalized=normalized,
                        rag_text=rag_text,
                        factory=self.session_factory,
                        route=route,
                    )
                    outbound_type = (getattr(bot_reply, "content_type", None) or "text").strip()
                    if outbound_type == "text" and default_outbound_type == "audio_tts":
                        outbound_type = "audio_tts"

                    payload_json = getattr(bot_reply, "payload_json", None)
                    safe_reply = str(getattr(bot_reply, "text", "") or "").strip()
                    if outbound_type.startswith("interactive_") and not safe_reply:
                        safe_reply = str((payload_json or {}).get("body") or "").strip()

                    if not safe_reply:
                        safe_reply = "Perdon, tuve un problema interno. Intenta nuevamente en unos segundos."
                        outbound_type = "text"
                        payload_json = None
                        logger.warning(
                            "Fallback por respuesta vacia user=%s provider_id=%s",
                            user.id,
                            msg.provider_message_id,
                        )

                    outbox_insert_stmt = text("""
                            INSERT INTO outgoing_messages
                                (idempotency_key, usuario_id, phone, content_type, content, payload_json, scheduled_at)
                            VALUES (:ikey, :uid, :ph, :ctype, :txt, :payload, NOW())
                            ON CONFLICT (idempotency_key) DO NOTHING
                            RETURNING id
                        """).bindparams(bindparam("payload", type_=JSONB))
                    result = await session.execute(
                        outbox_insert_stmt,
                        {
                            "ikey": idempotency_key,
                            "uid": user.id,
                            "ph": msg.phone,
                            "ctype": outbound_type,
                            "txt": safe_reply,
                            "payload": payload_json if payload_json else None,
                        },
                    )
                    outbox_row = result.fetchone()

                    if outbox_row is None:
                        logger.warning("Idempotency hit for message %s, skipping duplicate send and rolling back side-effects.", msg.provider_message_id)
                        await nested_sp.rollback()
                        
                        # Marcamos como "done" fuera del savepoint revocado pero dentro de la transaccion principal
                        await session.execute(
                            text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                            {"id": inbox_msg.id},
                        )
                        return
                    else:
                        await nested_sp.commit()

                    # Side-effects solo si el outbox ganó
                    await session.execute(
                        text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                        {"id": inbox_msg.id},
                    )

                    state.last_openai_response_id = new_response_id
                    state.last_provider_message_id = msg.provider_message_id
                    state.turns_since_last_prompt += 1
                    state.version += 1
                    await self.conv_repo.save_state(session, state)

                    # Persistir memoria solo después de confirmar que el outbox ganó
                    await self.orchestrator._append_to_chat_memory(session, user.id, normalized.text, safe_reply)

            # Publicar outbox ID en Redis para entrega instantánea
            if outbox_row:
                try:
                    await enqueue(OUTBOX_QUEUE, outbox_row.id)
                except Exception:
                    pass  # OutboxWorker usará fallback SQL

        finally:
            # Siempre liberar el lock del usuario
            if lock_token:
                try:
                    released = await release_lock(user_lock_key, lock_token)
                    if not released:
                        logger.warning(
                            "Lock no liberado por mismatch user=%s lock=%s token_suffix=%s",
                            user.id,
                            user_lock_key,
                            lock_token[-8:],
                        )
                except Exception:
                    logger.exception(
                        "Error liberando lock user=%s lock=%s token_suffix=%s",
                        user.id,
                        user_lock_key,
                        lock_token[-8:],
                    )

    async def _outbox_exists(self, idempotency_key: str) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT 1 FROM outgoing_messages WHERE idempotency_key = :ikey LIMIT 1"),
                {"ikey": idempotency_key},
            )
            return result.scalar() is not None

    async def _mark_failed(self, msg_id: int, error: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                try:
                    await session.execute(
                        text("""
                            UPDATE incoming_messages
                            SET status = 'failed',
                                retry_count = retry_count + 1,
                                transient_error_count = transient_error_count + 1,
                                error_detail = :err,
                                updated_at = NOW()
                            WHERE id = :id
                        """),
                        {"err": error[:500], "id": msg_id},
                    )
                except Exception:
                    # Compatibilidad temporal si aun no corrio la migracion 011.
                    await session.execute(
                        text("""
                            UPDATE incoming_messages
                            SET status = 'failed',
                                retry_count = retry_count + 1,
                                error_detail = :err,
                                updated_at = NOW()
                            WHERE id = :id
                        """),
                        {"err": error[:500], "id": msg_id},
                    )

    async def _mark_pending_recoverable(self, msg_id: int, detail: str) -> None:
        """Reencola con backoff cuando ocurre un conflicto recuperable."""
        async with self.session_factory() as session:
            async with session.begin():
                try:
                    await session.execute(
                        text(
                            """
                            UPDATE incoming_messages
                            SET status = 'pending',
                                locked_at = NULL,
                                conflict_count = conflict_count + 1,
                                error_detail = :detail,
                                updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        {"detail": detail[:500], "id": msg_id},
                    )
                except Exception:
                    # Compatibilidad temporal si aun no corrio la migracion 011.
                    await session.execute(
                        text(
                            """
                            UPDATE incoming_messages
                            SET status = 'pending',
                                locked_at = NULL,
                                error_detail = :detail,
                                updated_at = NOW()
                            WHERE id = :id
                            """
                        ),
                        {"detail": detail[:500], "id": msg_id},
                    )
        delay_seconds = 0.25 + random.random() * 0.75
        try:
            await asyncio.sleep(delay_seconds)
            await enqueue(INBOX_QUEUE, msg_id)
            logger.info("Reencolado recoverable inbox id=%s delay=%.3fs", msg_id, delay_seconds)
        except Exception:
            # Si Redis no está disponible, el fallback SQL lo volverá a reclamar.
            logger.exception("No se pudo reencolar recoverable inbox id=%s", msg_id)
