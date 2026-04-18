"""
Nutribot Backend - MessageOrchestratorService
Coordinador del pipeline conversacional. Funciona como director de la orquesta,
construyendo el contexto, resolviendo el handler y delegando la ejecucion.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState, NormalizedMessage, User
from domain.reply_objects import BotReply
from domain.router import RouteResult
from application.services.handlers.handler_registry import HandlerRegistry
from application.services.turn_context_service import TurnContextService
from application.services.conversation_memory_service import ConversationMemoryService
from application.services.conversation_state_service import ConversationStateService

logger = logging.getLogger(__name__)


class MessageOrchestratorService:
    def __init__(
        self,
        turn_context_service: TurnContextService,
        handler_registry: HandlerRegistry,
        memory_service: ConversationMemoryService,
        state_service: ConversationStateService,
    ):
        self._turn_context_service = turn_context_service
        self._handler_registry = handler_registry
        self._memory_service = memory_service
        self._state_service = state_service

    async def _append_to_chat_memory(self, session: AsyncSession, uid: int, user_text: str, assistant_reply: str):
        # Delegado directamente al MemoryService
        await self._memory_service.append_turn(session, uid, user_text, assistant_reply)

    async def process_turn(
        self,
        session: AsyncSession,
        state: ConversationState,
        state_snapshot: ConversationState,
        user: User,
        normalized: NormalizedMessage,
        rag_text: Optional[str],
        factory,  # Not strictly needed anymore but keeping signature for backward compatibility
        route: RouteResult,
    ) -> tuple[BotReply, Optional[str]]:
        
        logger.info(
            "Orchestrator: Empezando turno user=%s intent=%s conf=%.2f",
            user.id,
            route.intent.value,
            route.confidence,
        )

        # 1. Cargar contexto minimo del turno
        ctx = await self._turn_context_service.build(
            session=session,
            user=user,
            state=state,
            state_snapshot=state_snapshot,
            normalized=normalized,
            route=route,
            rag_text=rag_text,
        )

        # 2. Decidir que flujo aplica
        handler = self._handler_registry.resolve(ctx)
        
        logger.info("Orchestrator: Delegando a %s", handler.__class__.__name__)

        # 3. Delegar al handler
        bot_reply, new_response_id = await handler.handle(ctx)

        # Tracking de uso de recursos
        if normalized.used_audio:
            await session.execute(
                text("UPDATE formulario_en_progreso SET uso_audio = TRUE WHERE usuario_id = :uid"),
                {"uid": user.id},
            )
        if normalized.image_base64:
            await session.execute(
                text("UPDATE formulario_en_progreso SET uso_imagen = TRUE WHERE usuario_id = :uid"),
                {"uid": user.id},
            )

        # 4. Actualizar Estado Unificado
        self._state_service.update_interaction_details(
            state=ctx.state,
            provider_message_id=normalized.provider_message_id,
            openai_response_id=new_response_id
        )

        # El Orchestrator devuelve esto para que el InboxWorker persista el outbox record 
        return bot_reply, new_response_id

    async def _schedule_separate_message(self, session: AsyncSession, uid: int, phone: str, addon: BotReply, idemp_key: str):
        try:
            stmt = text(
                """
                INSERT INTO outgoing_messages (usuario_id, phone, content_type, content, payload_json, idempotency_key, status, scheduled_at, created_at, updated_at)
                VALUES (:uid, :phone, :ctype, :content, :payload, :key, 'pending', TIMEZONE('America/Lima', NOW()) + INTERVAL '1 second', TIMEZONE('America/Lima', NOW()), TIMEZONE('America/Lima', NOW()))
                ON CONFLICT (idempotency_key) DO NOTHING
                """
            ).bindparams(bindparam("payload", type_=JSONB))
            await session.execute(
                stmt,
                {
                    "uid": uid,
                    "phone": phone,
                    "ctype": addon.content_type,
                    "content": addon.text or "",
                    "payload": addon.payload_json,
                    "key": idemp_key,
                },
            )
            logger.info("Scheduling separate message for user %s, key=%s", uid, idemp_key)
        except Exception as e:
            logger.error("Error scheduling separate message: %s", e)


