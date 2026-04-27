"""
Nutribot Backend - MessageOrchestratorService
Coordinador del pipeline conversacional. Funciona como director de la orquesta,
construyendo el contexto, resolviendo el handler y delegando la ejecucion.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState, NormalizedMessage, User
from domain.reply_objects import BotReply
from domain.router import RouteResult, Intent
from domain.value_objects import OnboardingStatus, SessionMode
from application.services.handlers.handler_registry import HandlerRegistry
from application.services.turn_context_service import TurnContextService
from application.services.conversation_memory_service import ConversationMemoryService
from application.services.conversation_state_service import ConversationStateService
from application.services.profile_intent_extractor_service import ProfileIntentExtractorService

logger = logging.getLogger(__name__)

# Señales textuales que indican que un mensaje NUTRITION_QUERY/RECOMMENDATION
# podría contener datos de perfil implícitos.
_PROFILE_SIGNAL_RE = re.compile(
    r"(?:soy\s+(?:alergic|intolerant|vegetarian|vegan|diabetic))"
    r"|(?:tengo\s+(?:diabetes|hipertension|gastritis|anemia|colesterol))"
    r"|(?:no\s+(?:como|consumo|puedo\s+comer))"
    r"|(?:evito\s+)"
    r"|(?:prefiero\s+no\s+comer)"
    r"|(?:peso\s+\d)"
    r"|(?:mido\s+\d)",
    re.IGNORECASE,
)


class MessageOrchestratorService:
    def __init__(
        self,
        turn_context_service: TurnContextService,
        handler_registry: HandlerRegistry,
        memory_service: ConversationMemoryService,
        state_service: ConversationStateService,
        profile_intent_extractor: ProfileIntentExtractorService | None = None,
    ):
        self._turn_context_service = turn_context_service
        self._handler_registry = handler_registry
        self._memory_service = memory_service
        self._state_service = state_service
        self._profile_intent_extractor = profile_intent_extractor

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

        # 2. Extracción de intención de perfil (ANTES de decidir handler)
        #    Gate inteligente: solo pagar LLM cuando vale la pena.
        if self._profile_intent_extractor and self._should_run_profile_intent_extractor(ctx):
            ctx.profile_intent = await self._extract_profile_intent(ctx)

            if ctx.profile_intent and ctx.profile_intent.is_profile_update:
                ctx.looks_like_profile_update = True
                ctx.turn_kind = "PROFILE_MAINTENANCE"
                logger.info(
                    "Orchestrator: Intent extractor detected profile update user=%s field=%s op=%s conf=%.2f",
                    user.id,
                    ctx.profile_intent.field_code,
                    ctx.profile_intent.operation,
                    ctx.profile_intent.confidence,
                )
        # 2b. Pre-clasificar turn_kind ANTES del handler (el handler ya no trabaja con None)
        ctx.turn_kind = self._preclassify_turn_kind(ctx)

        # 3. Decidir que flujo aplica (ahora con contexto de intención)
        handler = self._handler_registry.resolve(ctx)
        
        logger.info("Orchestrator: Delegando a %s", handler.__class__.__name__)

        # 4. Delegar al handler
        bot_reply, new_response_id = await handler.handle(ctx)

        # 5. Refinar turn_kind de forma centralizada (post-handler)
        ctx.turn_kind = self._classify_turn_kind(ctx, handler)

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

        # 6. Actualizar Estado Unificado
        self._state_service.update_interaction_details(
            state=ctx.state,
            provider_message_id=normalized.provider_message_id,
            openai_response_id=new_response_id
        )

        # El Orchestrator devuelve esto para que el InboxWorker persista el outbox record 
        return bot_reply, new_response_id

    def _should_run_profile_intent_extractor(self, ctx) -> bool:
        """
        Gate inteligente: decide si vale la pena llamar al extractor LLM.
        Evita gastar tokens en saludos, imágenes sueltas, encuestas, resets, etc.
        """
        txt = (ctx.normalized.text or "").strip().lower()
        if not txt:
            return False

        # Durante onboarding: correr EXCEPTO para intents triviales que no
        # aportan datos de perfil (saludos, confirmaciones, negaciones, skips).
        # Esto ahorra ~1.5s de LLM por turno trivial.
        if ctx.state.awaiting_field_code or ctx.state.onboarding_step:
            trivial_during_onboarding = {
                Intent.GREETING, Intent.CONFIRMATION, Intent.DENIAL,
                Intent.SKIP, Intent.SMALL_TALK,
            }
            if ctx.route.intent in trivial_during_onboarding and ctx.route.confidence >= 0.80:
                return False
            return True

        # Correr durante encuesta si el usuario dice algo que no es respuesta numérica
        if ctx.state.mode == SessionMode.COLLECTING_USABILITY.value:
            return True

        # Correr si el router indica que podría ser profile update o ambiguo
        if ctx.route.intent in {
            Intent.PROFILE_UPDATE,
            Intent.CORRECTION_PAST_FIELD,
            Intent.ANSWER_CURRENT_STEP,
            Intent.AMBIGUOUS,
        }:
            return True

        # Para NUTRITION_QUERY y RECOMMENDATION_REQUEST, solo si hay señal de perfil
        if ctx.route.intent in {Intent.NUTRITION_QUERY, Intent.RECOMMENDATION_REQUEST}:
            return bool(_PROFILE_SIGNAL_RE.search(txt))

        # GREETING, RESET, SURVEY_CONTINUE, CONFIRMATION, DENIAL, SKIP, IMAGE, AUDIO
        # → NO correr. No gastar tokens.
        return False

    def _preclassify_turn_kind(self, ctx) -> str:
        """
        Pre-clasificación del tipo de turno ANTES del handler.
        Permite que el handler trabaje con un turn_kind informado.
        Se refinará después del handler en _classify_turn_kind().
        """
        if ctx.profile_intent and ctx.profile_intent.is_profile_update:
            return "PROFILE_MAINTENANCE"

        if ctx.state.onboarding_status in (
            OnboardingStatus.INVITED.value,
            OnboardingStatus.IN_PROGRESS.value,
        ):
            return "ONBOARDING_RESPONSE"

        if ctx.state.mode == SessionMode.COLLECTING_USABILITY.value:
            return "SURVEY_RESPONSE"

        if ctx.route.intent in (Intent.NUTRITION_QUERY, Intent.RECOMMENDATION_REQUEST):
            return "NUTRITION_VALUE"

        return "OTHER"

    def _classify_turn_kind(self, ctx, handler) -> str:
        """
        Clasificación centralizada del tipo de turno DESPUÉS de ejecutar el handler.
        Determina si el contador de interacciones significativas debe incrementarse.
        """
        # Si ya fue clasificado como PROFILE_MAINTENANCE por el extractor, mantener
        if ctx.turn_kind == "PROFILE_MAINTENANCE":
            return "PROFILE_MAINTENANCE"

        # Onboarding
        if ctx.onboarding_interception_happened:
            return "ONBOARDING_RESPONSE"

        # Encuesta
        if ctx.state.mode == SessionMode.COLLECTING_USABILITY.value:
            return "SURVEY_RESPONSE"

        # Si hubo actualización de perfil via legacy path
        if ctx.looks_like_profile_update or ctx.extracted_data:
            return "PROFILE_MAINTENANCE"

        # Solo consultas nutricionales reales cuentan como NUTRITION_VALUE
        if ctx.route.intent in (Intent.NUTRITION_QUERY, Intent.RECOMMENDATION_REQUEST):
            return "NUTRITION_VALUE"

        if getattr(ctx, "is_asking_for_recommendation", False):
            return "NUTRITION_VALUE"

        # Todo lo demás (saludos, agradecimientos, mensajes neutros) no cuenta
        return "OTHER"

    async def _extract_profile_intent(self, ctx):
        """
        Ejecuta la extracción de intención de perfil.
        Solo llama al LLM cuando el router no puede clasificar con confianza.
        Fast paths del router (greeting, reset, encuesta, etc.) no gastan tokens.
        """
        try:
            expected_field = None
            if (
                ctx.state.onboarding_status == OnboardingStatus.IN_PROGRESS.value
                and ctx.state.onboarding_step
            ):
                expected_field = ctx.state.onboarding_step

            return await self._profile_intent_extractor.extract(
                user_text=ctx.normalized.text,
                session=ctx.session,
                expected_field=expected_field,
                usuario_id=ctx.user.id,
                router_intent=ctx.route.intent.value,
                router_confidence=ctx.route.confidence,
            )
        except Exception:
            logger.exception("Orchestrator: profile intent extraction failed")
            return None

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
