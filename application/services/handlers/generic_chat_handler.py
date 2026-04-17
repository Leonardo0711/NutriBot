"""
Nutribot Backend - Generic Chat Handler
El "catch-all" que encripta la llamada a OpenAI, las intercepciones de encuestas/perfil y formateo final.
"""
from typing import Optional, Tuple, Callable

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply
from domain.value_objects import SessionMode
from application.services.handlers.base_handler import BaseHandler
from application.services.llm_reply_service import LlmReplyService
from application.services.profile_interception_service import ProfileInterceptionService
from application.services.survey_flow_service import SurveyFlowService
from application.services.conversation_state_service import ConversationStateService


class GenericChatHandler(BaseHandler):
    def __init__(
        self,
        llm_reply: LlmReplyService,
        profile_interception: ProfileInterceptionService,
        survey_flow: SurveyFlowService,
        state_service: ConversationStateService,
    ):
        self._llm_reply = llm_reply
        self._profile_interception = profile_interception
        self._survey_flow = survey_flow
        self._state_service = state_service

    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        reply = None
        mode_before_survey = ctx.state.mode

        # 1. Profile Interception (faltan datos, quiere personalizar, etc.)
        reply, int_happened = await self._profile_interception.maybe_start_personalization_flow(
            session=ctx.session,
            state=ctx.state,
            user_id=ctx.user.id,
            snapshot=ctx.snapshot,
            summary=ctx.summary,
            reply=reply,
            onboarding_interception_happened=ctx.onboarding_interception_happened,
            is_requesting_personalization=ctx.is_requesting_personalization,
            is_asking_for_recommendation=ctx.is_asking_for_recommendation,
        )
        if int_happened:
            ctx.onboarding_interception_happened = True

        reply, int_happened = await self._profile_interception.maybe_intercept_for_missing_profile(
            session=ctx.session,
            state=ctx.state,
            user_id=ctx.user.id,
            snapshot=ctx.snapshot,
            reply=reply,
            onboarding_interception_happened=ctx.onboarding_interception_happened,
            is_short_greeting=ctx.is_short_greeting,
            is_asking_for_recommendation=ctx.is_asking_for_recommendation,
        )
        if int_happened:
            ctx.onboarding_interception_happened = True

        # 2. Generación LLM (si no hay una respuesta impuesta previamente)
        reply, new_response_id = await self._llm_reply.generate_reply(
            onboarding_interception_happened=ctx.onboarding_interception_happened,
            reply=reply,
            state_snapshot=ctx.state_snapshot,
            normalized=ctx.normalized,
            route=ctx.route,
            rag_text=ctx.rag_text,
            history=ctx.history,
            profile_text=ctx.profile_text,
            snapshot=ctx.snapshot,
            extracted_data=ctx.extracted_data,
            has_absurd_profile_claim=ctx.has_absurd_profile_claim,
            is_asking_for_recommendation=ctx.is_asking_for_recommendation,
        )

        base_should_count = bool(not ctx.onboarding_interception_happened and reply)
        should_count_before_survey = bool(
            base_should_count and mode_before_survey != SessionMode.COLLECTING_USABILITY.value
        )
        projected_interactions_count = ctx.state.meaningful_interactions_count + (1 if should_count_before_survey else 0)

        # 3. Profile Interception Phase 2 (sugerencias progresivas)
        if reply and not ctx.onboarding_interception_happened and not ctx.is_requesting_survey:
            reply = await self._profile_interception.maybe_suggest_phase2_field(
                session=ctx.session,
                state=ctx.state,
                user_id=ctx.user.id,
                snapshot=ctx.snapshot,
                reply=reply,
            )

        # 4. Postprocesamiento (tips de continuidad)
        reply = self._llm_reply.append_continuity_tip(
            reply=reply,
            onboarding_interception_happened=ctx.onboarding_interception_happened,
            turns_since_last_prompt=ctx.state.turns_since_last_prompt,
            is_requesting_survey=ctx.is_requesting_survey,
        )

        # 5. Inyección de Encuestas (Satisfacción, etc.)
        final_bot_reply, survey_was_interrupted = await self._survey_flow.compose_reply_with_survey(
            session=ctx.session,
            state=ctx.state,
            normalized=ctx.normalized,
            user=ctx.user,
            reply=reply,
            new_response_id=new_response_id,
            onboarding_interception_happened=ctx.onboarding_interception_happened,
            is_requesting_survey=ctx.is_requesting_survey,
            projected_interactions_count=projected_interactions_count,
            schedule_separate_message=self._schedule_separate_message,
        )

        # 6. Actualizar Analytics de Estado
        if (should_count_before_survey or survey_was_interrupted) and ctx.state.mode == SessionMode.ACTIVE_CHAT.value:
            self._state_service.update_meaningful_interaction_count(
                state=ctx.state,
                survey_was_interrupted=survey_was_interrupted,
                projected_interactions_count=projected_interactions_count
            )

        # 7. Limpieza Final (sanitización)
        final_bot_reply = self._llm_reply.sanitize_final_reply(final_bot_reply, ctx.user.id, localization=self._llm_reply._localization)
        
        return final_bot_reply, new_response_id

    async def _schedule_separate_message(self, session, uid: int, phone: str, addon: BotReply, idemp_key: str):
        import logging
        from sqlalchemy import text, bindparam
        from sqlalchemy.dialects.postgresql import JSONB
        
        logger = logging.getLogger(__name__)
        try:
            stmt = text(
                """
                INSERT INTO outgoing_messages (usuario_id, phone, content_type, content, payload_json, idempotency_key, status, scheduled_at, created_at, updated_at)
                VALUES (:uid, :phone, :ctype, :content, :payload, :key, 'pending', NOW() + INTERVAL '1 second', NOW(), NOW())
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


