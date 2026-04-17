"""
Nutribot Backend - MessageOrchestratorService
Coordinador del pipeline conversacional.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.llm_reply_service import LlmReplyService
from application.services.onboarding_service import OnboardingService
from application.services.profile_context_service import ProfileContextService
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_interception_service import ProfileInterceptionService
from application.services.profile_read_service import ProfileReadService
from application.services.survey_flow_service import SurveyFlowService
from domain.entities import ConversationState, NormalizedMessage, User
from domain.profile_snapshot import ProfileSnapshot
from domain.reply_objects import BotReply
from domain.router import Intent, RouteResult
from domain.value_objects import OnboardingStatus, OnboardingStep, SessionMode

logger = logging.getLogger(__name__)


class MessageOrchestratorService:
    def __init__(
        self,
        onboarding_service: OnboardingService,
        profile_extractor: ProfileExtractionService,
        profile_reader: ProfileReadService,
        profile_context: ProfileContextService,
        profile_interception: ProfileInterceptionService,
        llm_reply: LlmReplyService,
        survey_flow: SurveyFlowService,
    ):
        self._onboarding_service = onboarding_service
        self._profile_extractor = profile_extractor
        self._profile_reader = profile_reader
        self._profile_context = profile_context
        self._profile_interception = profile_interception
        self._llm_reply = llm_reply
        self._survey_flow = survey_flow

    async def _get_recent_history(self, session: AsyncSession, uid: int) -> list[dict]:
        try:
            res = await session.execute(
                text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid"),
                {"uid": uid},
            )
            val = res.scalar()
            return val if isinstance(val, list) else []
        except Exception as e:
            logger.error("Error recuperando historial para usuario %s: %s", uid, e)
            return []

    async def _append_to_chat_memory(self, session: AsyncSession, uid: int, user_text: str, assistant_reply: str):
        try:
            new_pair = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_reply},
            ]

            await session.execute(
                text(
                    """
                    INSERT INTO memoria_chat (usuario_id, historial_mensajes, actualizado_en)
                    VALUES (:uid, '[]'::jsonb, NOW())
                    ON CONFLICT (usuario_id) DO NOTHING
                    """
                ),
                {"uid": uid},
            )

            res = await session.execute(
                text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid FOR UPDATE"),
                {"uid": uid},
            )
            hist = res.scalar() or []
            hist.extend(new_pair)
            hist = hist[-20:]

            update_stmt = text(
                """
                UPDATE memoria_chat
                SET historial_mensajes = :hist,
                    actualizado_en = NOW()
                WHERE usuario_id = :uid
                """
            ).bindparams(bindparam("hist", type_=JSONB))
            await session.execute(update_stmt, {"uid": uid, "hist": hist})
        except Exception as e:
            logger.error("Error actualizando memoria_chat para usuario %s: %s", uid, e)

    async def _load_profile_snapshot(self, session: AsyncSession, uid: int) -> ProfileSnapshot:
        snapshot = await self._profile_reader.fetch_snapshot(session, uid)
        if snapshot:
            return snapshot
        return ProfileSnapshot.from_row({"usuario_id": uid, "skipped_fields": {}})

    @staticmethod
    def _derive_route_flags(route: RouteResult) -> dict[str, bool]:
        return {
            "looks_like_profile_update": route.intent in (
                Intent.PROFILE_UPDATE,
                Intent.CORRECTION_PAST_FIELD,
                Intent.ANSWER_CURRENT_STEP,
            ),
            "is_asking_for_recommendation": route.intent in (
                Intent.NUTRITION_QUERY,
                Intent.RECOMMENDATION_REQUEST,
            ),
            "is_short_greeting": route.intent == Intent.GREETING,
            "is_requesting_personalization": route.intent == Intent.PERSONALIZE_REQUEST,
            "is_requesting_survey": route.intent == Intent.SURVEY_CONTINUE,
        }

    async def process_turn(
        self,
        session: AsyncSession,
        state: ConversationState,
        state_snapshot: ConversationState,
        user: User,
        normalized: NormalizedMessage,
        rag_text: Optional[str],
        factory,
        route: RouteResult,
    ) -> tuple[BotReply, Optional[str]]:
        snapshot = await self._load_profile_snapshot(session, user.id)
        profile_text, summary = self._profile_context.build_prompt_and_summary(snapshot)
        history = await self._get_recent_history(session, user.id)

        reply = None
        mode_before_survey = state.mode

        logger.info(
            "Router: user=%s intent=%s conf=%.2f field=%s value=%s reason='%s'",
            user.id,
            route.intent.value,
            route.confidence,
            route.resolved_field,
            route.resolved_value,
            route.reason,
        )

        route_flags = self._derive_route_flags(route)
        looks_like_profile_update = route_flags["looks_like_profile_update"]
        is_asking_for_recommendation = route_flags["is_asking_for_recommendation"]
        is_short_greeting = route_flags["is_short_greeting"]
        is_requesting_personalization = route_flags["is_requesting_personalization"]
        is_requesting_survey = route_flags["is_requesting_survey"]

        if route.intent == Intent.RESET:
            await self._onboarding_service._handle_system_reset(user.id, session)
            state.onboarding_status = OnboardingStatus.INVITED.value
            state.onboarding_step = OnboardingStep.INVITACION.value
            state.mode = SessionMode.ACTIVE_CHAT.value
            state.version += 1
            return BotReply(
                text="He borrado tus datos de perfil para empezar de cero cuando quieras. Quieres que empecemos ahora?",
                content_type="text",
            ), None

        onboarding_interception_happened = False
        extracted_data: dict = {}
        has_absurd_profile_claim = False

        if state.onboarding_status in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            onboarding_interception_happened = True
            reply = await self._onboarding_service.advance_flow(
                normalized.text,
                state,
                session,
                treat_ninguna_as_missing=False,
                pre_extracted_data=None,
                history=history,
            )
            if reply is None:
                onboarding_interception_happened = False

        if not onboarding_interception_happened and (looks_like_profile_update or is_requesting_personalization):
            has_absurd_profile_claim = self._profile_extractor.contains_absurd_claim(normalized.text)
            if (
                route.resolved_field
                and route.resolved_value
                and route.confidence >= 0.8
                and route.intent in (Intent.PROFILE_UPDATE, Intent.CORRECTION_PAST_FIELD, Intent.ANSWER_CURRENT_STEP)
            ):
                raw_extractions = {route.resolved_field: route.resolved_value}
                ext_result = await self._profile_extractor.apply_cleaning_and_save(
                    raw_extractions=raw_extractions,
                    user_text=normalized.text,
                    usuario_id=user.id,
                    session=session,
                    current_step=route.resolved_field,
                )
                logger.info(
                    "Router-based profile update (no LLM): user=%s field=%s value=%s",
                    user.id,
                    route.resolved_field,
                    route.resolved_value,
                )
            else:
                ext_result = await self._profile_extractor.extract_and_save(
                    user_text=normalized.text,
                    usuario_id=user.id,
                    session=session,
                    current_step=None,
                )

            if ext_result:
                extracted_data = ext_result.clean_data
                meta_flags = ext_result.meta_flags
                if extracted_data:
                    logger.info("Real-time profile update user=%s: %s", user.id, extracted_data)
                    snapshot = await self._load_profile_snapshot(session, user.id)
                    profile_text, summary = self._profile_context.build_prompt_and_summary(snapshot)
                if meta_flags.get("needs_health_clarification"):
                    return BotReply(
                        text=meta_flags.get("clarification_prompt", "Podrias aclarar ese aspecto medico un poco mas?"),
                        content_type="text",
                    ), None

        reply, onboarding_interception_happened = await self._profile_interception.maybe_start_personalization_flow(
            session=session,
            state=state,
            user_id=user.id,
            snapshot=snapshot,
            summary=summary,
            reply=reply,
            onboarding_interception_happened=onboarding_interception_happened,
            is_requesting_personalization=is_requesting_personalization,
            is_asking_for_recommendation=is_asking_for_recommendation,
        )

        reply, onboarding_interception_happened = await self._profile_interception.maybe_intercept_for_missing_profile(
            session=session,
            state=state,
            user_id=user.id,
            snapshot=snapshot,
            reply=reply,
            onboarding_interception_happened=onboarding_interception_happened,
            is_short_greeting=is_short_greeting,
            is_asking_for_recommendation=is_asking_for_recommendation,
        )

        reply, new_response_id = await self._llm_reply.generate_reply(
            onboarding_interception_happened=onboarding_interception_happened,
            reply=reply,
            state_snapshot=state_snapshot,
            normalized=normalized,
            route=route,
            rag_text=rag_text,
            history=history,
            profile_text=profile_text,
            snapshot=snapshot,
            extracted_data=extracted_data,
            has_absurd_profile_claim=has_absurd_profile_claim,
            is_asking_for_recommendation=is_asking_for_recommendation,
        )

        base_should_count_interaction = bool(not onboarding_interception_happened and reply)
        should_count_before_survey = bool(
            base_should_count_interaction and mode_before_survey != SessionMode.COLLECTING_USABILITY.value
        )
        projected_interactions_count = state.meaningful_interactions_count + (1 if should_count_before_survey else 0)

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

        reply = self._llm_reply.append_continuity_tip(
            reply=reply,
            onboarding_interception_happened=onboarding_interception_happened,
            turns_since_last_prompt=state.turns_since_last_prompt,
            is_requesting_survey=is_requesting_survey,
        )

        final_bot_reply, survey_was_interrupted = await self._survey_flow.compose_reply_with_survey(
            session=session,
            state=state,
            normalized=normalized,
            user=user,
            reply=reply,
            new_response_id=new_response_id,
            onboarding_interception_happened=onboarding_interception_happened,
            is_requesting_survey=is_requesting_survey,
            projected_interactions_count=projected_interactions_count,
            schedule_separate_message=self._schedule_separate_message,
        )

        if (should_count_before_survey or survey_was_interrupted) and state.mode == SessionMode.ACTIVE_CHAT.value:
            if survey_was_interrupted:
                state.meaningful_interactions_count = 0
            else:
                state.meaningful_interactions_count = projected_interactions_count
            logger.info("Universal interaction counter for user %s: %s", user.id, state.meaningful_interactions_count)

        final_bot_reply = self._llm_reply.sanitize_final_reply(final_bot_reply, user.id)
        return final_bot_reply, new_response_id

    async def _schedule_separate_message(self, session: AsyncSession, uid: int, phone: str, addon: BotReply, idemp_key: str):
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
