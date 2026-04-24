"""
Nutribot Backend - Survey Flow Service
"""
from __future__ import annotations

from domain.entities import ConversationState, NormalizedMessage, User
from domain.reply_objects import BotReply
from domain.value_objects import SessionMode


class SurveyFlowService:
    def __init__(self, survey_service):
        self._survey_service = survey_service

    async def compose_reply_with_survey(
        self,
        *,
        session,
        state: ConversationState,
        normalized: NormalizedMessage,
        user: User,
        reply: str | None,
        new_response_id: str | None,
        onboarding_interception_happened: bool,
        is_requesting_survey: bool,
        projected_interactions_count: int,
        schedule_separate_message,
    ) -> tuple[BotReply, bool]:
        original_mode = state.mode
        original_awaiting = state.awaiting_question_code
        consent_state = "esperando_consentimiento_encuesta"
        force_survey_start = bool(
            not onboarding_interception_happened
            and original_mode == SessionMode.ACTIVE_CHAT.value
            and is_requesting_survey
        )
        survey_projected_count = max(5, projected_interactions_count) if force_survey_start else projected_interactions_count

        addon = await self._survey_service.process(
            session,
            state,
            normalized,
            projected_interactions_count=survey_projected_count,
        )

        if addon:
            survey_now_active = (
                state.mode == SessionMode.COLLECTING_USABILITY.value
                or bool(state.awaiting_question_code)
            )
            if (
                original_mode == SessionMode.COLLECTING_USABILITY.value
                or is_requesting_survey
                or original_awaiting == consent_state
                or survey_now_active
            ):
                final_bot_reply = addon
            elif reply:
                final_bot_reply = BotReply(text=reply, content_type="text")
                addon_seed = new_response_id or normalized.provider_message_id
                await schedule_separate_message(
                    session=session,
                    uid=user.id,
                    phone=user.numero_whatsapp,
                    addon=addon,
                    idemp_key=f"addon:{user.id}:{addon_seed}",
                )
            else:
                final_bot_reply = addon
        elif (
            original_mode == SessionMode.COLLECTING_USABILITY.value
            and state.mode == SessionMode.COLLECTING_USABILITY.value
            and reply
        ):
            re_anchor = await self._survey_service.get_current_question_reply(session, state)
            if re_anchor:
                anchor_text = (
                    f"{reply}\n\n"
                    "Si quieres, retomamos la encuesta donde quedamos:\n\n"
                    f"{re_anchor.text}"
                )
                final_bot_reply = BotReply(text=anchor_text, content_type="text")
                anchor_seed = new_response_id or normalized.provider_message_id
                await schedule_separate_message(
                    session=session,
                    uid=user.id,
                    phone=user.numero_whatsapp,
                    addon=re_anchor,
                    idemp_key=f"reanchor:{user.id}:{anchor_seed}",
                )
            else:
                final_bot_reply = BotReply(text=reply, content_type="text")
        else:
            final_bot_reply = BotReply(text=reply, content_type="text")

        survey_was_interrupted = bool(
            original_mode == SessionMode.COLLECTING_USABILITY.value
            and state.mode == SessionMode.ACTIVE_CHAT.value
            and addon is None
        )
        if survey_was_interrupted and final_bot_reply.content_type == "text" and final_bot_reply.text:
            reminder = "Si quieres, luego retomamos la encuesta donde quedamos."
            low = (final_bot_reply.text or "").lower()
            if "retomamos la encuesta" not in low and "encuesta donde quedamos" not in low:
                final_bot_reply.text = f"{final_bot_reply.text}\n\n{reminder}"
        return final_bot_reply, survey_was_interrupted
