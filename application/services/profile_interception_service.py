"""
Nutribot Backend - Profile Interception Service
"""
from __future__ import annotations

from typing import Optional

from application.services.conversation_state_service import ConversationStateService
from application.services.onboarding_service import OnboardingService
from application.services.profile_context_service import ProfileContextService
from domain.entities import ConversationState
from domain.profile_snapshot import ProfileSnapshot
from domain.value_objects import ONBOARDING_PHASE_2, OnboardingStatus


class ProfileInterceptionService:
    def __init__(
        self,
        onboarding_service: OnboardingService,
        profile_context: ProfileContextService,
        state_service: ConversationStateService,
    ):
        self._onboarding_service = onboarding_service
        self._profile_context = profile_context
        self._state_service = state_service

    @staticmethod
    def _ask_single_field(step_name: str) -> str:
        clean = (step_name or "un dato de tu perfil").strip()
        return f"Perfecto. Vamos paso a paso 😊\n\nPrimero, me compartes {clean}?"

    async def maybe_start_personalization_flow(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        summary: str,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        is_requesting_personalization: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], bool]:
        if onboarding_interception_happened or not is_requesting_personalization:
            return reply, onboarding_interception_happened

        next_step = await self._onboarding_service._find_next_missing_step(
            session,
            user_id,
            ignore_skips=True,
            treat_ninguna_as_missing=False,
            phase=[s for s in ONBOARDING_PHASE_2] if state.onboarding_status == OnboardingStatus.COMPLETED.value else None,
        )
        if next_step:
            step_label = self._profile_context.human_step_label(next_step)
            reply = self._ask_single_field(step_label)
            self._state_service.set_onboarding_in_progress(state, next_step)
            return reply, True

        if not is_asking_for_recommendation:
            reply = (
                "Ya tengo tu perfil completo 😊\n\n"
                f"{summary}\n\n"
                "Si luego quieres corregir algun dato, dimelo directo y lo actualizo."
            )
        return reply, onboarding_interception_happened

    async def maybe_intercept_for_missing_profile(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        is_short_greeting: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], bool]:
        should_check = (
            not onboarding_interception_happened
            and state.onboarding_status != OnboardingStatus.COMPLETED.value
            and (is_short_greeting or is_asking_for_recommendation)
        )
        if not should_check:
            return reply, onboarding_interception_happened

        if is_asking_for_recommendation:
            missing_essential = self._profile_context.missing_essential_fields(snapshot)
            if not missing_essential:
                return reply, onboarding_interception_happened

            missing_step = await self._onboarding_service._find_next_missing_step(session, user_id, phase=None)
            if not missing_step:
                return reply, onboarding_interception_happened

            step_name = self._profile_context.human_step_label(missing_step)
            reply = (
                "Claro, te ayudo con eso 😊\n\n"
                f"Para afinar la recomendacion, primero me compartes {step_name}?"
            )
            self._state_service.set_onboarding_in_progress(state, missing_step)
            return reply, True

        if is_short_greeting:
            if state.onboarding_status == OnboardingStatus.NOT_STARTED.value:
                reply = (
                    "Hola 😊 Soy NutriBot, tu asistente de nutricion de EsSalud.\n\n"
                    "Estoy aqui para ayudarte con orientacion y recomendaciones de alimentacion saludable.\n\n"
                    "Preguntame lo que necesites, estoy para ayudarte 🍎"
                )
            else:
                reply = (
                    "Hola de nuevo 😊\n\n"
                    "Que gusto verte por aqui. ¿En que te puedo ayudar hoy? 🍏"
                )
            self._state_service.set_onboarding_invited(state)
            return reply, True

        return reply, onboarding_interception_happened

    async def maybe_suggest_phase2_field(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        reply: Optional[str],
    ) -> Optional[str]:
        """Si el onboarding basico ya esta completo, sugiere personalizacion extra de forma suave."""
        if state.onboarding_status != OnboardingStatus.COMPLETED.value:
            return reply
        if not reply:
            return reply
        if state.awaiting_question_code:
            return reply
        normalized_reply = (reply or "").lower()
        if "tip nutribot" in normalized_reply:
            return reply
        if state.turns_since_last_prompt < 22:
            return reply

        next_phase2 = await self._onboarding_service._find_next_missing_step(
            session,
            user_id,
            phase=[s for s in ONBOARDING_PHASE_2],
            start_from_idx=0,
        )
        if not next_phase2:
            return reply

        self._state_service.set_turns_since_last_prompt(state, 0)

        suggestion = (
            "\n\nTip NutriBot: si quieres personalizar mas tu perfil, escribe "
            "*quiero actualizar mi perfil nutricional*."
        )
        return reply + suggestion
