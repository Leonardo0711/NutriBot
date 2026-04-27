"""
Nutribot Backend - ConversationStateService
Encapsula la mutacion controlada de la entidad ConversationState.
"""
from datetime import timedelta

from domain.entities import ConversationState
from domain.value_objects import OnboardingStatus, OnboardingStep, SessionMode
from domain.utils import get_now_peru


class ConversationStateService:
    def apply_reset(self, state: ConversationState) -> None:
        """Reinicia el estado al punto de partida para el onboarding."""
        state.onboarding_status = OnboardingStatus.INVITED.value
        state.onboarding_step = OnboardingStep.INVITACION.value
        state.mode = SessionMode.ACTIVE_CHAT.value
        self.bump_version(state)

    def bump_version(self, state: ConversationState) -> None:
        """Incrementa la version para control optimista de concurrencia."""
        state.version += 1

    def set_turns_since_last_prompt(self, state: ConversationState, value: int) -> None:
        state.turns_since_last_prompt = max(0, int(value))

    def set_onboarding_in_progress(self, state: ConversationState, step: str) -> None:
        now = get_now_peru()
        state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
        state.onboarding_step = step
        state.onboarding_updated_at = now
        self.bump_version(state)

    def set_onboarding_invited(self, state: ConversationState) -> None:
        now = get_now_peru()
        state.onboarding_status = OnboardingStatus.INVITED.value
        state.onboarding_step = OnboardingStep.INVITACION.value
        state.onboarding_last_invited_at = now
        state.onboarding_updated_at = now
        self.bump_version(state)

    def set_onboarding_completed(self, state: ConversationState) -> None:
        now = get_now_peru()
        state.onboarding_status = OnboardingStatus.COMPLETED.value
        state.onboarding_step = None
        state.mode = SessionMode.ACTIVE_CHAT.value
        state.onboarding_updated_at = now
        self.bump_version(state)

    def set_onboarding_paused(self, state: ConversationState, days_until_retry: int = 3) -> None:
        now = get_now_peru()
        state.onboarding_status = OnboardingStatus.PAUSED.value
        state.onboarding_next_eligible_at = now + timedelta(days=days_until_retry)
        state.onboarding_updated_at = now
        self.bump_version(state)

    def set_onboarding_skipped(self, state: ConversationState, days_until_retry: int = 14) -> None:
        now = get_now_peru()
        state.onboarding_status = OnboardingStatus.SKIPPED.value
        state.onboarding_step = None
        state.onboarding_skip_count += 1
        state.onboarding_next_eligible_at = now + timedelta(days=days_until_retry)
        state.onboarding_updated_at = now
        self.bump_version(state)

    def schedule_next_onboarding_eligibility(self, state: ConversationState, days_until_retry: int) -> None:
        now = get_now_peru()
        state.onboarding_next_eligible_at = now + timedelta(days=days_until_retry)
        state.onboarding_updated_at = now
        self.bump_version(state)

    def update_interaction_details(
        self,
        state: ConversationState,
        provider_message_id: str,
        openai_response_id: str | None = None
    ) -> None:
        """Actualiza metadatos de interaccion tras un turno valido."""
        state.last_provider_message_id = provider_message_id
        if openai_response_id:
            state.last_openai_response_id = openai_response_id
        state.turns_since_last_prompt += 1
        self.bump_version(state)

    def update_meaningful_interaction_count(
        self,
        state: ConversationState,
        survey_was_interrupted: bool,
        projected_interactions_count: int
    ) -> None:
        """Actualiza el contador de interacciones significativas."""
        if state.mode == SessionMode.ACTIVE_CHAT.value:
            if survey_was_interrupted:
                state.meaningful_interactions_count = 0
            else:
                state.meaningful_interactions_count = projected_interactions_count

    def pause_survey_for_profile_maintenance(
        self,
        state: ConversationState,
        reason: str = "PROFILE_MAINTENANCE",
        additional_valid_turns: int = 5,
    ) -> None:
        """
        Pospone la encuesta de usabilidad porque el usuario está
        actualizando su perfil en medio de una conversación.
        """
        now = get_now_peru()
        state.survey_paused_reason = reason
        state.survey_next_eligible_count = (
            state.meaningful_interactions_count + additional_valid_turns
        )
        state.survey_updated_at = now
        self.bump_version(state)

    def can_offer_survey(self, state: ConversationState) -> bool:
        """
        Verifica si el estado permite ofrecer una encuesta ahora.
        Si hay un snooze activo y no se ha alcanzado el umbral de interacciones, NO.
        """
        if state.survey_next_eligible_count is not None:
            if state.meaningful_interactions_count < state.survey_next_eligible_count:
                return False
            # Umbral alcanzado → limpiar snooze
            state.survey_next_eligible_count = None
            state.survey_paused_reason = None
        return True

