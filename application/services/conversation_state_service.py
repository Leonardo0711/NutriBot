"""
Nutribot Backend - ConversationStateService
Encapsula la mutación controlada de la entidad ConversationState.
"""
from domain.entities import ConversationState
from domain.value_objects import OnboardingStatus, OnboardingStep, SessionMode


class ConversationStateService:
    def apply_reset(self, state: ConversationState) -> None:
        """Reinicia el estado al punto de partida para el onboarding."""
        state.onboarding_status = OnboardingStatus.INVITED.value
        state.onboarding_step = OnboardingStep.INVITACION.value
        state.mode = SessionMode.ACTIVE_CHAT.value
        self.bump_version(state)

    def bump_version(self, state: ConversationState) -> None:
        """Incrementa la versión para control optimista de concurrencia."""
        state.version += 1

    def update_interaction_details(
        self,
        state: ConversationState,
        provider_message_id: str,
        openai_response_id: str | None = None
    ) -> None:
        """Actualiza metadatos de interacción tras un turno válido."""
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

