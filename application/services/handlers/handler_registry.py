"""
Nutribot Backend - Handler Registry
Resuelve el handler correcto (Estrategia) dependiendo de la intención y estado.
"""
from domain.turn_context import TurnContext
from domain.router import Intent
from domain.value_objects import OnboardingStatus
from application.services.handlers.base_handler import BaseHandler


class HandlerRegistry:
    def __init__(
        self,
        reset_handler: BaseHandler,
        onboarding_handler: BaseHandler,
        profile_update_handler: BaseHandler,
        generic_chat_handler: BaseHandler,
    ):
        self._reset_handler = reset_handler
        self._onboarding_handler = onboarding_handler
        self._profile_update_handler = profile_update_handler
        self._generic_chat_handler = generic_chat_handler

    def resolve(self, ctx: TurnContext) -> BaseHandler:
        """
        Determina qué handler debe iniciar el procesamiento del turno.
        """
        if ctx.route.intent == Intent.RESET:
            return self._reset_handler

        # Priorizar flujo de onboarding si está activo o invitado
        if ctx.state.onboarding_status in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            return self._onboarding_handler

        # ── Intent extractor: comprensión real supera al router ──
        # Si el extractor detectó actualización de perfil con confianza alta,
        # priorizar el ProfileUpdateHandler incluso si el router dijo AMBIGUOUS o NUTRITION_QUERY.
        if (
            ctx.profile_intent
            and ctx.profile_intent.is_profile_update
            and ctx.profile_intent.is_confident
        ):
            return self._profile_update_handler

        # Si parece una actualización de datos perfil desde el chat activo (legacy router hint)
        if ctx.looks_like_profile_update:
            return self._profile_update_handler

        # Fallback de todo lo demás
        return self._generic_chat_handler


