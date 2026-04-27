"""
Nutribot Backend - Onboarding Handler
Maneja los turnos cuando el usuario esta en el flujo de Onboarding.
"""
from typing import Optional, Tuple

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply
from application.services.handlers.base_handler import BaseHandler
from application.services.onboarding_service import OnboardingService


class OnboardingHandler(BaseHandler):
    def __init__(
        self,
        onboarding_service: OnboardingService,
        fallback_handler: BaseHandler,
    ):
        self._onboarding_service = onboarding_service
        self._fallback_handler = fallback_handler

    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        reply = await self._onboarding_service.advance_flow(
            user_text=ctx.normalized.text,
            state=ctx.state,
            session=ctx.session,
            treat_ninguna_as_missing=False,
            pre_extracted_intent=ctx.profile_intent,
            history=ctx.history,
        )

        if reply is not None:
            ctx.onboarding_interception_happened = True
            if isinstance(reply, str):
                reply = BotReply(text=reply, content_type="text")
            return reply, None

        ctx.onboarding_interception_happened = False
        return await self._fallback_handler.handle(ctx)

