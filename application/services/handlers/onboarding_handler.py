"""
Nutribot Backend - Onboarding Handler
Maneja los turnos cuando el usuario está en el flujo de Onboarding.
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
        # Intentamos avanzar explícitamente en el flujo de onboarding
        reply = await self._onboarding_service.advance_flow(
            user_text=ctx.normalized.text,
            state=ctx.state,
            session=ctx.session,
            treat_ninguna_as_missing=False,
            pre_extracted_data=None,
            history=ctx.history,
        )
        
        if reply is not None:
            # El Onboarding manejó la respuesta (ej. guardó un dato y pide el siguiente)
            # Registramos que la intercepción de onboarding ocurrió
            ctx.onboarding_interception_happened = True
            return reply, None
            
        # Si el OnboardingService retornó None, significa que no entendió la respuesta
        # o que el usuario dijo algo fuera de contexto. Delegamos al flujo general.
        ctx.onboarding_interception_happened = False
        return await self._fallback_handler.handle(ctx)

