"""
Nutribot Backend - Reset Handler
Maneja el intento de RESET del usuario.
"""
from typing import Optional, Tuple

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply
from application.services.handlers.base_handler import BaseHandler
from application.services.onboarding_service import OnboardingService
from application.services.conversation_state_service import ConversationStateService


class ResetHandler(BaseHandler):
    def __init__(
        self,
        onboarding_service: OnboardingService,
        state_service: ConversationStateService,
    ):
        self._onboarding_service = onboarding_service
        self._state_service = state_service

    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        # Borrado de la base de datos subyacente
        await self._onboarding_service._handle_system_reset(ctx.user.id, ctx.session)
        
        # Reinicio del estado a través del state_service
        self._state_service.apply_reset(ctx.state)
        
        bot_reply = BotReply(
            text="He borrado tus datos de perfil para empezar de cero cuando quieras. Quieres que empecemos ahora?",
            content_type="text",
        )
        return bot_reply, None

