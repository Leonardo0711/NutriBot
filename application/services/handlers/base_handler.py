"""
Nutribot Backend - Base Handler
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply

class BaseHandler(ABC):
    @abstractmethod
    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        """
        Procesa el turno basado en el TurnContext.
        Retorna la respuesta del bot y el ID de respuesta del LLM (si aplica).
        """
        pass
