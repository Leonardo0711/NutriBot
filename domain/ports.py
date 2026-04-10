"""
Nutribot Backend — Domain Ports (Interfaces Abstractas)
Definen contratos que la capa de infraestructura debe implementar.
"""
from __future__ import annotations

import abc
from typing import Optional

from .entities import (
    User,
    ConversationState,
    NormalizedMessage,
    IncomingWebhookMessage,
    ProfileExtraction,
)


class UserRepository(abc.ABC):
    """Puerto: persistencia de usuarios."""

    @abc.abstractmethod
    async def get_or_create(self, phone: str) -> User:
        """Obtiene o crea un usuario por número de WhatsApp.
        Garantiza que siempre exista su conversation_state asociado."""
        ...


class ConversationRepository(abc.ABC):
    """Puerto: persistencia de estado conversacional."""

    @abc.abstractmethod
    async def get_state_no_lock(self, usuario_id: int) -> ConversationState:
        """Lee el estado sin lock (para snapshot pre-LLM)."""
        ...

    @abc.abstractmethod
    async def get_state_for_update(self, session, usuario_id: int) -> ConversationState:
        """Lee el estado con SELECT ... FOR UPDATE (dentro de transacción)."""
        ...

    @abc.abstractmethod
    async def save_state(self, session, state: ConversationState) -> None:
        """Persiste el estado actualizado (dentro de transacción existente)."""
        ...


class MediaService(abc.ABC):
    """Puerto: normalización de medios (STT, Vision, descarga)."""

    @abc.abstractmethod
    async def normalize(self, msg: IncomingWebhookMessage) -> NormalizedMessage:
        """Convierte un mensaje crudo en texto normalizado.
        - TEXT: pasa directo.
        - AUDIO/PTT: descarga ogg, convierte a mp3, envía a STT.
        - IMAGE: descarga, codifica base64 para Vision.
        """
        ...


class LLMService(abc.ABC):
    """Puerto: generación de respuesta vía LLM."""

    @abc.abstractmethod
    async def generate_reply(
        self,
        state: ConversationState,
        normalized: NormalizedMessage,
        instructions: str,
        rag_context: Optional[str] = None,
        profile_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> tuple[str, Optional[str]]:
        """Genera respuesta. Retorna (reply_text, new_response_id)."""
        ...


class TTSService(abc.ABC):
    """Puerto: generación de audio TTS."""

    @abc.abstractmethod
    async def generate_audio_base64(self, text: str) -> str:
        """Genera audio en memoria y retorna string base64.
        Sin persistencia local — stateless."""
        ...


class EvolutionClient(abc.ABC):
    """Puerto: envío de mensajes via Evolution API."""

    @abc.abstractmethod
    async def send_text(self, phone: str, text: str) -> bool:
        """Envía mensaje de texto. Retorna True si fue exitoso."""
        ...

    @abc.abstractmethod
    async def send_audio_base64(self, phone: str, audio_base64: str) -> bool:
        """Envía audio como base64. Retorna True si fue exitoso."""
        ...

    @abc.abstractmethod
    async def send_presence(self, phone: str, presence: str = "composing") -> bool:
        """Envía señal de presencia (typing/composing)."""
        ...
