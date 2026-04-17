"""
Nutribot Backend — Responses API Adapter (LLM)
Genera respuestas usando OpenAI Responses API con encadenamiento por previous_response_id.
"""
from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from config import get_settings
from domain.entities import ConversationState, NormalizedMessage
from domain.ports import LLMService

logger = logging.getLogger(__name__)


class OpenAIResponsesAdapter(LLMService):
    """Adaptador para la Responses API de OpenAI."""

    def __init__(
        self,
        system_instructions: str,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.openai_model
        self._instructions = system_instructions

    async def generate_reply(
        self,
        state: ConversationState,
        normalized: NormalizedMessage,
        instructions: str,
        rag_context: Optional[str] = None,
        profile_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Genera una respuesta del LLM.

        - CRÍTICO: Reenvía el bloque completo de instructions maestras en CADA solicitud.
          previous_response_id NO hereda las directivas del sistema de turnos anteriores.
        - Encadena turnos usando previous_response_id para mantener contexto.
        - Inyecta contexto RAG y perfil como parte del input si corresponde.

        Returns:
            (reply_text, new_response_id)
        """
        try:
            has_prev_response = bool(state.last_openai_response_id)
            # Si ya hay previous_response_id, no reenviamos historial textual para evitar duplicar tokens.
            include_history = not has_prev_response
            user_input = self._build_user_input(
                normalized,
                rag_context,
                profile_context,
                history,
                include_history=include_history,
            )

            # Construir los parámetros de la solicitud
            params = {
                "model": self._model,
                "instructions": instructions or self._instructions,
                "input": user_input,
            }
            if max_tokens and max_tokens > 0:
                params["max_output_tokens"] = int(max_tokens)

            # Encadenar turno previo si existe
            if has_prev_response:
                params["previous_response_id"] = state.last_openai_response_id

            response = await self._client.responses.create(**params)

            reply_text = response.output_text or ""
            new_response_id = response.id

            logger.debug(
                "LLM reply (%d chars, response_id=%s): %s...",
                len(reply_text),
                new_response_id,
                reply_text[:80],
            )

            return reply_text.strip(), new_response_id

        except Exception:
            logger.exception("Error generando reply con Responses API")
            raise

    def _build_user_input(
        self,
        normalized: NormalizedMessage,
        rag_context: Optional[str],
        profile_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
        include_history: bool = True,
    ) -> list[dict]:
        """
        Construye el array de input para la Responses API.
        Soporta texto plano, imágenes (Vision), RAG y perfil del usuario.
        """
        parts: list[dict] = []

        # Historial reciente (Conciencia Total)
        if include_history and history:
            parts.append({
                "role": "user",
                "content": "[HISTORIAL RECIENTE PARA CONTEXTO]\n" + "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
            })

        # Perfil del usuario (SIEMPRE visible para el LLM)
        if profile_context:
            parts.append({
                "role": "user",
                "content": profile_context,
            })

        # Contexto RAG si hay
        if rag_context:
            parts.append({
                "role": "user",
                "content": f"[Contexto nutricional relevante]\n{rag_context}",
            })

        # Mensaje principal del usuario
        if normalized.image_base64:
            # Vision: texto + imagen
            content_parts = []
            if normalized.text:
                content_parts.append({"type": "input_text", "text": normalized.text})
            image_mimetype = normalized.image_mimetype or "image/jpeg"
            content_parts.append({
                "type": "input_image",
                "image_url": f"data:{image_mimetype};base64,{normalized.image_base64}",
            })
            parts.append({"role": "user", "content": content_parts})
        else:
            # Solo texto (original o transcripción STT)
            parts.append({"role": "user", "content": normalized.text})

        return parts
