"""
Nutribot Backend — OpenAI Embeddings Adapter
Genera embeddings para búsquedas semánticas RAG.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsAdapter:
    """Genera embeddings usando la API de OpenAI."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.openai_embedding_model

    async def embed(self, text: str) -> list[float]:
        """Genera un embedding vector para el texto dado."""
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception:
            logger.exception("Error generando embedding")
            return []
