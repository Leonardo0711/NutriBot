"""
Nutribot Backend — RAG Repository
Búsqueda semántica en fragmentos_rag usando pgvector.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)


class RagRepository:
    """Repositorio para búsqueda vectorial en fragmentos RAG."""

    async def search(
        self,
        query_embedding: list[float],
        threshold: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[str]:
        """
        Busca fragmentos relevantes usando cosine distance con threshold.
        Retorna lista de contenidos textuales ordenados por relevancia.
        """
        settings = get_settings()
        _threshold = threshold or settings.rag_threshold
        _limit = limit or settings.rag_limit

        if not query_embedding:
            return []

        factory = get_session_factory()
        try:
            async with factory() as session:
                # pgvector cosine distance operator: <=>
                # We must embed the vector literal directly since asyncpg
                # can't handle ::vector cast on bind parameters
                embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"
                result = await session.execute(
                    text(f"""
                        SELECT contenido,
                               embedding <=> '{embedding_literal}'::vector AS distance
                        FROM fragmentos_rag
                        WHERE embedding <=> '{embedding_literal}'::vector < :threshold
                        ORDER BY embedding <=> '{embedding_literal}'::vector
                        LIMIT :lim
                    """),
                    {
                        "threshold": _threshold,
                        "lim": _limit,
                    },
                )
                rows = result.fetchall()

                if rows:
                    logger.debug(
                        "RAG: %d fragmentos encontrados (mejor distancia: %.3f)",
                        len(rows),
                        rows[0].distance,
                    )
                return [row.contenido for row in rows]

        except Exception:
            logger.exception("Error en búsqueda RAG")
            return []
