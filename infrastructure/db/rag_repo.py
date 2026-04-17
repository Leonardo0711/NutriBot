"""
Nutribot Backend — RAG Repository
Busqueda semantica en fragmentos_rag usando pgvector.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory

logger = logging.getLogger(__name__)


class RagRepository:
    """Repositorio para Busqueda vectorial en fragmentos RAG."""

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

        # Sanitizar y normalizar embedding para un cast estable en PostgreSQL.
        cleaned_embedding: list[str] = []
        for val in query_embedding:
            try:
                num = float(val)
            except (TypeError, ValueError):
                logger.warning("RAG: embedding con valor no numerico, se descarta consulta")
                return []
            if not math.isfinite(num):
                logger.warning("RAG: embedding con valor no finito, se descarta consulta")
                return []
            cleaned_embedding.append(f"{num:.12g}")

        embedding_literal = "[" + ",".join(cleaned_embedding) + "]"

        factory = get_session_factory()
        try:
            async with factory() as session:
                result = await session.execute(
                    text("""
                        SELECT contenido,
                               embedding <=> CAST(:embedding AS vector) AS distance
                        FROM fragmentos_rag
                        WHERE embedding <=> CAST(:embedding AS vector) < :threshold
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                        LIMIT :lim
                    """),
                    {
                        "embedding": embedding_literal,
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
            logger.exception("Error en Busqueda RAG")
            return []

