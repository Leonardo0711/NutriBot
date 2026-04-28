# application/services/semantic_entity_resolver.py
"""
Nutribot Backend — SemanticEntityResolver
==========================================
Resuelve valores libres contra maestros semánticos.
Estrategia: caché → alias exacto → catálogo exacto → trigram → vector → revisión.

Este servicio NO decide intención ni operación. Solo mapea texto libre
a entidades canónicas de la BD.

Columnas y tablas usadas coinciden con el esquema real de 001_v3_schema.sql.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SemanticResolution:
    """Resultado de la resolución semántica de una entidad."""
    entity_type: str
    entity_code: str | None
    entity_label: str | None
    confidence: float
    strategy: str
    cache_hit: bool = False
    ambiguous: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)


class SemanticEntityResolver:
    """
    Resuelve valores libres del usuario contra las tablas semánticas:
    - semantic_resolution_cache
    - mae_alias_semantico
    - semantic_catalog (exacto, trigram, vector)

    NO duplica _resolve_master_id de ProfileExtractionService.
    Se enfoca en dar metadata rica (confianza, estrategia, candidatos)
    para que el extractor de intención pueda decidir si pedir aclaración.
    """

    ENTITY_TYPE_BY_FIELD = {
        "enfermedades": "ENFERMEDAD_CIE10",
        "restricciones_alimentarias": "RESTRICCION_ALIMENTARIA",
        "alergias": "RESTRICCION_ALIMENTARIA",
        "tipo_dieta": "PATRON_ALIMENTARIO",
        "objetivo_nutricional": "OBJETIVO_NUTRICIONAL",
        "distrito": "DISTRITO",
        "provincia": "PROVINCIA",
    }

    MIN_TRGM_SCORE = 0.78
    MIN_VECTOR_SCORE = 0.75
    MIN_MARGIN = 0.08
    SCOPE = "PROFILE_FIELD"

    def __init__(self, embeddings_adapter=None):
        self._embeddings = embeddings_adapter

    @staticmethod
    def normalize(value: str) -> str:
        """Normaliza texto para comparación: minúsculas, sin tildes, sin puntuación."""
        if not value:
            return ""
        s = value.lower().strip()
        s = "".join(
            c for c in unicodedata.normalize("NFKD", s)
            if not unicodedata.combining(c)
        )
        s = re.sub(r"[^a-z0-9ñ\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    async def resolve(
        self,
        session: AsyncSession,
        *,
        field_code: str,
        raw_value: str,
        usuario_id: int | None = None,
        incoming_message_id: int | None = None,
    ) -> SemanticResolution:
        """
        Resuelve un valor libre contra los maestros semánticos.
        Retorna metadata rica para que el caller decida qué hacer.
        """
        entity_type = self.ENTITY_TYPE_BY_FIELD.get(field_code)
        normalized = self.normalize(raw_value)
        started = perf_counter()

        if not entity_type or not normalized:
            return SemanticResolution(
                entity_type=entity_type or "UNKNOWN",
                entity_code=None,
                entity_label=None,
                confidence=0.0,
                strategy="UNSUPPORTED",
                ambiguous=True,
            )

        try:
            # 1. Caché
            async with session.begin_nested():
                cached = await self._try_cache(session, field_code, normalized)
            if cached:
                elapsed = int((perf_counter() - started) * 1000)
                await self._log_match(session, usuario_id, incoming_message_id,
                                      field_code, raw_value, normalized, cached, elapsed)
                return cached

            # 2. Alias exacto
            async with session.begin_nested():
                alias = await self._try_alias_exact(session, entity_type, normalized)
                if alias:
                    await self._save_cache(session, field_code, raw_value, normalized, entity_type, alias)
                    elapsed = int((perf_counter() - started) * 1000)
                    await self._log_match(session, usuario_id, incoming_message_id,
                                          field_code, raw_value, normalized, alias, elapsed)
                    return alias

            # 3. Catálogo exacto
            async with session.begin_nested():
                catalog_exact = await self._try_catalog_exact(session, entity_type, normalized)
            if catalog_exact:
                await self._save_cache(session, field_code, raw_value, normalized, entity_type, catalog_exact)
                elapsed = int((perf_counter() - started) * 1000)
                await self._log_match(session, usuario_id, incoming_message_id,
                                      field_code, raw_value, normalized, catalog_exact, elapsed)
                return catalog_exact

            # 4. Trigram (fuzzy)
            async with session.begin_nested():
                trgm = await self._try_trgm(session, entity_type, normalized)
                if trgm and not trgm.ambiguous and trgm.confidence >= self.MIN_TRGM_SCORE:
                    await self._save_cache(session, field_code, raw_value, normalized, entity_type, trgm)
                    elapsed = int((perf_counter() - started) * 1000)
                    await self._log_match(session, usuario_id, incoming_message_id,
                                          field_code, raw_value, normalized, trgm, elapsed)
                    return trgm

            # 5. Vector (pgvector)
            async with session.begin_nested():
                vector = await self._try_vector(session, entity_type, normalized)
                if vector and not vector.ambiguous and vector.confidence >= self.MIN_VECTOR_SCORE:
                    await self._save_cache(session, field_code, raw_value, normalized, entity_type, vector)
                    elapsed = int((perf_counter() - started) * 1000)
                    await self._log_match(session, usuario_id, incoming_message_id,
                                          field_code, raw_value, normalized, vector, elapsed)
                    return vector

            # 6. Sin match claro → marcar ambiguo y encolar revisión
            best = trgm or vector
            if best:
                async with session.begin_nested():
                    best.ambiguous = True
                    elapsed = int((perf_counter() - started) * 1000)
                    await self._enqueue_review(session, usuario_id, incoming_message_id,
                                               field_code, raw_value, normalized, best, "AMBIGUOUS_LOW_CONFIDENCE")
                    await self._log_match(session, usuario_id, incoming_message_id,
                                          field_code, raw_value, normalized, best, elapsed)
                    return best

            unresolved = SemanticResolution(
                entity_type=entity_type,
                entity_code=None,
                entity_label=None,
                confidence=0.0,
                strategy="NO_MATCH",
                ambiguous=True,
            )
            async with session.begin_nested():
                elapsed = int((perf_counter() - started) * 1000)
                await self._enqueue_review(session, usuario_id, incoming_message_id,
                                           field_code, raw_value, normalized, unresolved, "NO_MATCH")
            return unresolved

        except Exception:
            logger.exception("SemanticEntityResolver: error resolving field=%s value=%s", field_code, raw_value)
            return SemanticResolution(
                entity_type=entity_type,
                entity_code=None,
                entity_label=None,
                confidence=0.0,
                strategy="ERROR",
                ambiguous=True,
            )

    # ─── Cache ───

    async def _try_cache(self, session: AsyncSession, field_code: str, normalized: str) -> SemanticResolution | None:
        res = await session.execute(
            text("""
                SELECT entidad_tipo_resuelta, entidad_codigo_resuelto,
                       confidence, estrategia_usada
                FROM semantic_resolution_cache
                WHERE scope = :scope
                  AND field_code = :field_code
                  AND query_normalizada = :normalized
                ORDER BY actualizado_en DESC NULLS LAST
                LIMIT 1
            """),
            {"scope": self.SCOPE, "field_code": field_code, "normalized": normalized},
        )
        row = res.mappings().first()
        if not row:
            return None

        # Incrementar hit_count
        try:
            await session.execute(
                text("""
                    UPDATE semantic_resolution_cache
                    SET hit_count = hit_count + 1,
                        actualizado_en = TIMEZONE('America/Lima', NOW())
                    WHERE scope = :scope
                      AND field_code = :field_code
                      AND query_normalizada = :normalized
                """),
                {"scope": self.SCOPE, "field_code": field_code, "normalized": normalized},
            )
        except Exception:
            pass  # No bloquear por el hit counter

        return SemanticResolution(
            entity_type=row["entidad_tipo_resuelta"] or "",
            entity_code=row["entidad_codigo_resuelto"],
            entity_label=row["entidad_codigo_resuelto"],  # El label real se puede enriquecer luego
            confidence=float(row["confidence"] or 0.0),
            strategy=f"CACHE_{row['estrategia_usada']}" if row["estrategia_usada"] else "CACHE",
            cache_hit=True,
        )

    # ─── Alias Exacto ───

    async def _try_alias_exact(self, session: AsyncSession, entity_type: str, normalized: str) -> SemanticResolution | None:
        res = await session.execute(
            text("""
                SELECT entidad_tipo, entidad_codigo, alias_texto
                FROM mae_alias_semantico
                WHERE entidad_tipo = :entity_type
                  AND activo = TRUE
                  AND alias_normalizado = :normalized
                ORDER BY prioridad ASC, es_canonico DESC
                LIMIT 1
            """),
            {"entity_type": entity_type, "normalized": normalized},
        )
        row = res.mappings().first()
        if not row:
            return None

        return SemanticResolution(
            entity_type=row["entidad_tipo"],
            entity_code=row["entidad_codigo"],
            entity_label=row["alias_texto"],
            confidence=1.0,
            strategy="ALIAS_EXACT",
        )

    # ─── Catálogo Exacto ───

    async def _try_catalog_exact(self, session: AsyncSession, entity_type: str, normalized: str) -> SemanticResolution | None:
        res = await session.execute(
            text("""
                SELECT entidad_tipo, entidad_codigo, texto_busqueda
                FROM semantic_catalog
                WHERE entidad_tipo = :entity_type
                  AND activo = TRUE
                  AND texto_normalizado = :normalized
                ORDER BY peso_lexico DESC
                LIMIT 1
            """),
            {"entity_type": entity_type, "normalized": normalized},
        )
        row = res.mappings().first()
        if not row:
            return None

        return SemanticResolution(
            entity_type=row["entidad_tipo"],
            entity_code=row["entidad_codigo"],
            entity_label=row["texto_busqueda"],
            confidence=0.97,
            strategy="CATALOG_EXACT",
        )

    # ─── Trigram ───

    async def _try_trgm(self, session: AsyncSession, entity_type: str, normalized: str) -> SemanticResolution | None:
        try:
            res = await session.execute(
                text("""
                    SELECT
                        entidad_tipo,
                        entidad_codigo,
                        texto_busqueda,
                        similarity(texto_normalizado, :normalized) AS score
                    FROM semantic_catalog
                    WHERE entidad_tipo = :entity_type
                      AND activo = TRUE
                    ORDER BY similarity(texto_normalizado, :normalized) DESC
                    LIMIT 3
                """),
                {"entity_type": entity_type, "normalized": normalized},
            )
        except Exception:
            # pg_trgm puede no estar disponible
            logger.debug("Trigram search failed for entity_type=%s", entity_type)
            return None

        rows = list(res.mappings())
        if not rows:
            return None

        top = rows[0]
        score = float(top["score"])
        if score < 0.3:
            return None  # Ruido total

        second_score = float(rows[1]["score"]) if len(rows) > 1 else 0.0
        ambiguous = (score - second_score) < self.MIN_MARGIN

        return SemanticResolution(
            entity_type=top["entidad_tipo"],
            entity_code=top["entidad_codigo"],
            entity_label=top["texto_busqueda"],
            confidence=score,
            strategy="TRGM",
            ambiguous=ambiguous,
            candidates=[
                {
                    "code": r["entidad_codigo"],
                    "label": r["texto_busqueda"],
                    "score": float(r["score"]),
                    "strategy": "TRGM",
                }
                for r in rows
            ],
        )

    # ─── Vector ───

    async def _try_vector(self, session: AsyncSession, entity_type: str, normalized: str) -> SemanticResolution | None:
        if not self._embeddings:
            return None

        try:
            embedding = await self._embeddings.embed(normalized)
            if not embedding:
                return None
        except Exception:
            logger.debug("Embedding generation failed")
            return None

        try:
            res = await session.execute(
                text("""
                    SELECT
                        entidad_tipo,
                        entidad_codigo,
                        texto_busqueda,
                        1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM semantic_catalog
                    WHERE entidad_tipo = :entity_type
                      AND activo = TRUE
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT 3
                """),
                {"entity_type": entity_type, "embedding": str(embedding)},
            )
        except Exception:
            logger.debug("Vector search failed for entity_type=%s", entity_type)
            return None

        rows = list(res.mappings())
        if not rows:
            return None

        top = rows[0]
        score = float(top["score"])
        if score < 0.5:
            return None

        second_score = float(rows[1]["score"]) if len(rows) > 1 else 0.0
        ambiguous = (score - second_score) < self.MIN_MARGIN

        return SemanticResolution(
            entity_type=top["entidad_tipo"],
            entity_code=top["entidad_codigo"],
            entity_label=top["texto_busqueda"],
            confidence=score,
            strategy="VECTOR",
            ambiguous=ambiguous,
            candidates=[
                {
                    "code": r["entidad_codigo"],
                    "label": r["texto_busqueda"],
                    "score": float(r["score"]),
                    "strategy": "VECTOR",
                }
                for r in rows
            ],
        )

    # ─── Persistencia ───

    async def _save_cache(
        self,
        session: AsyncSession,
        field_code: str,
        raw_text: str,
        normalized: str,
        entity_type: str | None,
        result: SemanticResolution,
    ) -> None:
        if not result.entity_code:
            return
        try:
            await session.execute(
                text("""
                    INSERT INTO semantic_resolution_cache
                    (scope, field_code, query_texto, query_normalizada,
                     entidad_tipo_resuelta, entidad_codigo_resuelto,
                     estrategia_usada, confidence, hit_count,
                     top_candidates_json,
                     creado_en, actualizado_en)
                    VALUES
                    (:scope, :field_code, :raw_text, :normalized,
                     :entity_type, :entity_code,
                     :strategy, :confidence, 0,
                     CAST(:candidates AS jsonb),
                     TIMEZONE('America/Lima', NOW()),
                     TIMEZONE('America/Lima', NOW()))
                    ON CONFLICT (scope, field_code, query_normalizada)
                    DO UPDATE SET
                        entidad_tipo_resuelta = EXCLUDED.entidad_tipo_resuelta,
                        entidad_codigo_resuelto = EXCLUDED.entidad_codigo_resuelto,
                        estrategia_usada = EXCLUDED.estrategia_usada,
                        confidence = EXCLUDED.confidence,
                        top_candidates_json = EXCLUDED.top_candidates_json,
                        actualizado_en = TIMEZONE('America/Lima', NOW())
                """),
                {
                    "scope": self.SCOPE,
                    "field_code": field_code,
                    "raw_text": raw_text,
                    "normalized": normalized,
                    "entity_type": entity_type or result.entity_type,
                    "entity_code": result.entity_code,
                    "strategy": result.strategy.replace("CACHE_", ""),
                    "confidence": result.confidence,
                    "candidates": json.dumps(result.candidates, ensure_ascii=False) if result.candidates else "[]",
                },
            )
        except Exception:
            logger.exception("SemanticEntityResolver: failed to save cache")

    async def _log_match(
        self,
        session: AsyncSession,
        usuario_id: int | None,
        incoming_message_id: int | None,
        field_code: str,
        raw_text: str,
        normalized: str,
        result: SemanticResolution,
        latency_ms: int = 0,
    ) -> None:
        try:
            await session.execute(
                text("""
                    INSERT INTO semantic_match_log
                    (usuario_id, incoming_message_id, scope, field_code,
                     query_texto, query_normalizada, estrategia_usada,
                     exact_match, trigram_score, vector_score,
                     confidence_final, entidad_tipo_resuelta, entidad_codigo_resuelto,
                     latency_ms, creado_en)
                    VALUES
                    (:uid, :mid, :scope, :field_code,
                     :raw_text, :normalized, :strategy,
                     :exact, :trgm, :vec,
                     :confidence, :entity_type, :entity_code,
                     :latency, TIMEZONE('America/Lima', NOW()))
                """),
                {
                    "uid": usuario_id,
                    "mid": incoming_message_id,
                    "scope": self.SCOPE,
                    "field_code": field_code,
                    "raw_text": raw_text,
                    "normalized": normalized,
                    "strategy": result.strategy,
                    "exact": result.strategy in ("ALIAS_EXACT", "CATALOG_EXACT", "EXACT"),
                    "trgm": result.confidence if "TRGM" in result.strategy else None,
                    "vec": result.confidence if "VECTOR" in result.strategy else None,
                    "confidence": result.confidence,
                    "entity_type": result.entity_type,
                    "entity_code": result.entity_code,
                    "latency": latency_ms,
                },
            )
        except Exception:
            logger.debug("SemanticEntityResolver: failed to log match")

    async def _enqueue_review(
        self,
        session: AsyncSession,
        usuario_id: int | None,
        incoming_message_id: int | None,
        field_code: str,
        raw_text: str,
        normalized: str,
        result: SemanticResolution,
        reason: str,
    ) -> None:
        try:
            await session.execute(
                text("""
                    INSERT INTO semantic_review_queue
                    (usuario_id, incoming_message_id, scope, field_code,
                     query_texto, query_normalizada, top_candidates_json,
                     razon, estado, creado_en)
                    VALUES
                    (:uid, :mid, :scope, :field_code,
                     :raw_text, :normalized, CAST(:candidates AS jsonb),
                     :reason, 'PENDING', TIMEZONE('America/Lima', NOW()))
                """),
                {
                    "uid": usuario_id,
                    "mid": incoming_message_id,
                    "scope": self.SCOPE,
                    "field_code": field_code,
                    "raw_text": raw_text,
                    "normalized": normalized,
                    "candidates": json.dumps(result.candidates, ensure_ascii=False) if result.candidates else "[]",
                    "reason": reason,
                },
            )
        except Exception:
            logger.debug("SemanticEntityResolver: failed to enqueue review")
