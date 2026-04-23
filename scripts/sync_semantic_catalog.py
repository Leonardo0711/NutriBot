"""
Nutribot Backend - Sync semantic_catalog from master tables + aliases.

Usage:
    python scripts/sync_semantic_catalog.py

What this does:
1) Upsert canonical searchable terms from nutrition master tables.
2) Upsert alias searchable terms from mae_alias_semantico.
3) Queue missing embeddings into embedding_jobs (without touching campanias).
"""
from __future__ import annotations

import asyncio
import logging
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text

from config import get_settings
from infrastructure.db.connection import get_session_factory, dispose_engine
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MasterSource:
    table: str
    entity_type: str
    code_column: str
    name_column: str
    extra_where: str = "TRUE"


SOURCES: tuple[MasterSource, ...] = (
    MasterSource("mae_enfermedad_cie10", "ENFERMEDAD_CIE10", "codigo_cie10", "nombre"),
    MasterSource("mae_grupo_nutricional", "GRUPO_NUTRICIONAL", "codigo", "nombre"),
    MasterSource("mae_patron_alimentario", "PATRON_ALIMENTARIO", "codigo", "nombre"),
    MasterSource("mae_dieta_terapeutica", "DIETA_TERAPEUTICA", "codigo", "nombre"),
    MasterSource("mae_textura_dieta", "TEXTURA_DIETA", "codigo", "nombre"),
    MasterSource("mae_restriccion_alimentaria", "RESTRICCION_ALIMENTARIA", "codigo", "nombre"),
    MasterSource("mae_objetivo_nutricional", "OBJETIVO_NUTRICIONAL", "codigo", "nombre"),
    MasterSource("mae_distrito", "DISTRITO", "ubigeo", "nombre"),
)


def normalize_text(value: str) -> str:
    txt = unicodedata.normalize("NFKD", str(value or ""))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.lower().strip()
    txt = " ".join(txt.split())
    return txt[:255]


def chunked(items: list[dict], size: int = 1000) -> Iterable[list[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def _upsert_rows(session, rows: list[dict]) -> int:
    if not rows:
        return 0
    stmt = text(
        """
        INSERT INTO semantic_catalog (
            entidad_tipo, entidad_codigo, alias_semantico_id,
            nombre_canonico, texto_busqueda, texto_normalizado,
            peso_lexico, peso_semantico, activo, generado_en, actualizado_en
        )
        VALUES (
            :entidad_tipo, :entidad_codigo, :alias_semantico_id,
            :nombre_canonico, :texto_busqueda, :texto_normalizado,
            1.0, 1.0, TRUE, :now, :now
        )
        ON CONFLICT (entidad_tipo, entidad_codigo, texto_normalizado)
        DO UPDATE SET
            alias_semantico_id = COALESCE(EXCLUDED.alias_semantico_id, semantic_catalog.alias_semantico_id),
            nombre_canonico = EXCLUDED.nombre_canonico,
            texto_busqueda = EXCLUDED.texto_busqueda,
            activo = TRUE,
            actualizado_en = EXCLUDED.actualizado_en
        """
    )
    now = get_now_peru()
    total = 0
    for batch in chunked(rows, size=1000):
        payload = []
        for row in batch:
            payload.append(
                {
                    "entidad_tipo": row["entidad_tipo"],
                    "entidad_codigo": row["entidad_codigo"],
                    "alias_semantico_id": row.get("alias_semantico_id"),
                    "nombre_canonico": row["nombre_canonico"],
                    "texto_busqueda": row["texto_busqueda"],
                    "texto_normalizado": row["texto_normalizado"],
                    "now": now,
                }
            )
        await session.execute(stmt, payload)
        total += len(payload)
    return total


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()
    logger.info("Sync semantic_catalog started (db=%s)", settings.database_url.split("@")[-1])

    session_factory = get_session_factory()
    canonical_index: dict[str, dict[str, str]] = {}
    inserted_canonical = 0
    inserted_aliases = 0

    async with session_factory() as session:
        async with session.begin():
            # 1) Canonical rows from master tables.
            for src in SOURCES:
                rows = (
                    await session.execute(
                        text(
                            f"""
                            SELECT {src.code_column} AS codigo, {src.name_column} AS nombre
                            FROM {src.table}
                            WHERE activo = TRUE
                              AND {src.extra_where}
                            """
                        )
                    )
                ).mappings().all()

                canonical_map: dict[str, str] = {}
                upserts: list[dict] = []
                for row in rows:
                    code = str(row.get("codigo") or "").strip()
                    name = str(row.get("nombre") or "").strip()
                    if not code or not name:
                        continue
                    canonical_map[code] = name
                    upserts.append(
                        {
                            "entidad_tipo": src.entity_type,
                            "entidad_codigo": code[:60],
                            "alias_semantico_id": None,
                            "nombre_canonico": name[:255],
                            "texto_busqueda": name[:255],
                            "texto_normalizado": normalize_text(name),
                        }
                    )
                canonical_index[src.entity_type] = canonical_map
                inserted_canonical += await _upsert_rows(session, upserts)
                logger.info("%s canonical rows prepared=%d", src.entity_type, len(upserts))

            # 2) Alias rows from mae_alias_semantico.
            alias_rows = (
                await session.execute(
                    text(
                        """
                        SELECT id, entidad_tipo, entidad_codigo, alias_texto, alias_normalizado
                        FROM mae_alias_semantico
                        WHERE activo = TRUE
                        """
                    )
                )
            ).mappings().all()

            alias_upserts: list[dict] = []
            for row in alias_rows:
                entity_type = str(row.get("entidad_tipo") or "").strip()
                entity_code = str(row.get("entidad_codigo") or "").strip()
                alias_text = str(row.get("alias_texto") or "").strip()
                alias_norm = str(row.get("alias_normalizado") or "").strip()
                if not entity_type or not entity_code or not alias_text:
                    continue
                canonical_name = canonical_index.get(entity_type, {}).get(entity_code)
                if not canonical_name:
                    canonical_name = alias_text
                alias_upserts.append(
                    {
                        "entidad_tipo": entity_type[:30],
                        "entidad_codigo": entity_code[:60],
                        "alias_semantico_id": int(row.get("id")),
                        "nombre_canonico": canonical_name[:255],
                        "texto_busqueda": alias_text[:255],
                        "texto_normalizado": normalize_text(alias_norm or alias_text),
                    }
                )
            inserted_aliases += await _upsert_rows(session, alias_upserts)
            logger.info("Alias rows prepared=%d", len(alias_upserts))

            # 3) Queue pending embedding jobs for semantic_catalog.
            now = get_now_peru()
            enqueue_result = await session.execute(
                text(
                    """
                    INSERT INTO embedding_jobs (
                        source_table, source_id, texto_fuente, modelo,
                        estado, retry_count, error_detail, creado_en, actualizado_en, procesado_en
                    )
                    SELECT
                        'semantic_catalog', sc.id, sc.texto_busqueda, :model,
                        'PENDING', 0, NULL, :now, :now, NULL
                    FROM semantic_catalog sc
                    WHERE sc.activo = TRUE
                      AND sc.embedding IS NULL
                    ON CONFLICT (source_table, source_id, modelo)
                    DO UPDATE SET
                        texto_fuente = EXCLUDED.texto_fuente,
                        estado = 'PENDING',
                        error_detail = NULL,
                        actualizado_en = EXCLUDED.actualizado_en,
                        procesado_en = NULL
                    """
                ),
                {"model": settings.openai_embedding_model[:100], "now": now},
            )

    logger.info(
        "Sync semantic_catalog done: canonical_upserts=%d alias_upserts=%d embedding_jobs_touched=%s",
        inserted_canonical,
        inserted_aliases,
        enqueue_result.rowcount,
    )

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
