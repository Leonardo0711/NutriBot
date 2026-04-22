"""
Nutribot Backend - ProfileReadService
Read-model projection from normalized V3 profile tables.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.profile_snapshot import ProfileSnapshot


class ProfileReadService:
    """Builds profile snapshots and compatibility projections from normalized tables."""

    _PROFILE_PROJECTION_SQL = text(
        """
        WITH base_user AS (
            SELECT CAST(:uid AS bigint) AS usuario_id
        ),
        profile_base AS (
            SELECT *
            FROM perfil_nutricional
            WHERE usuario_id = :uid
            LIMIT 1
        ),
        latest_measure AS (
            SELECT DISTINCT ON (m.perfil_nutricional_id, m.tipo_medicion)
                m.perfil_nutricional_id,
                m.tipo_medicion,
                m.valor_decimal
            FROM perfil_nutricional_medicion m
            JOIN profile_base pb ON pb.id = m.perfil_nutricional_id
            ORDER BY
                m.perfil_nutricional_id,
                m.tipo_medicion,
                m.es_valor_actual DESC,
                m.fecha_medicion DESC NULLS LAST,
                m.id DESC
        ),
        peso_actual AS (
            SELECT perfil_nutricional_id, valor_decimal AS peso_kg
            FROM latest_measure
            WHERE tipo_medicion = 'PESO_KG'
        ),
        altura_actual AS (
            SELECT perfil_nutricional_id, valor_decimal AS altura_cm
            FROM latest_measure
            WHERE tipo_medicion = 'ALTURA_CM'
        ),
        enfermedades_agg AS (
            SELECT
                t.perfil_nutricional_id,
                string_agg(t.nombre, ', ' ORDER BY t.nombre) AS enfermedades,
                array_agg(t.nombre ORDER BY t.nombre) AS enfermedades_items
            FROM (
                SELECT DISTINCT pe.perfil_nutricional_id, e.nombre
                FROM perfil_nutricional_enfermedad pe
                JOIN mae_enfermedad_cie10 e ON e.id = pe.enfermedad_id
                JOIN profile_base pb ON pb.id = pe.perfil_nutricional_id
                WHERE pe.vigente = TRUE
            ) t
            GROUP BY t.perfil_nutricional_id
        ),
        restricciones_agg AS (
            SELECT
                t.perfil_nutricional_id,
                string_agg(t.nombre, ', ' ORDER BY t.nombre) AS restricciones_alimentarias,
                array_agg(t.nombre ORDER BY t.nombre) AS restricciones_items
            FROM (
                SELECT DISTINCT pr.perfil_nutricional_id, r.nombre
                FROM perfil_nutricional_restriccion pr
                JOIN mae_restriccion_alimentaria r ON r.id = pr.restriccion_id
                JOIN profile_base pb ON pb.id = pr.perfil_nutricional_id
                WHERE pr.vigente = TRUE
            ) t
            GROUP BY t.perfil_nutricional_id
        ),
        alergias_agg AS (
            SELECT
                t.perfil_nutricional_id,
                string_agg(t.nombre, ', ' ORDER BY t.nombre) AS alergias,
                array_agg(t.nombre ORDER BY t.nombre) AS alergias_items
            FROM (
                SELECT DISTINCT pr.perfil_nutricional_id, r.nombre
                FROM perfil_nutricional_restriccion pr
                JOIN mae_restriccion_alimentaria r ON r.id = pr.restriccion_id
                JOIN profile_base pb ON pb.id = pr.perfil_nutricional_id
                WHERE pr.vigente = TRUE
                  AND r.tipo = 'ALERGENO'
            ) t
            GROUP BY t.perfil_nutricional_id
        ),
        latest_extractions AS (
            SELECT DISTINCT ON (pe.usuario_id, pe.field_code)
                pe.usuario_id,
                pe.field_code,
                COALESCE(NULLIF(btrim(pe.raw_value), ''), pe.normalized_value) AS value_text
            FROM profile_extractions pe
            WHERE pe.usuario_id = :uid
              AND pe.status = 'confirmed'
            ORDER BY pe.usuario_id, pe.field_code, pe.extracted_at DESC NULLS LAST, pe.id DESC
        ),
        extract_pivot AS (
            SELECT
                usuario_id,
                max(value_text) FILTER (WHERE field_code = 'tipo_dieta') AS tipo_dieta,
                max(value_text) FILTER (WHERE field_code = 'alergias') AS alergias,
                max(value_text) FILTER (WHERE field_code = 'enfermedades') AS enfermedades,
                max(value_text) FILTER (WHERE field_code = 'restricciones_alimentarias') AS restricciones_alimentarias,
                max(value_text) FILTER (WHERE field_code = 'objetivo_nutricional') AS objetivo_nutricional,
                max(value_text) FILTER (WHERE field_code = 'region') AS region,
                max(value_text) FILTER (WHERE field_code = 'provincia') AS provincia,
                max(value_text) FILTER (WHERE field_code = 'distrito') AS distrito
            FROM latest_extractions
            GROUP BY usuario_id
        )
        SELECT
            bu.usuario_id,
            pb.edad_reportada AS edad,
            pw.peso_kg,
            ah.altura_cm,
            COALESCE(pa.nombre, ep.tipo_dieta) AS tipo_dieta,
            COALESCE(al.alergias, ep.alergias) AS alergias,
            COALESCE(enf.enfermedades, ep.enfermedades) AS enfermedades,
            COALESCE(rs.restricciones_alimentarias, ep.restricciones_alimentarias) AS restricciones_alimentarias,
            COALESCE(
                al.alergias_items,
                CASE
                    WHEN ep.alergias IS NULL OR btrim(ep.alergias) = '' THEN NULL
                    ELSE regexp_split_to_array(ep.alergias, '\\s*,\\s*')
                END
            ) AS alergias_items,
            COALESCE(
                enf.enfermedades_items,
                CASE
                    WHEN ep.enfermedades IS NULL OR btrim(ep.enfermedades) = '' THEN NULL
                    ELSE regexp_split_to_array(ep.enfermedades, '\\s*,\\s*')
                END
            ) AS enfermedades_items,
            COALESCE(
                rs.restricciones_items,
                CASE
                    WHEN ep.restricciones_alimentarias IS NULL OR btrim(ep.restricciones_alimentarias) = '' THEN NULL
                    ELSE regexp_split_to_array(ep.restricciones_alimentarias, '\\s*,\\s*')
                END
            ) AS restricciones_items,
            COALESCE(obj.nombre, ep.objetivo_nutricional) AS objetivo_nutricional,
            COALESCE(dep.nombre, ep.region) AS region,
            COALESCE(prov.nombre, ep.provincia) AS provincia,
            COALESCE(dist.nombre, ep.distrito) AS distrito,
            COALESCE(pb.skipped_fields, '{}'::jsonb) AS skipped_fields
        FROM base_user bu
        LEFT JOIN profile_base pb ON pb.usuario_id = bu.usuario_id
        LEFT JOIN peso_actual pw ON pw.perfil_nutricional_id = pb.id
        LEFT JOIN altura_actual ah ON ah.perfil_nutricional_id = pb.id
        LEFT JOIN enfermedades_agg enf ON enf.perfil_nutricional_id = pb.id
        LEFT JOIN restricciones_agg rs ON rs.perfil_nutricional_id = pb.id
        LEFT JOIN alergias_agg al ON al.perfil_nutricional_id = pb.id
        LEFT JOIN mae_patron_alimentario pa ON pa.id = pb.patron_alimentario_id
        LEFT JOIN mae_objetivo_nutricional obj ON obj.id = pb.objetivo_nutricional_id
        LEFT JOIN mae_distrito dist ON dist.id = pb.distrito_id
        LEFT JOIN mae_provincia prov ON prov.id = dist.provincia_id
        LEFT JOIN mae_departamento dep ON dep.id = prov.departamento_id
        LEFT JOIN extract_pivot ep ON ep.usuario_id = bu.usuario_id
        """
    )

    async def fetch_snapshot(self, session: AsyncSession, uid: int) -> Optional[ProfileSnapshot]:
        result = await session.execute(self._PROFILE_PROJECTION_SQL, {"uid": uid})
        row = result.mappings().fetchone()
        if not row:
            return None
        return ProfileSnapshot.from_row(dict(row))

    async def fetch_projection(self, session: AsyncSession, uid: int) -> dict:
        """
        Compatibility projection used by legacy callers while V3 adoption completes.
        """
        snapshot = await self.fetch_snapshot(session, uid)
        if not snapshot:
            return {}
        return snapshot.to_legacy_dict()
