"""
Nutribot Backend — NutritionalRulesService
============================================
Motor de reglas nutricionales que cruza enfermedades del usuario contra
las tablas relacionales para generar contexto clínico estructurado y
órdenes dietéticas automáticas.

Este servicio es ADITIVO: no modifica ninguna lógica existente.
Solo lee datos del perfil y tablas de reglas, y genera/actualiza
órdenes dietéticas de tipo RECOMENDADA.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.utils import get_now_peru

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DietRule:
    """Una dieta sugerida/obligatoria por las reglas."""
    dieta_id: int
    dieta_codigo: str
    dieta_nombre: str
    tipo_relacion: str  # OBLIGATORIA, SUGERIDA, CONTRAINDICADA
    prioridad: int
    grupo_origen: str   # nombre del grupo que la sugiere


@dataclass(frozen=True)
class RestrictionRule:
    """Una restricción derivada de las reglas."""
    restriccion_id: int
    restriccion_codigo: str
    restriccion_nombre: str
    obligatoria: bool
    grupo_origen: str


@dataclass(frozen=True)
class NutritionalRulesContext:
    """Contexto nutricional derivado de las reglas de la BD."""
    groups: tuple[str, ...] = ()
    suggested_diets: tuple[DietRule, ...] = ()
    mandatory_diets: tuple[DietRule, ...] = ()
    contraindicated_diets: tuple[DietRule, ...] = ()
    rule_restrictions: tuple[RestrictionRule, ...] = ()
    has_active_order: bool = False

    @property
    def has_rules(self) -> bool:
        return bool(self.groups)

    @property
    def group_names(self) -> tuple[str, ...]:
        return self.groups

    @property
    def all_diet_names(self) -> tuple[str, ...]:
        seen = set()
        result = []
        for d in list(self.mandatory_diets) + list(self.suggested_diets):
            if d.dieta_nombre not in seen:
                seen.add(d.dieta_nombre)
                result.append(d.dieta_nombre)
        return tuple(result)

    @property
    def mandatory_restriction_names(self) -> tuple[str, ...]:
        return tuple(r.restriccion_nombre for r in self.rule_restrictions if r.obligatoria)

    @property
    def all_restriction_names(self) -> tuple[str, ...]:
        seen = set()
        result = []
        for r in self.rule_restrictions:
            if r.restriccion_nombre not in seen:
                seen.add(r.restriccion_nombre)
                result.append(r.restriccion_nombre)
        return tuple(result)


class NutritionalRulesService:
    """
    Motor de reglas nutricionales.
    Lee las tablas rel_* para generar contexto clínico y órdenes dietéticas.
    """

    # ─── SQL Queries ───

    _RESOLVE_GROUPS_SQL = text("""
        SELECT DISTINCT
            gn.id AS grupo_id,
            gn.codigo AS grupo_codigo,
            gn.nombre AS grupo_nombre,
            reg.tipo_relacion
        FROM perfil_nutricional_enfermedad pne
        JOIN perfil_nutricional pn ON pn.id = pne.perfil_nutricional_id
        JOIN rel_enfermedad_grupo_nutricional reg ON reg.enfermedad_id = pne.enfermedad_id AND reg.activo = true
        JOIN mae_grupo_nutricional gn ON gn.id = reg.grupo_nutricional_id AND gn.activo = true
        WHERE pn.usuario_id = :uid
          AND pne.vigente = true
        ORDER BY gn.id
    """)

    _RESOLVE_DIETS_SQL = text("""
        SELECT DISTINCT
            dt.id AS dieta_id,
            dt.codigo AS dieta_codigo,
            dt.nombre AS dieta_nombre,
            rgd.tipo_relacion,
            rgd.prioridad,
            gn.nombre AS grupo_origen
        FROM rel_grupo_nutricional_dieta rgd
        JOIN mae_dieta_terapeutica dt ON dt.id = rgd.dieta_terapeutica_id AND dt.activo = true
        JOIN mae_grupo_nutricional gn ON gn.id = rgd.grupo_nutricional_id
        WHERE rgd.grupo_nutricional_id = ANY(:group_ids)
          AND rgd.activo = true
        ORDER BY rgd.prioridad, dt.nombre
    """)

    _RESOLVE_RESTRICTIONS_SQL = text("""
        SELECT DISTINCT
            ra.id AS restriccion_id,
            ra.codigo AS restriccion_codigo,
            ra.nombre AS restriccion_nombre,
            rgr.obligatoria,
            gn.nombre AS grupo_origen
        FROM rel_grupo_nutricional_restriccion rgr
        JOIN mae_restriccion_alimentaria ra ON ra.id = rgr.restriccion_id AND ra.activo = true
        JOIN mae_grupo_nutricional gn ON gn.id = rgr.grupo_nutricional_id
        WHERE rgr.grupo_nutricional_id = ANY(:group_ids)
          AND rgr.activo = true
        ORDER BY rgr.prioridad, ra.nombre
    """)

    _CHECK_ACTIVE_ORDER_SQL = text("""
        SELECT od.id
        FROM orden_dietetica od
        JOIN perfil_nutricional pn ON pn.id = od.perfil_nutricional_id
        WHERE pn.usuario_id = :uid
          AND od.vigente = true
          AND od.estado IN ('ACTIVA', 'BORRADOR')
        LIMIT 1
    """)

    # ─── Core Methods ───

    async def resolve_nutritional_context(
        self,
        session: AsyncSession,
        usuario_id: int,
    ) -> NutritionalRulesContext:
        """
        Dado un usuario, resuelve su contexto nutricional completo
        cruzando perfil → enfermedades → grupos → dietas/restricciones.
        """
        try:
            # 1. Obtener grupos nutricionales del usuario
            result = await session.execute(self._RESOLVE_GROUPS_SQL, {"uid": usuario_id})
            group_rows = result.fetchall()

            if not group_rows:
                return NutritionalRulesContext()

            group_names = tuple(r.grupo_nombre for r in group_rows)
            group_ids = [r.grupo_id for r in group_rows]

            # 2. Obtener dietas sugeridas/obligatorias
            result = await session.execute(self._RESOLVE_DIETS_SQL, {"group_ids": group_ids})
            diet_rows = result.fetchall()

            mandatory = []
            suggested = []
            contraindicated = []
            seen_diets = set()

            for r in diet_rows:
                if r.dieta_id in seen_diets:
                    continue
                seen_diets.add(r.dieta_id)
                rule = DietRule(
                    dieta_id=r.dieta_id,
                    dieta_codigo=r.dieta_codigo,
                    dieta_nombre=r.dieta_nombre,
                    tipo_relacion=r.tipo_relacion,
                    prioridad=r.prioridad,
                    grupo_origen=r.grupo_origen,
                )
                if r.tipo_relacion == "OBLIGATORIA":
                    mandatory.append(rule)
                elif r.tipo_relacion == "CONTRAINDICADA":
                    contraindicated.append(rule)
                else:
                    suggested.append(rule)

            # 3. Obtener restricciones derivadas
            result = await session.execute(self._RESOLVE_RESTRICTIONS_SQL, {"group_ids": group_ids})
            restriction_rows = result.fetchall()

            restrictions = []
            seen_restrictions = set()
            for r in restriction_rows:
                if r.restriccion_id in seen_restrictions:
                    continue
                seen_restrictions.add(r.restriccion_id)
                restrictions.append(RestrictionRule(
                    restriccion_id=r.restriccion_id,
                    restriccion_codigo=r.restriccion_codigo,
                    restriccion_nombre=r.restriccion_nombre,
                    obligatoria=r.obligatoria,
                    grupo_origen=r.grupo_origen,
                ))

            # 4. Verificar si ya tiene una orden activa
            result = await session.execute(self._CHECK_ACTIVE_ORDER_SQL, {"uid": usuario_id})
            has_order = result.fetchone() is not None

            ctx = NutritionalRulesContext(
                groups=group_names,
                mandatory_diets=tuple(mandatory),
                suggested_diets=tuple(suggested),
                contraindicated_diets=tuple(contraindicated),
                rule_restrictions=tuple(restrictions),
                has_active_order=has_order,
            )

            if ctx.has_rules:
                logger.info(
                    "NutritionalRules: user=%s groups=%s diets=%d restrictions=%d",
                    usuario_id,
                    group_names,
                    len(mandatory) + len(suggested),
                    len(restrictions),
                )

            return ctx

        except Exception as e:
            logger.error("NutritionalRules: Error resolving context for user=%s: %s", usuario_id, e)
            return NutritionalRulesContext()

    async def generate_or_update_dietary_order(
        self,
        session: AsyncSession,
        usuario_id: int,
    ) -> Optional[int]:
        """
        Genera o actualiza una orden dietética tipo RECOMENDADA para el usuario,
        basada en sus enfermedades y las reglas nutricionales.
        
        Retorna el ID de la orden creada/actualizada, o None si no hay reglas aplicables.
        """
        try:
            ctx = await self.resolve_nutritional_context(session, usuario_id)
            if not ctx.has_rules:
                return None

            all_diets = list(ctx.mandatory_diets) + list(ctx.suggested_diets)
            if not all_diets and not ctx.rule_restrictions:
                return None

            now = get_now_peru()

            # Obtener perfil_nutricional_id
            result = await session.execute(
                text("SELECT id FROM perfil_nutricional WHERE usuario_id = :uid LIMIT 1"),
                {"uid": usuario_id},
            )
            perfil_row = result.fetchone()
            if not perfil_row:
                return None
            perfil_id = perfil_row.id

            # Cerrar órdenes anteriores de tipo RECOMENDADA del mismo usuario
            await session.execute(
                text("""
                    UPDATE orden_dietetica
                    SET vigente = false, estado = 'FINALIZADA', fecha_fin = :now, actualizado_en = :now
                    WHERE perfil_nutricional_id = :pid
                      AND fuente_orden = 'RECOMENDADA'
                      AND vigente = true
                """),
                {"pid": perfil_id, "now": now},
            )

            # Crear nueva orden
            result = await session.execute(
                text("""
                    INSERT INTO orden_dietetica
                        (perfil_nutricional_id, estado, fecha_inicio, vigente,
                         indicada_por, fuente_orden, observacion, creado_en, actualizado_en)
                    VALUES
                        (:pid, 'ACTIVA', :now, true,
                         'nutribot_sistema', 'RECOMENDADA',
                         :obs, :now, :now)
                    RETURNING id
                """),
                {
                    "pid": perfil_id,
                    "now": now,
                    "obs": f"Generada automaticamente. Grupos: {', '.join(ctx.group_names)}",
                },
            )
            orden_id = result.scalar_one()

            # Insertar dietas de la orden
            for i, diet in enumerate(all_diets):
                await session.execute(
                    text("""
                        INSERT INTO orden_dietetica_dieta
                            (orden_dietetica_id, dieta_terapeutica_id, es_principal, prioridad,
                             observacion, activo, creado_en)
                        VALUES (:oid, :did, :principal, :prio, :obs, true, :now)
                        ON CONFLICT (orden_dietetica_id, dieta_terapeutica_id) DO NOTHING
                    """),
                    {
                        "oid": orden_id,
                        "did": diet.dieta_id,
                        "principal": i == 0,
                        "prio": diet.prioridad,
                        "obs": f"Derivada del grupo: {diet.grupo_origen} ({diet.tipo_relacion})",
                        "now": now,
                    },
                )

            # Insertar restricciones de la orden
            for restr in ctx.rule_restrictions:
                await session.execute(
                    text("""
                        INSERT INTO orden_dietetica_restriccion
                            (orden_dietetica_id, restriccion_id, obligatoria, observacion, activo, creado_en)
                        VALUES (:oid, :rid, :oblig, :obs, true, :now)
                        ON CONFLICT (orden_dietetica_id, restriccion_id) DO NOTHING
                    """),
                    {
                        "oid": orden_id,
                        "rid": restr.restriccion_id,
                        "oblig": restr.obligatoria,
                        "obs": f"Derivada del grupo: {restr.grupo_origen}",
                        "now": now,
                    },
                )

            logger.info(
                "NutritionalRules: Created dietary order id=%s for user=%s (diets=%d, restrictions=%d)",
                orden_id,
                usuario_id,
                len(all_diets),
                len(ctx.rule_restrictions),
            )
            return orden_id

        except Exception as e:
            logger.error("NutritionalRules: Error generating dietary order for user=%s: %s", usuario_id, e)
            return None

    def build_rules_prompt_context(self, ctx: NutritionalRulesContext) -> Optional[str]:
        """
        Genera un bloque de texto compacto con el contexto de reglas nutricionales
        para inyectar en el prompt del LLM.
        
        Retorna None si no hay reglas aplicables.
        """
        if not ctx.has_rules:
            return None

        parts = ["[CONTEXTO NUTRICIONAL DERIVADO DE REGLAS CLINICAS]"]
        parts.append(f"Grupos nutricionales detectados: {', '.join(ctx.group_names)}")

        if ctx.mandatory_diets:
            names = [f"{d.dieta_nombre} (obligatoria)" for d in ctx.mandatory_diets]
            parts.append(f"Dietas indicadas: {', '.join(names)}")

        if ctx.suggested_diets:
            names = [f"{d.dieta_nombre} (sugerida)" for d in ctx.suggested_diets]
            parts.append(f"Dietas sugeridas: {', '.join(names)}")

        if ctx.contraindicated_diets:
            names = [d.dieta_nombre for d in ctx.contraindicated_diets]
            parts.append(f"Dietas CONTRAINDICADAS: {', '.join(names)}")

        if ctx.rule_restrictions:
            mandatory = [r.restriccion_nombre for r in ctx.rule_restrictions if r.obligatoria]
            suggested = [r.restriccion_nombre for r in ctx.rule_restrictions if not r.obligatoria]
            if mandatory:
                parts.append(f"Restricciones obligatorias: {', '.join(mandatory)}")
            if suggested:
                parts.append(f"Restricciones sugeridas: {', '.join(suggested)}")

        parts.append(
            "IMPORTANTE: Esta informacion es orientacion basada en reglas clinicas. "
            "El usuario puede estar preguntando para familiares u otras personas. "
            "Nunca restrinjas ni rechaces un pedido; si hay un conflicto con su perfil, "
            "incluye una advertencia breve al inicio y luego responde normalmente."
        )

        return "\n".join(parts)
