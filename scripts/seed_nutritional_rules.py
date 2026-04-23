"""
Nutribot Backend — Seed Nutritional Rules
==========================================
Carga las relaciones clínicas en las tablas rel_* para que el motor
de reglas nutricionales funcione.

Uso:
    python scripts/seed_nutritional_rules.py

Idempotente: usa ON CONFLICT DO NOTHING.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from infrastructure.db.connection import get_session_factory


# ─── rel_enfermedad_grupo_nutricional ───
# Mapea CIE-10 a Grupo Nutricional
ENFERMEDAD_GRUPO = [
    # DIABETES (grupo 1)
    ("E10", 1, "OBLIGATORIA"),
    ("E10.9", 1, "OBLIGATORIA"),
    ("E11", 1, "OBLIGATORIA"),
    ("E11.9", 1, "OBLIGATORIA"),
    # HIPERTENSION (grupo 2)
    ("I10", 2, "OBLIGATORIA"),
    ("I11", 2, "OBLIGATORIA"),
    ("I15", 2, "OBLIGATORIA"),
    # RENAL CONSERVADOR (grupo 3)
    ("N18", 3, "OBLIGATORIA"),
    ("N18.9", 3, "OBLIGATORIA"),
    # SOBREPESO_OBESIDAD (grupo 5)
    ("E66", 5, "OBLIGATORIA"),
    ("E66.9", 5, "OBLIGATORIA"),
    # DISLIPIDEMIA (grupo 6)
    ("E78", 6, "OBLIGATORIA"),
    ("E78.0", 6, "OBLIGATORIA"),
    # GASTRO_ALTO (grupo 7)
    ("K21", 7, "OBLIGATORIA"),
    ("K29", 7, "OBLIGATORIA"),
    # GASTRO_BAJO (grupo 8)
    ("K58", 8, "OBLIGATORIA"),
    # HEPATOPATIA (grupo 9)
    ("K76.0", 9, "OBLIGATORIA"),
    # ONCOLOGIA (grupo 10)
    ("C00", 10, "FRECUENTE"),
    ("C80", 10, "FRECUENTE"),
    # DESNUTRICION (grupo 11)
    ("E44", 11, "OBLIGATORIA"),
    ("E46", 11, "OBLIGATORIA"),
    # CELIAQUIA (grupo 12)
    ("K90.0", 12, "OBLIGATORIA"),
    # RIESGO_CARDIOVASCULAR (grupo 13)
    ("I25", 13, "OBLIGATORIA"),
    ("I50", 13, "OBLIGATORIA"),
    # GESTACION_LACTANCIA (grupo 14)
    ("O99", 14, "FRECUENTE"),
    # HIPOTIROIDISMO (grupo 19)
    ("E03", 19, "OBLIGATORIA"),
    ("E03.9", 19, "OBLIGATORIA"),
    # HIPERURICEMIA (grupo 20)
    ("E79", 20, "OBLIGATORIA"),
    ("E79.0", 20, "OBLIGATORIA"),
    # SALUD_OSEA (grupo 17)
    ("M81", 17, "OBLIGATORIA"),
    ("M81.0", 17, "OBLIGATORIA"),
    # ANEMIA -> podemos asociar a DESNUTRICION si no hay grupo específico
    ("D50", 11, "FRECUENTE"),
    ("D50.9", 11, "FRECUENTE"),
]

# ─── rel_grupo_nutricional_dieta ───
# Mapea Grupo Nutricional -> Dietas Terapéuticas
GRUPO_DIETA = [
    # DIABETES (1) -> Control Glucémico (6), Hipocalórica sugerida (3)
    (1, 6, 1, "OBLIGATORIA"),
    (1, 3, 2, "SUGERIDA"),
    # HIPERTENSION (2) -> DASH (15), Hiposódica (2)
    (2, 15, 1, "OBLIGATORIA"),
    (2, 2, 2, "OBLIGATORIA"),
    # RENAL_CONSERVADOR (3) -> Renal prediálisis (7)
    (3, 7, 1, "OBLIGATORIA"),
    (3, 2, 2, "SUGERIDA"),
    # RENAL_DIALISIS (4) -> Renal diálisis (8), Hiperproteica (5)
    (4, 8, 1, "OBLIGATORIA"),
    (4, 5, 2, "SUGERIDA"),
    # SOBREPESO_OBESIDAD (5) -> Hipocalórica (3)
    (5, 3, 1, "OBLIGATORIA"),
    (5, 12, 2, "SUGERIDA"),  # Alta en fibra
    # DISLIPIDEMIA (6) -> Hipolipídica (9)
    (6, 9, 1, "OBLIGATORIA"),
    (6, 12, 2, "SUGERIDA"),  # Alta en fibra
    # GASTRO_ALTO (7) -> Protección Gástrica (10)
    (7, 10, 1, "OBLIGATORIA"),
    # GASTRO_BAJO (8) -> Baja en FODMAPs (14), Astringente sugerida (11)
    (8, 14, 1, "SUGERIDA"),
    (8, 11, 2, "SUGERIDA"),
    (8, 12, 3, "SUGERIDA"),  # Alta en fibra
    # HEPATOPATIA (9) -> Hepatoprotectora (13)
    (9, 13, 1, "OBLIGATORIA"),
    # ONCOLOGIA (10) -> Hipercalórica (4), Hiperproteica (5)
    (10, 4, 1, "SUGERIDA"),
    (10, 5, 2, "SUGERIDA"),
    # DESNUTRICION (11) -> Hipercalórica (4), Hiperproteica (5)
    (11, 4, 1, "OBLIGATORIA"),
    (11, 5, 2, "OBLIGATORIA"),
    # CELIAQUIA (12) -> Normal Basal (1) con restricción
    (12, 1, 1, "SUGERIDA"),
    # RIESGO_CARDIOVASCULAR (13) -> DASH (15), Hipolipídica (9), Hiposódica (2)
    (13, 15, 1, "OBLIGATORIA"),
    (13, 9, 2, "SUGERIDA"),
    (13, 2, 3, "SUGERIDA"),
    # GESTACION_LACTANCIA (14) -> Normal basal reforzada
    (14, 1, 1, "SUGERIDA"),
    # SANO_PREVENTIVO (15) -> Normal basal
    (15, 1, 1, "SUGERIDA"),
    # HIPOTIROIDISMO (19) -> Hipocalórica sugerida
    (19, 3, 1, "SUGERIDA"),
    # HIPERURICEMIA (20) -> Hipopurínica (16)
    (20, 16, 1, "OBLIGATORIA"),
    # SALUD_OSEA (17) -> Normal basal
    (17, 1, 1, "SUGERIDA"),
]

# ─── rel_grupo_nutricional_restriccion ───
# Mapea Grupo Nutricional -> Restricciones Alimentarias
GRUPO_RESTRICCION = [
    # DIABETES (1) -> Sin azúcar añadida (20)
    (1, 20, 1, True),
    # HIPERTENSION (2) -> (implícita en la dieta, sin restricción extra)
    # RENAL_CONSERVADOR (3) -> Bajo en fósforo (24), Bajo en potasio (25), Control líquidos (30)
    (3, 24, 1, True),
    (3, 25, 2, True),
    (3, 30, 3, False),
    # RENAL_DIALISIS (4) -> Bajo en fósforo (24), Bajo en potasio (25), Control líquidos (30)
    (4, 24, 1, True),
    (4, 25, 2, True),
    (4, 30, 3, True),
    # GASTRO_ALTO (7) -> Sin picante (23), Sin cafeína (22)
    (7, 23, 1, True),
    (7, 22, 2, False),
    # HEPATOPATIA (9) -> Sin alcohol (21)
    (9, 21, 1, True),
    # CELIAQUIA (12) -> Sin gluten (2)
    (12, 2, 1, True),
    # HIPERURICEMIA (20) -> Sin alcohol (21)
    (20, 21, 1, True),
]


async def seed():
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            # 1. rel_enfermedad_grupo_nutricional
            count_eg = 0
            for cie10, grupo_id, tipo in ENFERMEDAD_GRUPO:
                result = await session.execute(
                    text("""
                        INSERT INTO rel_enfermedad_grupo_nutricional
                            (enfermedad_id, grupo_nutricional_id, tipo_relacion, activo)
                        SELECT e.id, :grupo_id, :tipo, true
                        FROM mae_enfermedad_cie10 e
                        WHERE e.codigo_cie10 = :cie10
                        ON CONFLICT (enfermedad_id, grupo_nutricional_id) DO NOTHING
                    """),
                    {"cie10": cie10, "grupo_id": grupo_id, "tipo": tipo},
                )
                count_eg += result.rowcount

            # 2. rel_grupo_nutricional_dieta
            count_gd = 0
            for grupo_id, dieta_id, prioridad, tipo in GRUPO_DIETA:
                result = await session.execute(
                    text("""
                        INSERT INTO rel_grupo_nutricional_dieta
                            (grupo_nutricional_id, dieta_terapeutica_id, prioridad, tipo_relacion, activo)
                        VALUES (:grupo_id, :dieta_id, :prioridad, :tipo, true)
                        ON CONFLICT (grupo_nutricional_id, dieta_terapeutica_id) DO NOTHING
                    """),
                    {"grupo_id": grupo_id, "dieta_id": dieta_id, "prioridad": prioridad, "tipo": tipo},
                )
                count_gd += result.rowcount

            # 3. rel_grupo_nutricional_restriccion
            count_gr = 0
            for grupo_id, restriccion_id, prioridad, obligatoria in GRUPO_RESTRICCION:
                result = await session.execute(
                    text("""
                        INSERT INTO rel_grupo_nutricional_restriccion
                            (grupo_nutricional_id, restriccion_id, prioridad, obligatoria, activo)
                        VALUES (:grupo_id, :restriccion_id, :prioridad, :obligatoria, true)
                        ON CONFLICT (grupo_nutricional_id, restriccion_id) DO NOTHING
                    """),
                    {"grupo_id": grupo_id, "restriccion_id": restriccion_id, "prioridad": prioridad, "obligatoria": obligatoria},
                )
                count_gr += result.rowcount

            print(f"rel_enfermedad_grupo_nutricional: {count_eg} filas insertadas")
            print(f"rel_grupo_nutricional_dieta: {count_gd} filas insertadas")
            print(f"rel_grupo_nutricional_restriccion: {count_gr} filas insertadas")
            print("Seed completado.")


if __name__ == "__main__":
    asyncio.run(seed())
