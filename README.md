# Nutribot Backend (Fuente Oficial)

Este backend usa arquitectura de perfil nutricional normalizada y onboarding progresivo.

## Fuente de verdad

- Modelo de datos funcional y relaciones: `BD/dbdiagram.txt`
- Evolucion de esquema en produccion: `migrations/001_v3_schema.sql`, `migrations/002_v3_profile_compat.sql`, `migrations/003_semantic_alias_local_food_terms.sql`, `migrations/004_peru_timestamp_defaults.sql`

## Flujo conversacional vigente

- Orquestacion: `application/services/message_orchestrator.py`
- Pipeline final de respuesta (peruanizacion + tono + limpieza markdown + disclaimer): `application/services/llm_reply_service.py`
- Onboarding progresivo fase 1/2: `application/services/onboarding_service.py` y `application/services/profile_interception_service.py`
- Cableado de dependencias: `di.py`

## Nota de mantenimiento

Se eliminaron snapshots SQL legacy para evitar inconsistencias documentales.
Si necesitas un snapshot nuevo, generarlo siempre desde la BD actual y fecharlo explicitamente.
