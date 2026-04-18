-- ============================================================================
-- 004_peru_timestamp_defaults.sql
-- Objetivo:
-- 1) Evitar desfases horarios (ej. +5h) en columnas TIMESTAMP WITHOUT TIME ZONE.
-- 2) Backfill de filas antiguas con timestamps nulos en tablas operativas clave.
-- 3) Estandarizar defaults en hora Peru usando TIMEZONE('America/Lima', NOW()).
-- ============================================================================

-- Usuarios
ALTER TABLE usuarios
  ALTER COLUMN creado_en SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN actualizado_en SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE usuarios
SET
  creado_en = COALESCE(creado_en, TIMEZONE('America/Lima', NOW())),
  actualizado_en = COALESCE(actualizado_en, creado_en, TIMEZONE('America/Lima', NOW()));

-- Inbox
ALTER TABLE incoming_messages
  ALTER COLUMN created_at SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN updated_at SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE incoming_messages
SET
  created_at = COALESCE(created_at, TIMEZONE('America/Lima', NOW())),
  updated_at = COALESCE(updated_at, created_at, TIMEZONE('America/Lima', NOW()));

-- Outbox
ALTER TABLE outgoing_messages
  ALTER COLUMN created_at SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN updated_at SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN scheduled_at SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE outgoing_messages
SET
  created_at = COALESCE(created_at, TIMEZONE('America/Lima', NOW())),
  updated_at = COALESCE(updated_at, created_at, TIMEZONE('America/Lima', NOW())),
  scheduled_at = COALESCE(scheduled_at, created_at, TIMEZONE('America/Lima', NOW()));

-- Estado conversacional
ALTER TABLE conversation_state
  ALTER COLUMN updated_at SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE conversation_state
SET updated_at = COALESCE(updated_at, TIMEZONE('America/Lima', NOW()));

-- Memoria de chat
ALTER TABLE memoria_chat
  ALTER COLUMN ultima_interaccion SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN actualizado_en SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE memoria_chat
SET
  ultima_interaccion = COALESCE(ultima_interaccion, TIMEZONE('America/Lima', NOW())),
  actualizado_en = COALESCE(actualizado_en, ultima_interaccion, TIMEZONE('America/Lima', NOW()));

-- Perfil nutricional base
ALTER TABLE perfil_nutricional
  ALTER COLUMN creado_en SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN actualizado_en SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE perfil_nutricional
SET
  creado_en = COALESCE(creado_en, TIMEZONE('America/Lima', NOW())),
  actualizado_en = COALESCE(actualizado_en, creado_en, TIMEZONE('America/Lima', NOW()));

-- Jobs de extraccion
ALTER TABLE extraction_jobs
  ALTER COLUMN created_at SET DEFAULT TIMEZONE('America/Lima', NOW()),
  ALTER COLUMN updated_at SET DEFAULT TIMEZONE('America/Lima', NOW());

UPDATE extraction_jobs
SET
  created_at = COALESCE(created_at, TIMEZONE('America/Lima', NOW())),
  updated_at = COALESCE(updated_at, created_at, TIMEZONE('America/Lima', NOW()));

