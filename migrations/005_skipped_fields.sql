-- 005_skipped_fields.sql
-- Añadir columna para rastrear campos omitidos explícitamente en el perfil nutricional

ALTER TABLE perfil_nutricional
ADD COLUMN IF NOT EXISTS skipped_fields JSONB NOT NULL DEFAULT '{}';

-- Comentario para documentación
COMMENT ON COLUMN perfil_nutricional.skipped_fields IS 'Almacena los campos que el usuario decidió omitir explícitamente (ej: {"alergias": true})';
