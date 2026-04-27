-- Migration 006: Survey snooze fields
-- Permite pausar la encuesta de usabilidad cuando el usuario
-- está actualizando su perfil, evitando interrumpir el flujo.

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS survey_next_eligible_count int;

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS survey_decline_count int NOT NULL DEFAULT 0;

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS survey_paused_reason varchar(30);

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS survey_updated_at timestamp;
