-- 006_meaningful_interactions_count.sql

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS meaningful_interactions_count INTEGER NOT NULL DEFAULT 0;
