-- ======================================================
-- NUTRIBOT BACKEND — Migración 003
-- 5 tablas nuevas para el backend FastAPI
-- Se ejecuta DESPUÉS de 02_esquema.sql
-- ======================================================

-- 1. Idempotencia + True Inbox Pattern
CREATE TABLE IF NOT EXISTS incoming_messages (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider_message_id VARCHAR(255) NOT NULL UNIQUE,
    webhook_payload     JSONB NOT NULL,
    status              VARCHAR(20)  NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','processing','done','failed')),
    locked_at           TIMESTAMP,
    error_detail        TEXT,
    retry_count         SMALLINT NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_incoming_messages_status
    ON incoming_messages(status);
CREATE INDEX IF NOT EXISTS idx_incoming_messages_locked_at_processing
    ON incoming_messages(locked_at) WHERE status = 'processing';


-- 2. State machine completa persistida por usuario
CREATE TABLE IF NOT EXISTS conversation_state (
    usuario_id                  BIGINT PRIMARY KEY REFERENCES usuarios(id) ON DELETE CASCADE,
    mode                        VARCHAR(30) NOT NULL DEFAULT 'active_chat'
                                CHECK (mode IN ('active_chat','closing',
                                                'collecting_profile','collecting_usability')),
    awaiting_field_code         VARCHAR(50),
    awaiting_question_code      VARCHAR(10),
    last_provider_message_id    VARCHAR(255),
    last_turn_at                TIMESTAMP,
    last_form_prompt_at         TIMESTAMP,
    turns_since_last_prompt     SMALLINT NOT NULL DEFAULT 0,
    closure_score               SMALLINT,
    reply_resolved_something    BOOLEAN NOT NULL DEFAULT FALSE,
    profile_completion_pct      SMALLINT NOT NULL DEFAULT 0,
    usability_completion_pct    SMALLINT NOT NULL DEFAULT 0,
    last_openai_response_id     VARCHAR(255),
    version                     INTEGER NOT NULL DEFAULT 1,
    updated_at                  TIMESTAMP NOT NULL DEFAULT NOW()
);


-- 3. Extracciones de perfil con confianza y evidencia
CREATE TABLE IF NOT EXISTS profile_extractions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    usuario_id      BIGINT NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    field_code      VARCHAR(50)  NOT NULL,
    raw_value       TEXT         NOT NULL,
    confidence      NUMERIC(3,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    evidence_text   TEXT         NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'tentative'
                    CHECK (status IN ('tentative','confirmed','rejected')),
    extracted_at    TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_profile_extractions_lookup
    ON profile_extractions(usuario_id, field_code, status);


-- 4. Jobs de extracción de perfil (outbox robusto)
CREATE TABLE IF NOT EXISTS extraction_jobs (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    usuario_id   BIGINT NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    raw_text     TEXT NOT NULL,
    status       VARCHAR(20) NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending','processing','done','failed')),
    retry_count  SMALLINT NOT NULL DEFAULT 0,
    locked_at    TIMESTAMP,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_pending
    ON extraction_jobs(status, created_at) WHERE status IN ('pending','failed');
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_locked
    ON extraction_jobs(locked_at) WHERE status = 'processing';


-- 5. Outbox de mensajes salientes (entrega robusta y TTS diferido)
CREATE TABLE IF NOT EXISTS outgoing_messages (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    idempotency_key  VARCHAR(255) NOT NULL UNIQUE,
    usuario_id       BIGINT NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    phone            VARCHAR(20) NOT NULL,
    content_type     VARCHAR(20) NOT NULL CHECK (content_type IN ('text','audio','audio_tts')),
    content          TEXT NOT NULL,
    status           VARCHAR(20) NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','processing','sending','sent','failed')),
    attempt_count    SMALLINT NOT NULL DEFAULT 0,
    locked_at        TIMESTAMP,
    sent_at          TIMESTAMP,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_outgoing_messages_pending
    ON outgoing_messages(status, created_at) WHERE status IN ('pending','failed');
CREATE INDEX IF NOT EXISTS idx_outgoing_messages_locked
    ON outgoing_messages(locked_at) WHERE status IN ('processing','sending');
