-- 004_onboarding.sql

ALTER TABLE conversation_state
ADD COLUMN IF NOT EXISTS onboarding_status VARCHAR(20) NOT NULL DEFAULT 'not_started',
ADD COLUMN IF NOT EXISTS onboarding_step VARCHAR(50),
ADD COLUMN IF NOT EXISTS onboarding_last_invited_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS onboarding_next_eligible_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS onboarding_skip_count SMALLINT NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS onboarding_updated_at TIMESTAMP NOT NULL DEFAULT NOW();

-- Verificar si la tabla perfil_nutricional existe y está correcta
CREATE TABLE IF NOT EXISTS perfil_nutricional (
    id BIGSERIAL PRIMARY KEY,
    usuario_id BIGINT NOT NULL REFERENCES usuarios(id),
    edad INT,
    peso_kg DECIMAL(5,2),
    altura_cm DECIMAL(5,2),
    tipo_dieta VARCHAR(50),
    alergias TEXT,
    enfermedades TEXT,
    restricciones_alimentarias TEXT,
    objetivo_nutricional TEXT,
    region VARCHAR(100),
    provincia VARCHAR(100),
    distrito VARCHAR(100),
    fuente_ubicacion VARCHAR(50),
    actualizado_en TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(usuario_id)
);
