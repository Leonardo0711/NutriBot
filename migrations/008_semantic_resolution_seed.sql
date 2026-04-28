-- Semillas operativas seguras para resolver frases frecuentes sin LLM.
-- No escriben maestros definitivos; solo cachean resoluciones revisables.

INSERT INTO semantic_resolution_cache (
    scope,
    field_code,
    query_texto,
    query_normalizada,
    entidad_tipo_resuelta,
    entidad_codigo_resuelto,
    estrategia_usada,
    confidence,
    hit_count,
    top_candidates_json,
    creado_en,
    actualizado_en
)
VALUES
    (
        'PROFILE_FIELD',
        'objetivo_nutricional',
        'bajar de peso',
        'bajar de peso',
        'OBJETIVO_NUTRICIONAL',
        'PERDIDA_PESO_GENERAL',
        'REVIEW_SEED',
        0.96,
        0,
        '[]'::jsonb,
        TIMEZONE('America/Lima', NOW()),
        TIMEZONE('America/Lima', NOW())
    ),
    (
        'PROFILE_FIELD',
        'objetivo_nutricional',
        'perder peso',
        'perder peso',
        'OBJETIVO_NUTRICIONAL',
        'PERDIDA_PESO_GENERAL',
        'REVIEW_SEED',
        0.96,
        0,
        '[]'::jsonb,
        TIMEZONE('America/Lima', NOW()),
        TIMEZONE('America/Lima', NOW())
    ),
    (
        'PROFILE_FIELD',
        'provincia',
        'Callao',
        'callao',
        'PROVINCIA',
        '0701',
        'REVIEW_SEED',
        0.98,
        0,
        '[]'::jsonb,
        TIMEZONE('America/Lima', NOW()),
        TIMEZONE('America/Lima', NOW())
    ),
    (
        'PROFILE_FIELD',
        'distrito',
        'Callao',
        'callao',
        'DISTRITO',
        '070101',
        'REVIEW_SEED',
        0.98,
        0,
        '[]'::jsonb,
        TIMEZONE('America/Lima', NOW()),
        TIMEZONE('America/Lima', NOW())
    )
ON CONFLICT (scope, field_code, query_normalizada)
DO UPDATE SET
    entidad_tipo_resuelta = EXCLUDED.entidad_tipo_resuelta,
    entidad_codigo_resuelto = EXCLUDED.entidad_codigo_resuelto,
    estrategia_usada = EXCLUDED.estrategia_usada,
    confidence = EXCLUDED.confidence,
    top_candidates_json = EXCLUDED.top_candidates_json,
    actualizado_en = TIMEZONE('America/Lima', NOW());

