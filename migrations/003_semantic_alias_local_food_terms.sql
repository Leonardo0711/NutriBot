-- 003_semantic_alias_local_food_terms.sql
-- Semillas de alias locales para mejorar entendimiento de terminos internacionales
-- en capas semanticas y normalizacion previa al LLM.

INSERT INTO mae_alias_semantico (
    entidad_tipo,
    entidad_codigo,
    alias_texto,
    alias_normalizado,
    idioma,
    es_canonico,
    prioridad,
    activo,
    creado_en,
    actualizado_en
)
VALUES
    ('LEXICO_ALIMENTO', 'CAMOTE', 'boniato', 'boniato', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'CAMOTE', 'batata', 'batata', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'PALTA', 'aguacate', 'aguacate', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'QUINUA', 'quinoa', 'quinoa', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'CHOCLO', 'elote', 'elote', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'CHOCLO', 'maiz tierno', 'maiz tierno', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'REFRIGERIO', 'merienda', 'merienda', 'es', false, 5, true, NOW(), NOW()),
    ('LEXICO_ALIMENTO', 'REFRIGERIO', 'snack', 'snack', 'es', false, 5, true, NOW(), NOW())
ON CONFLICT (entidad_tipo, entidad_codigo, alias_normalizado)
DO UPDATE SET
    alias_texto = EXCLUDED.alias_texto,
    idioma = EXCLUDED.idioma,
    es_canonico = EXCLUDED.es_canonico,
    prioridad = EXCLUDED.prioridad,
    activo = EXCLUDED.activo,
    actualizado_en = NOW();
