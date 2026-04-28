-- 007_performance_indexes.sql
-- Indices conservadores para bajar latencia operativa sin cambiar logica.
-- Se enfocan en los WHERE/ORDER BY existentes de inbox, outbox, snapshots
-- de perfil, reglas nutricionales, memoria y busqueda semantica textual.

-- Colas operativas: claim de workers y rescate de zombies.
CREATE INDEX IF NOT EXISTS idx_incoming_messages_claim_pending
ON incoming_messages (created_at ASC, id ASC)
WHERE status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_incoming_messages_processing_locked
ON incoming_messages (locked_at ASC, id ASC)
WHERE status = 'processing';

CREATE INDEX IF NOT EXISTS idx_outgoing_messages_claim_pending_due
ON outgoing_messages (scheduled_at ASC, created_at ASC, id ASC)
WHERE status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_outgoing_messages_processing_locked
ON outgoing_messages (locked_at ASC, id ASC)
WHERE status IN ('processing', 'sending');

-- Read model del perfil por usuario.
CREATE INDEX IF NOT EXISTS idx_perfil_medicion_actual
ON perfil_nutricional_medicion (
    perfil_nutricional_id,
    tipo_medicion,
    es_valor_actual DESC,
    fecha_medicion DESC,
    id DESC
);

CREATE INDEX IF NOT EXISTS idx_perfil_enfermedad_vigente
ON perfil_nutricional_enfermedad (perfil_nutricional_id, enfermedad_id)
WHERE vigente = TRUE;

CREATE INDEX IF NOT EXISTS idx_perfil_restriccion_vigente
ON perfil_nutricional_restriccion (perfil_nutricional_id, restriccion_id)
WHERE vigente = TRUE;

CREATE INDEX IF NOT EXISTS idx_profile_extractions_latest_confirmed
ON profile_extractions (usuario_id, field_code, extracted_at DESC, id DESC)
WHERE status = 'confirmed';

-- Reglas nutricionales por tablas relacionales.
CREATE INDEX IF NOT EXISTS idx_rel_enf_grupo_active_enfermedad
ON rel_enfermedad_grupo_nutricional (enfermedad_id, grupo_nutricional_id)
WHERE activo = TRUE;

CREATE INDEX IF NOT EXISTS idx_rel_grupo_dieta_active_group
ON rel_grupo_nutricional_dieta (grupo_nutricional_id, prioridad, dieta_terapeutica_id)
WHERE activo = TRUE;

CREATE INDEX IF NOT EXISTS idx_rel_grupo_restriccion_active_group
ON rel_grupo_nutricional_restriccion (grupo_nutricional_id, prioridad, restriccion_id)
WHERE activo = TRUE;

CREATE INDEX IF NOT EXISTS idx_orden_dietetica_active_by_profile
ON orden_dietetica (perfil_nutricional_id, estado)
WHERE vigente = TRUE;

-- Capa semantica textual. Los indices base existen, estos agregan el filtro activo
-- usado en las consultas frecuentes y evitan escanear entradas inactivas.
CREATE INDEX IF NOT EXISTS idx_mae_alias_semantico_active_exact
ON mae_alias_semantico (entidad_tipo, alias_normalizado, prioridad ASC, es_canonico DESC)
WHERE activo = TRUE;

CREATE INDEX IF NOT EXISTS idx_semantic_catalog_active_exact
ON semantic_catalog (entidad_tipo, texto_normalizado, peso_lexico DESC)
WHERE activo = TRUE;

-- Memoria e historial conversacional.
CREATE INDEX IF NOT EXISTS idx_memoria_chat_usuario
ON memoria_chat (usuario_id);

-- Jobs de embeddings procesados por sweeper.
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_claim_pending
ON embedding_jobs (creado_en ASC, id ASC)
WHERE estado IN ('PENDING', 'FAILED');
