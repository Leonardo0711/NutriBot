-- Ensure pgvector-backed columns are typed as vector in runtime PostgreSQL.
-- This keeps RAG and semantic matching queries using "<=>" working reliably.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'fragmentos_rag'
      AND column_name = 'embedding'
      AND udt_name <> 'vector'
  ) THEN
    ALTER TABLE public.fragmentos_rag
      ALTER COLUMN embedding TYPE vector
      USING CASE
        WHEN embedding IS NULL OR btrim(embedding) = '' THEN NULL
        ELSE embedding::vector
      END;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'semantic_catalog'
      AND column_name = 'embedding'
      AND udt_name <> 'vector'
  ) THEN
    ALTER TABLE public.semantic_catalog
      ALTER COLUMN embedding TYPE vector
      USING CASE
        WHEN embedding IS NULL OR btrim(embedding) = '' THEN NULL
        ELSE embedding::vector
      END;
  END IF;
END $$;
