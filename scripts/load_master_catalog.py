import os
import asyncio
import pandas as pd
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración (Ajustar según .env o config.py si es necesario)
DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://nutribot:nutribot@localhost:5432/nutribot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
engine = create_async_engine(DB_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_embedding(text: str) -> List[float]:
    """Obtiene el vector embedding de OpenAI."""
    try:
        resp = await client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Error generando embedding para '{text}': {e}")
        return []

async def load_csv_to_catalog(session: AsyncSession, file_path: str, categoria: str, col_nombre: str, col_codigo: str = None, col_meta: str = None):
    """Carga un CSV y genera embeddings para cada fila."""
    if not os.path.exists(file_path):
        logger.warning(f"Archivo no encontrado: {file_path}")
        return

    logger.info(f"Procesando categoría '{categoria}' desde {file_path}...")
    
    # Manejar diferentes encodigs y separadores
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
    except:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='latin1')

    total = len(df)
    for i, row in df.iterrows():
        nombre = str(row[col_nombre]).strip()
        codigo = str(row[col_codigo]).strip() if col_codigo and col_codigo in row else None
        
        metadata = {}
        if col_meta and col_meta in row:
            metadata["descripcion"] = str(row[col_meta]).strip()

        # Evitar duplicados básicos
        res = await session.execute(
            text("SELECT id FROM catalogo_maestro WHERE categoria = :cat AND nombre = :nom"),
            {"cat": categoria, "nom": nombre}
        )
        if res.fetchone():
            continue

        embedding = await get_embedding(nombre)
        if not embedding:
            continue

        await session.execute(
            text("""
                INSERT INTO catalogo_maestro (categoria, nombre, codigo_externo, metadata, embedding)
                VALUES (:cat, :nom, :cod, :meta, :emb)
            """),
            {
                "cat": categoria,
                "nom": nombre,
                "cod": codigo,
                "meta": pd.io.json.dumps(metadata) if metadata else None,
                "emb": embedding
            }
        )
        
        if (i + 1) % 50 == 0:
            logger.info(f"Progreso '{categoria}': {i+1}/{total}")
            await session.commit()

    await session.commit()
    logger.info(f"Categoría '{categoria}' completada.")

async def main():
    paths = {
        "enfermedad": {
            "path": "MAESTROS_CIE10.csv",
            "col_nom": "NOM_DIAG",
            "col_cod": "COD_DIAG"
        },
        "dieta": {
            "path": "masters/dietas.csv",
            "col_nom": "nombre",
            "col_meta": "descripcion"
        },
        "alergia": {
            "path": "masters/alergias.csv",
            "col_nom": "nombre",
            "col_meta": "categoria"
        },
        "restriccion": {
            "path": "masters/restricciones.csv",
            "col_nom": "nombre",
            "col_meta": "tipo"
        },
        "ubicacion": {
            "path": "masters/ubicaciones.csv",
            "col_nom": "Distrito",
            "col_cod": "Ubigeo"
        }
    }

    async with AsyncSessionLocal() as session:
        for cat, cfg in paths.items():
            await load_csv_to_catalog(
                session, 
                file_path=cfg["path"],
                categoria=cat,
                col_nombre=cfg["col_nom"],
                col_codigo=cfg.get("col_cod"),
                col_meta=cfg.get("col_meta")
            )

if __name__ == "__main__":
    asyncio.run(main())
