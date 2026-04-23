import asyncio
import pandas as pd
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    engine = create_async_engine(Settings().database_url)
    
    try:
        # Load departaments
        logger.info("Loading departamentos...")
        df_dep = pd.read_csv('masters/mae_departamento.csv', sep=';', dtype=str, encoding='utf-8')
        df_dep = df_dep.rename(columns={'cod_departamento': 'id', 'nombre_departamento': 'nombre'})
        
        # Load provincias
        logger.info("Loading provincias...")
        df_prov = pd.read_csv('masters/mae_provincia.csv', sep=';', dtype=str, encoding='utf-8')
        df_prov = df_prov.rename(columns={'cod_provincia': 'id', 'nombre_provincia': 'nombre', 'cod_departamento': 'departamento_id'})
        
        # Load distritos
        logger.info("Loading distritos...")
        df_dist = pd.read_csv('masters/mae_distrito.csv', sep=';', dtype=str, encoding='utf-8')
        df_dist = df_dist.rename(columns={'cod_distrito': 'id', 'nombre_distrito': 'nombre', 'cod_provincia': 'provincia_id', 'cod_departamento': 'departamento_id'})

        async with engine.begin() as conn:
            # clear tables first
            await conn.execute(text("TRUNCATE TABLE mae_distrito CASCADE"))
            await conn.execute(text("TRUNCATE TABLE mae_provincia CASCADE"))
            await conn.execute(text("TRUNCATE TABLE mae_departamento CASCADE"))

            # insert departamentos
            for _, row in df_dep.iterrows():
                await conn.execute(
                    text("INSERT INTO mae_departamento (id, nombre) VALUES (:id, :nombre)"),
                    {"id": row['id'], "nombre": row['nombre']}
                )

            # insert provincias
            for _, row in df_prov.iterrows():
                await conn.execute(
                    text("INSERT INTO mae_provincia (id, nombre, departamento_id) VALUES (:id, :nombre, :departamento_id)"),
                    {"id": row['id'], "nombre": row['nombre'], "departamento_id": row['departamento_id']}
                )

            # insert distritos
            for _, row in df_dist.iterrows():
                await conn.execute(
                    text("INSERT INTO mae_distrito (id, nombre, provincia_id, departamento_id) VALUES (:id, :nombre, :provincia_id, :departamento_id)"),
                    {"id": row['id'], "nombre": row['nombre'], "provincia_id": row['provincia_id'], "departamento_id": row['departamento_id']}
                )
                
        logger.info("Ubigeo seeded successfully!")
    except Exception as e:
        logger.error(f"Error seeding ubigeo: {e}")
    finally:
        await engine.dispose()

if __name__ == '__main__':
    asyncio.run(main())
