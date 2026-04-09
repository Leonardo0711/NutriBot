import asyncio
import os
import sys

# Añadir el path del proyecto
sys.path.append(os.getcwd())

from infrastructure.database import SessionLocal
from sqlalchemy import text

async def check_db():
    session = SessionLocal()
    try:
        r = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = 1"))
        row = r.fetchone()
        if row:
            print("DATABASE STATE FOR USER 1:")
            # Convertir Row a dict de forma segura
            for k, v in row._mapping.items():
                print(f"  {k}: {v}")
        else:
            print("NO PROFILE FOUND FOR USER 1")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await session.close()

if __name__ == "__main__":
    asyncio.run(check_db())
