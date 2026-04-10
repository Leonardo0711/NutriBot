import asyncio
from sqlalchemy import text
from infrastructure.db.connection import get_engine

async def list_tables():
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        tables = [r[0] for r in res]
        print("Tables in public schema:")
        for t in tables:
            print(f"- {t}")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(list_tables())
