
import asyncio
from infrastructure.db.connection import get_session_factory
from sqlalchemy import text

async def check_mem():
    factory = get_session_factory()
    async with factory() as s:
        try:
            res = await s.execute(text("SELECT * FROM memoria_chat LIMIT 3"))
            print("MEMORIA CHAT CONTENT:", res.fetchall())
            
            res_schema = await s.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'memoria_chat'"))
            print("MEMORIA CHAT SCHEMA:", res_schema.fetchall())
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(check_mem())
