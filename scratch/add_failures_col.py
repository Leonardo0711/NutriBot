import asyncio
from sqlalchemy import text
from infrastructure.db.connection import engine_manager

async def add_column():
    async with engine_manager.get_engine().begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE conversation_state ADD COLUMN IF NOT EXISTS onboarding_consecutive_failures INTEGER DEFAULT 0"))
            print("Column added successfully")
        except Exception as e:
            print(f"Error adding column: {e}")

if __name__ == "__main__":
    asyncio.run(add_column())
