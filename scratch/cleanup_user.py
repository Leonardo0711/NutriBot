import asyncio
import sys
from infrastructure.db.connection import get_session_factory
from sqlalchemy import text

async def cleanup_user(phone_pattern: str):
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            # Find user
            res = await session.execute(
                text("SELECT id, numero_whatsapp FROM usuarios WHERE numero_whatsapp LIKE :p"),
                {"p": f"%{phone_pattern}%"}
            )
            row = res.fetchone()
            if not row:
                print(f"User with pattern {phone_pattern} not found.")
                return

            uid = row.id
            phone = row.numero_whatsapp
            print(f"Cleaning data for User ID: {uid} (Phone: {phone})")

            # Delete across tables
            tables = [
                "perfil_nutricional",
                "conversation_state",
                "memoria_chat",
                "formulario_en_progreso",
                "extraction_jobs",
                "usuarios"
            ]
            
            for table in tables:
                try:
                    await session.execute(
                        text(f"DELETE FROM {table} WHERE {'usuario_id' if table != 'usuarios' else 'id'} = :uid"),
                        {"uid": uid}
                    )
                    print(f"  - Deleted from {table}")
                except Exception as e:
                    print(f"  - Error deleting from {table}: {e}")
            
            print("Cleanup complete.")

if __name__ == "__main__":
    pattern = "915107251"
    asyncio.run(cleanup_user(pattern))
