import asyncio
import os
import sys
from sqlalchemy import text

# Añadir el path del proyecto
sys.path.append(os.getcwd())

from infrastructure.db.connection import get_session_factory

async def check_user_profile(phone: str):
    factory = get_session_factory()
    async with factory() as session:
        # Get user ID
        r = await session.execute(text("SELECT id FROM usuarios WHERE phone = :ph"), {"ph": phone})
        user = r.fetchone()
        if not user:
            print(f"User with phone {phone} NOT FOUND")
            return
            
        uid = user.id
        print(f"USER ID: {uid}")
        
        # Check profile
        r = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        row = r.fetchone()
        if row:
            print("PROFILE DATA:")
            for k, v in row._mapping.items():
                print(f"  {k}: {v}")
        else:
            print("NO PROFILE FOUND")
            
        # Check conversation state
        r = await session.execute(text("SELECT * FROM conversation_states WHERE usuario_id = :uid"), {"uid": uid})
        state = r.fetchone()
        if state:
            print("CONVERSATION STATE:")
            for k, v in state._mapping.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(check_user_profile("+51915107251"))
