import asyncio
from infrastructure.db.connection import get_session_factory
from sqlalchemy import text
import re
import sys

async def main():
    factory = get_session_factory()
    # Test casting logic
    value = "23"
    try:
        f_val = float(re.sub(r"[^\d\.]", "", value))
        f_val2 = int(re.sub(r"\D", "", value))
        print(f"Casting passed: {f_val}, {f_val2}")
    except Exception as e:
        print("Casting failed:", e)

    async with factory() as session:
        # Check raw current state
        res = await session.execute(text("SELECT usuario_id, onboarding_status, onboarding_step FROM conversation_state ORDER BY updated_at DESC LIMIT 1"))
        state = res.fetchone()
        row_dict = dict(state._mapping) if state else None
        print("Latest State:", row_dict)
        if not state:
            return

        # Check this user's profile
        res2 = await session.execute(text("SELECT id, usuario_id, edad, peso_kg, altura_cm FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state[0]})
        prof = res2.fetchone()
        print("Latest Profile:", dict(prof._mapping) if prof else "None!!")
        
        # Test directly finding next missing step
        from application.advance_onboarding_flow import _find_next_missing_step
        nxt = await _find_next_missing_step(session, state[0])
        print("Next missing step via func:", nxt)

asyncio.run(main())
