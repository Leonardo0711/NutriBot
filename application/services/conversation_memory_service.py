"""
Nutribot Backend - ConversationMemoryService
Maneja la recuperacion y persistencia de memoria_chat.
"""
import logging
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class ConversationMemoryService:
    async def load_recent_history(self, session: AsyncSession, uid: int) -> list[dict]:
        try:
            res = await session.execute(
                text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid"),
                {"uid": uid},
            )
            val = res.scalar()
            return val if isinstance(val, list) else []
        except Exception as e:
            logger.error("Error recuperando historial para usuario %s: %s", uid, e)
            return []

    async def append_turn(self, session: AsyncSession, uid: int, user_text: str, assistant_reply: str) -> None:
        try:
            new_pair = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_reply},
            ]

            await session.execute(
                text(
                    """
                    INSERT INTO memoria_chat (usuario_id, historial_mensajes, actualizado_en)
                    VALUES (:uid, '[]'::jsonb, TIMEZONE('America/Lima', NOW()))
                    ON CONFLICT (usuario_id) DO NOTHING
                    """
                ),
                {"uid": uid},
            )

            res = await session.execute(
                text("SELECT historial_mensajes FROM memoria_chat WHERE usuario_id = :uid FOR UPDATE"),
                {"uid": uid},
            )
            hist = res.scalar() or []
            hist.extend(new_pair)
            hist = hist[-20:]

            update_stmt = text(
                """
                UPDATE memoria_chat
                SET historial_mensajes = :hist,
                    actualizado_en = TIMEZONE('America/Lima', NOW())
                WHERE usuario_id = :uid
                """
            ).bindparams(bindparam("hist", type_=JSONB))
            await session.execute(update_stmt, {"uid": uid, "hist": hist})
        except Exception as e:
            logger.error("Error actualizando memoria_chat para usuario %s: %s", uid, e)


