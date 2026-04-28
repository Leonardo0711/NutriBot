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
    _RAW_HISTORY_LIMIT = 6
    _STORED_HISTORY_LIMIT = 20
    _SUMMARY_CHAR_LIMIT = 900

    async def load_recent_history(self, session: AsyncSession, uid: int) -> list[dict]:
        try:
            res = await session.execute(
                text("""
                    SELECT resumen, temas_clave, ultima_recomendacion, historial_mensajes
                    FROM memoria_chat
                    WHERE usuario_id = :uid
                """),
                {"uid": uid},
            )
            row = res.mappings().first()
            if not row:
                return []

            history = row.get("historial_mensajes")
            recent = history[-self._RAW_HISTORY_LIMIT:] if isinstance(history, list) else []
            compact = self._build_compact_memory_item(
                row.get("resumen"),
                row.get("temas_clave"),
                row.get("ultima_recomendacion"),
            )
            return ([compact] if compact else []) + recent
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
            hist = hist[-self._STORED_HISTORY_LIMIT:]
            compact_summary = self._next_summary(hist)
            last_recommendation = self._extract_last_recommendation(hist)
            key_topics = self._extract_key_topics(hist)

            update_stmt = text(
                """
                UPDATE memoria_chat
                SET historial_mensajes = :hist,
                    resumen = :summary,
                    temas_clave = :topics,
                    ultima_recomendacion = COALESCE(:last_recommendation, ultima_recomendacion),
                    actualizado_en = TIMEZONE('America/Lima', NOW())
                WHERE usuario_id = :uid
                """
            ).bindparams(bindparam("hist", type_=JSONB))
            await session.execute(
                update_stmt,
                {
                    "uid": uid,
                    "hist": hist,
                    "summary": compact_summary,
                    "topics": key_topics,
                    "last_recommendation": last_recommendation,
                },
            )
        except Exception as e:
            logger.error("Error actualizando memoria_chat para usuario %s: %s", uid, e)

    def _build_compact_memory_item(
        self,
        summary: str | None,
        topics: str | None,
        last_recommendation: str | None,
    ) -> dict | None:
        parts = []
        if summary:
            parts.append(f"Resumen compacto: {summary}")
        if topics:
            parts.append(f"Temas clave: {topics}")
        if last_recommendation:
            parts.append(f"Ultima recomendacion relevante: {last_recommendation}")
        if not parts:
            return None
        return {"role": "memory_summary", "content": "\n".join(parts)}

    def _next_summary(self, history: list[dict]) -> str:
        useful = []
        for item in history[-8:]:
            role = item.get("role", "")
            content = self._clean_line(item.get("content", ""))
            if not content:
                continue
            useful.append(f"{role}: {content}")
        summary = " | ".join(useful)
        if len(summary) <= self._SUMMARY_CHAR_LIMIT:
            return summary
        return summary[-self._SUMMARY_CHAR_LIMIT:].lstrip(" |")

    def _extract_last_recommendation(self, history: list[dict]) -> str | None:
        markers = ("recom", "menu", "receta", "desayuno", "almuerzo", "cena", "alimento")
        for item in reversed(history):
            if item.get("role") != "assistant":
                continue
            content = self._clean_line(item.get("content", ""))
            normalized = content.lower()
            if content and any(marker in normalized for marker in markers):
                return content[:500]
        return None

    def _extract_key_topics(self, history: list[dict]) -> str | None:
        markers = (
            "anemia", "hierro", "hemoglobina", "diabetes", "hipertension",
            "peso", "talla", "imc", "alergia", "restriccion", "cafe",
            "menu", "receta", "dieta", "ejercicio",
        )
        found = []
        blob = " ".join(self._clean_line(item.get("content", "")).lower() for item in history[-12:])
        for marker in markers:
            if marker in blob:
                found.append(marker)
        return ", ".join(found[:10]) if found else None

    @staticmethod
    def _clean_line(value: str) -> str:
        return " ".join(str(value or "").split())
