"""
Nutribot Backend - Conversation Repository (SQLAlchemy)
Implementa el puerto ConversationRepository.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState
from domain.ports import ConversationRepository
from .connection import get_session_factory


class SqlAlchemyConversationRepository(ConversationRepository):

    async def get_state_no_lock(self, usuario_id: int) -> ConversationState:
        """
        Lee el estado conversacional SIN lock.
        Usa una sesion corta aislada para evitar retener transacciones
        mientras el llamador hace operaciones de red (STT, LLM, etc.).
        """
        factory = get_session_factory()
        async with factory() as session:
            async with session.begin():
                result = await session.execute(
                    text("SELECT * FROM conversation_state WHERE usuario_id = :uid"),
                    {"uid": usuario_id},
                )
                row = result.fetchone()
                if row is None:
                    await session.execute(
                        text("""
                            INSERT INTO conversation_state (usuario_id)
                            VALUES (:uid)
                            ON CONFLICT (usuario_id) DO NOTHING
                        """),
                        {"uid": usuario_id},
                    )
                    result = await session.execute(
                        text("SELECT * FROM conversation_state WHERE usuario_id = :uid"),
                        {"uid": usuario_id},
                    )
                    row = result.fetchone()
                return self._row_to_state(row, usuario_id)

    async def get_state_for_update(
        self, session: AsyncSession, usuario_id: int
    ) -> ConversationState:
        """
        Lee el estado con SELECT ... FOR UPDATE.
        DEBE ejecutarse dentro de una transaccion activa (session.begin()).
        """
        result = await session.execute(
            text("SELECT * FROM conversation_state WHERE usuario_id = :uid FOR UPDATE"),
            {"uid": usuario_id},
        )
        row = result.fetchone()
        if row is None:
            await session.execute(
                text("""
                    INSERT INTO conversation_state (usuario_id)
                    VALUES (:uid)
                    ON CONFLICT (usuario_id) DO NOTHING
                """),
                {"uid": usuario_id},
            )
            result = await session.execute(
                text("SELECT * FROM conversation_state WHERE usuario_id = :uid FOR UPDATE"),
                {"uid": usuario_id},
            )
            row = result.fetchone()
        return self._row_to_state(row, usuario_id)

    async def save_state(
        self, session: AsyncSession, state: ConversationState
    ) -> None:
        """
        Persiste el estado actualizado.
        DEBE ejecutarse dentro de la misma transaccion que get_state_for_update.
        """
        await session.execute(
            text("""
                UPDATE conversation_state SET
                    mode = :mode,
                    awaiting_question_code = :aqc,
                    last_provider_message_id = :lpmid,
                    last_turn_at = TIMEZONE('America/Lima', NOW()),
                    last_form_prompt_at = :lfpa,
                    turns_since_last_prompt = :tslp,
                    profile_completion_pct = :pcp,
                    usability_completion_pct = :ucp,
                    meaningful_interactions_count = :mic,
                    last_openai_response_id = :lorid,
                    onboarding_status = :ost,
                    onboarding_step = :ostep,
                    onboarding_last_invited_at = :olia,
                    onboarding_next_eligible_at = :onea,
                    onboarding_skip_count = :osc,
                    onboarding_updated_at = :ouat,
                    survey_next_eligible_count = :snec,
                    survey_decline_count = :sdc,
                    survey_paused_reason = :spr,
                    survey_updated_at = :suat,
                    version = :ver,
                    updated_at = TIMEZONE('America/Lima', NOW())
                WHERE usuario_id = :uid
            """),
            {
                "uid": state.usuario_id,
                "mode": state.mode,
                "aqc": state.awaiting_question_code,
                "lpmid": state.last_provider_message_id,
                "lfpa": state.last_form_prompt_at,
                "tslp": state.turns_since_last_prompt,
                "pcp": state.profile_completion_pct,
                "ucp": state.usability_completion_pct,
                "mic": state.meaningful_interactions_count,
                "lorid": state.last_openai_response_id,
                "ost": state.onboarding_status,
                "ostep": state.onboarding_step,
                "olia": state.onboarding_last_invited_at,
                "onea": state.onboarding_next_eligible_at,
                "osc": state.onboarding_skip_count,
                "ouat": state.onboarding_updated_at,
                "snec": state.survey_next_eligible_count,
                "sdc": state.survey_decline_count,
                "spr": state.survey_paused_reason,
                "suat": state.survey_updated_at,
                "ver": state.version,
            },
        )

    @staticmethod
    def _row_to_state(row, usuario_id: int) -> ConversationState:
        """Convierte una fila de la BD a un ConversationState."""
        if row is None:
            return ConversationState(usuario_id=usuario_id)
        return ConversationState(
            usuario_id=row.usuario_id,
            mode=row.mode,
            awaiting_question_code=row.awaiting_question_code,
            last_provider_message_id=row.last_provider_message_id,
            last_turn_at=row.last_turn_at,
            last_form_prompt_at=row.last_form_prompt_at,
            turns_since_last_prompt=row.turns_since_last_prompt,
            profile_completion_pct=row.profile_completion_pct,
            usability_completion_pct=row.usability_completion_pct,
            meaningful_interactions_count=getattr(row, "meaningful_interactions_count", 0),
            last_openai_response_id=row.last_openai_response_id,
            onboarding_status=getattr(row, "onboarding_status", "not_started"),
            onboarding_step=getattr(row, "onboarding_step", None),
            onboarding_last_invited_at=getattr(row, "onboarding_last_invited_at", None),
            onboarding_next_eligible_at=getattr(row, "onboarding_next_eligible_at", None),
            onboarding_skip_count=getattr(row, "onboarding_skip_count", 0),
            onboarding_updated_at=getattr(row, "onboarding_updated_at", None),
            survey_next_eligible_count=getattr(row, "survey_next_eligible_count", None),
            survey_decline_count=getattr(row, "survey_decline_count", 0),
            survey_paused_reason=getattr(row, "survey_paused_reason", None),
            survey_updated_at=getattr(row, "survey_updated_at", None),
            version=row.version,
            updated_at=row.updated_at,
        )

