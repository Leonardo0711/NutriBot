"""
Nutribot Backend - LLM Reply Service
"""
from __future__ import annotations

import logging
from typing import Optional

from application.services.profile_context_service import ProfileContextService
from domain.context_builder import build_llm_context, try_fast_response
from domain.entities import ConversationState, NormalizedMessage
from domain.ports import LLMService
from domain.profile_snapshot import ProfileSnapshot
from domain.reply_objects import BotReply
from domain.router import RouteResult

logger = logging.getLogger(__name__)


class LlmReplyService:
    def __init__(self, llm_service: LLMService, system_instructions: str, profile_context: ProfileContextService):
        self._llm_service = llm_service
        self._system_instructions = system_instructions
        self._profile_context = profile_context

    async def generate_reply(
        self,
        *,
        onboarding_interception_happened: bool,
        reply: Optional[str],
        state_snapshot: ConversationState,
        normalized: NormalizedMessage,
        route: RouteResult,
        rag_text: Optional[str],
        history: list[dict],
        profile_text: str,
        snapshot: ProfileSnapshot,
        extracted_data: dict,
        has_absurd_profile_claim: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        new_response_id = state_snapshot.last_openai_response_id
        if onboarding_interception_happened or reply is not None:
            return reply, new_response_id

        fast = try_fast_response(route)
        if fast:
            logger.info(
                "FastPath: user=%s intent=%s reply sin LLM",
                getattr(state_snapshot, "usuario_id", "unknown"),
                route.intent.value,
            )
            return fast, new_response_id

        extra_instr = ""
        if extracted_data:
            confirm_list = []
            for key, value in extracted_data.items():
                if key == "peso_kg":
                    c_name = "peso"
                elif key == "altura_cm":
                    c_name = "talla"
                elif key == "restricciones_alimentarias":
                    c_name = "restricciones"
                elif key == "objetivo_nutricional":
                    c_name = "objetivo"
                else:
                    c_name = key
                confirm_list.append(f"{c_name} a '{value}'")
            extra_instr = (
                "\n\n[INSTRUCCION CRITICA: El sistema acaba de actualizar estos datos del perfil: "
                + ", ".join(confirm_list)
                + ". DEBES empezar tu respuesta confirmando de forma breve y natural que ya guardaste "
                + "esta informacion (ej: 'Listo, ya registre tu nuevo peso'). NO ignores esta instruccion.]"
            )

        if has_absurd_profile_claim:
            extra_instr += (
                "\n\n[INSTRUCCION CRITICA: El usuario menciono un dato de alergia/salud inverosimil o ficticio. "
                "NO lo confirmes ni lo guardes. Responde de forma amable pidiendo aclaracion con un dato real.]"
            )

        final_profile_context = profile_text if profile_text else None
        if final_profile_context and is_asking_for_recommendation:
            citation = self._profile_context.recommendation_citation(snapshot)
            extra_instr += (
                "\n\n[REGLA DE PERSONALIZACION]\n"
                "PRIORIDAD DE PETICION: Si el usuario pide explicitamente algo que esta en sus restricciones "
                "o alergias (ej: pide receta de pescado teniendo restriccion de pescado), CUMPLE con la peticion "
                "pero adviertele brevemente sobre su restriccion registrada. Su deseo actual manda sobre su perfil previo."
            )
            final_profile_context = (
                "[INSTRUCCION CRITICA DE FORMATO]\n"
                "Tu respuesta DEBE comenzar OBLIGATORIAMENTE con el siguiente texto exacto "
                "(no agregues 'Hola' antes de esto):\n"
                f'"{citation}"\n\n'
                "[DATOS DE PERFIL PARA TU ANALISIS INTERNO]\n"
                f"{profile_text}"
            )

        final_instructions = self._system_instructions + extra_instr
        llm_ctx = build_llm_context(
            route=route,
            instructions=final_instructions,
            history=history,
            rag_context=rag_text,
            profile_context=final_profile_context,
        )

        reply, new_response_id = await self._llm_service.generate_reply(
            state=state_snapshot,
            normalized=normalized,
            instructions=llm_ctx.instructions,
            rag_context=llm_ctx.rag_context,
            profile_context=llm_ctx.profile_context,
            history=llm_ctx.history,
            max_tokens=llm_ctx.max_tokens,
        )
        return reply, new_response_id

    @staticmethod
    def append_continuity_tip(
        *,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        turns_since_last_prompt: int,
        is_requesting_survey: bool,
    ) -> Optional[str]:
        if not reply or onboarding_interception_happened:
            return reply
        should_append_tip = bool(
            "PD:" not in reply
            and turns_since_last_prompt > 0
            and turns_since_last_prompt % 4 == 0
            and not is_requesting_survey
        )
        if should_append_tip:
            reply += (
                '\n\nTip NutriBot: Si quieres actualizar tu perfil cuando gustes, solo escribe: '
                '"Nutribot, quiero actualizar mi perfil nutricional".'
            )
        return reply

    @staticmethod
    def sanitize_final_reply(final_bot_reply: BotReply, uid: int) -> BotReply:
        if final_bot_reply.content_type == "text":
            if not final_bot_reply.text or not str(final_bot_reply.text).strip():
                final_bot_reply.text = "Perdon, tuve un problema interno. Intenta nuevamente en unos segundos."
                logger.warning("Fallback por respuesta vacia en orchestrator user=%s", uid)
            return final_bot_reply

        if not final_bot_reply.payload_json:
            return BotReply(
                text="Perdon, tuve un problema interno. Intenta nuevamente en unos segundos.",
                content_type="text",
            )
        if not final_bot_reply.text:
            final_bot_reply.text = str(final_bot_reply.payload_json.get("body") or "").strip()
        return final_bot_reply
