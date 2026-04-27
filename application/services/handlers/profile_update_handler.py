"""
Nutribot Backend - Profile Update Handler
Atiende actualizaciones manuales del perfil (cuando no está en flujo cerrado de onboarding).
"""
import logging
from typing import Optional, Tuple

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply
from domain.router import Intent
from application.services.handlers.base_handler import BaseHandler
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_context_service import ProfileContextService


logger = logging.getLogger(__name__)


class ProfileUpdateHandler(BaseHandler):
    def __init__(
        self,
        profile_extractor: ProfileExtractionService,
        profile_context: ProfileContextService,
        fallback_handler: BaseHandler,
    ):
        self._profile_extractor = profile_extractor
        self._profile_context = profile_context
        self._fallback_handler = fallback_handler

    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        # Detector de absurdos para bloquear cosas locas
        ctx.has_absurd_profile_claim = self._profile_extractor.contains_absurd_claim(ctx.normalized.text)
        
        extracted_data = {}

        # ── Prioridad 1: profile_intent del extractor (comprensión real) ──
        # Usa apply_profile_intent() que respeta operation, entity_code, strategy
        if ctx.profile_intent and ctx.profile_intent.is_profile_update:
            intent = ctx.profile_intent

            # Si el resolvedor semántico detectó ambigüedad → pedir aclaración
            if intent.needs_clarification and intent.clarification_question:
                return BotReply(
                    text=intent.clarification_question,
                    content_type="text",
                ), None

            # Aplicar la intención completa respetando operación y entidades resueltas
            ext_result = await self._profile_extractor.apply_profile_intent(
                session=ctx.session,
                usuario_id=ctx.user.id,
                intent=intent,
                state=ctx.state,
            )
            if ext_result:
                extracted_data = ext_result.clean_data
                ctx.extracted_data = extracted_data
                if extracted_data:
                    logger.info(
                        "Intent-based profile update user=%s field=%s op=%s: %s",
                        ctx.user.id, intent.field_code, intent.operation, extracted_data,
                    )

                if ext_result.meta_flags.get("needs_health_clarification"):
                    return BotReply(
                        text=ext_result.meta_flags.get(
                            "clarification_prompt",
                            "¿Podrías aclarar ese aspecto médico un poco más?",
                        ),
                        content_type="text",
                    ), None

            # Continuar al fallback para generar respuesta contextual
            return await self._fallback_handler.handle(ctx)

        # ── Prioridad 2: Fast path del router (campo y valor claros, sin LLM) ──
        if (
            ctx.route.resolved_field
            and ctx.route.resolved_value
            and ctx.route.confidence >= 0.7
            and ctx.route.intent in (Intent.PROFILE_UPDATE, Intent.CORRECTION_PAST_FIELD, Intent.ANSWER_CURRENT_STEP)
        ):
            raw_extractions = {ctx.route.resolved_field: ctx.route.resolved_value}
            current_step_hint = (
                ctx.route.resolved_field
                if ctx.route.intent == Intent.ANSWER_CURRENT_STEP
                else None
            )
            ext_result = await self._profile_extractor.apply_cleaning_and_save(
                raw_extractions=raw_extractions,
                user_text=ctx.normalized.text,
                usuario_id=ctx.user.id,
                session=ctx.session,
                current_step=current_step_hint,
            )
            logger.info(
                "Router-based profile update (no LLM): user=%s field=%s value=%s",
                ctx.user.id,
                ctx.route.resolved_field,
                ctx.route.resolved_value,
            )
        else:
            # Slow path: llamamos a LLM local para extraer
            ext_result = await self._profile_extractor.extract_and_save(
                user_text=ctx.normalized.text,
                usuario_id=ctx.user.id,
                session=ctx.session,
                current_step=None,
            )

        if ext_result:
            extracted_data = ext_result.clean_data
            meta_flags = ext_result.meta_flags
            
            ctx.extracted_data = extracted_data

            if extracted_data:
                logger.info("Real-time profile update user=%s: %s", ctx.user.id, extracted_data)

            # Bloqueo interactivo si hay duda médica
            if meta_flags.get("needs_health_clarification"):
                return BotReply(
                    text=meta_flags.get("clarification_prompt", "¿Podrías aclarar ese aspecto médico un poco más?"),
                    content_type="text",
                ), None

        # Si llegamos aquí, se actualizaron los datos o no hubo match exacto.
        # Continuamos con el flujo general para generar la respuesta contextual final.
        return await self._fallback_handler.handle(ctx)
