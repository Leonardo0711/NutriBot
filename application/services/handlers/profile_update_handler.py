"""
Nutribot Backend - Profile Update Handler
Atiende actualizaciones manuales del perfil (cuando no está en flujo cerrado de onboarding).
"""
import logging
import re
import unicodedata
from typing import Optional, Tuple

from domain.turn_context import TurnContext
from domain.reply_objects import BotReply
from domain.router import Intent
from domain.value_objects import OnboardingStatus, OnboardingStep
from domain.parsers import parse_age, parse_weight, parse_height
from application.services.handlers.base_handler import BaseHandler
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_context_service import ProfileContextService
from application.services.profile_read_service import ProfileReadService
from application.services.conversation_state_service import ConversationStateService


logger = logging.getLogger(__name__)


class ProfileUpdateHandler(BaseHandler):
    def __init__(
        self,
        profile_extractor: ProfileExtractionService,
        profile_context: ProfileContextService,
        fallback_handler: BaseHandler,
        profile_reader: ProfileReadService | None = None,
        state_service: ConversationStateService | None = None,
    ):
        self._profile_extractor = profile_extractor
        self._profile_context = profile_context
        self._fallback_handler = fallback_handler
        self._profile_reader = profile_reader or ProfileReadService()
        self._state_service = state_service or ConversationStateService()

    @staticmethod
    def _ascii(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()

    @staticmethod
    def _first_numeric(patterns: tuple[str, ...], text: str) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_explicit_basic_fields(self, user_text: str) -> dict[str, float | int]:
        """
        Rescate deterministico para edad/peso/talla escritos juntos o con unidades.
        Evita depender del LLM en mensajes como:
        "tengo 46 anos, peso 73 kg y mido 1.72 m".
        """
        txt = self._ascii(user_text)
        if not txt:
            return {}

        clean_data: dict[str, float | int] = {}

        age_raw = self._first_numeric(
            (
                r"\b(\d{1,3})\s*(?:anos?|anios?)\b",
                r"\b(?:edad)\D{0,12}(\d{1,3})\b",
                r"\btengo\s+(\d{1,3})\s*(?:anos?|anios?)\b",
            ),
            txt,
        )
        if age_raw is not None:
            age = parse_age(age_raw)
            if age is not None:
                clean_data["edad"] = age

        weight_match = re.search(
            r"\b(?:peso|pesa|pesando)\D{0,12}(\d+(?:[\.,]\d+)?)\s*(kg|kilos?|quilos?|libras?|lb)?\b",
            txt,
            flags=re.IGNORECASE,
        )
        if not weight_match:
            weight_match = re.search(
                r"\b(\d+(?:[\.,]\d+)?)\s*(kg|kilos?|quilos?|libras?|lb)\b",
                txt,
                flags=re.IGNORECASE,
            )
        if weight_match:
            raw = f"{weight_match.group(1)} {weight_match.group(2) or ''}".strip()
            weight = parse_weight(raw)
            if weight is not None:
                clean_data["peso_kg"] = weight

        height_match = re.search(
            r"\b(?:mido|mide|talla|altura|estatura)\D{0,12}(\d+(?:[\.,]\d+)?)\s*(cm|m|mts?|metros?)?\b",
            txt,
            flags=re.IGNORECASE,
        )
        if not height_match:
            height_match = re.search(
                r"\b(\d+(?:[\.,]\d+)?)\s*(cm|m|mts?|metros?)\b",
                txt,
                flags=re.IGNORECASE,
            )
        if height_match:
            raw = f"{height_match.group(1)} {height_match.group(2) or ''}".strip()
            height = parse_height(raw)
            if height is not None:
                clean_data["altura_cm"] = height

        return clean_data

    @staticmethod
    def _contains_basic_fields(extracted_data: dict) -> bool:
        return bool({"edad", "peso_kg", "altura_cm"}.intersection(extracted_data.keys()))

    @staticmethod
    def _basic_value(snapshot, field_code: str):
        if not snapshot:
            return None
        return snapshot.value_for_step(field_code)

    def _next_missing_basic_step(self, snapshot) -> str | None:
        for step in (OnboardingStep.EDAD, OnboardingStep.PESO, OnboardingStep.ALTURA):
            value = self._basic_value(snapshot, step.value)
            if value is None or (isinstance(value, str) and not value.strip()):
                return step.value
        return None

    @staticmethod
    def _basic_question(step: str) -> str:
        questions = {
            OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂",
            OnboardingStep.PESO.value: "¿Cuánto pesas aproximadamente en kilos? ⚖️",
            OnboardingStep.ALTURA.value: (
                "¿Cuánto mides? 📐\n"
                "Puedes decirme en metros o centímetros.\n"
                "Ej: 1.65 m, 170 cm..."
            ),
        }
        return questions.get(step, "")

    @staticmethod
    def _field_labels(fields: dict) -> str:
        labels = {
            "edad": "edad",
            "peso_kg": "peso",
            "altura_cm": "talla",
        }
        present = [labels[key] for key in ("edad", "peso_kg", "altura_cm") if key in fields]
        if not present:
            return "tu perfil"
        if len(present) == 1:
            return present[0]
        return ", ".join(present[:-1]) + " y " + present[-1]

    @staticmethod
    def _format_basic_summary(snapshot) -> str:
        age = snapshot.value_for_step("edad") if snapshot else None
        weight = snapshot.value_for_step("peso_kg") if snapshot else None
        height = snapshot.value_for_step("altura_cm") if snapshot else None

        lines = []
        if age is not None:
            lines.append(f"- Edad: {int(age)} años")
        if weight is not None:
            lines.append(f"- Peso: {float(weight):.1f} kg")
        if height is not None:
            lines.append(f"- Talla: {float(height) / 100:.2f} m")
        return "\n".join(lines)

    def _should_control_basic_reply(self, ctx: TurnContext) -> bool:
        if ctx.route.intent in {
            Intent.PROFILE_UPDATE,
            Intent.CORRECTION_PAST_FIELD,
            Intent.ANSWER_CURRENT_STEP,
            Intent.AMBIGUOUS,
        }:
            return True
        return ctx.state.onboarding_status in {
            OnboardingStatus.INVITED.value,
            OnboardingStatus.IN_PROGRESS.value,
            OnboardingStatus.PAUSED.value,
        }

    async def _build_basic_profile_reply(
        self,
        ctx: TurnContext,
        extracted_data: dict,
    ) -> BotReply | None:
        if not self._contains_basic_fields(extracted_data):
            return None
        if not self._should_control_basic_reply(ctx):
            return None

        snapshot = getattr(ctx, "snapshot", None)
        next_step = self._next_missing_basic_step(snapshot)
        updated_label = self._field_labels(extracted_data)

        if next_step:
            self._state_service.set_onboarding_in_progress(ctx.state, next_step)
            return BotReply(
                text=(
                    f"Listo, ya actualicé {updated_label} en tu perfil. 😊\n\n"
                    "Para personalizar mejor tus respuestas, sigamos con este dato:\n\n"
                    f"{self._basic_question(next_step)}"
                ),
                content_type="text",
            )

        self._state_service.set_onboarding_completed(ctx.state)
        summary = self._format_basic_summary(snapshot)
        summary_block = f"\n\n{summary}" if summary else ""
        return BotReply(
            text=(
                f"Listo 😊 ya actualicé tu perfil básico.{summary_block}\n\n"
                "Con esto ya puedo darte recomendaciones más personalizadas."
            ),
            content_type="text",
        )

    async def _refresh_profile_context(self, ctx: TurnContext) -> None:
        """Mantiene el turno alineado con los datos recién guardados."""
        try:
            snapshot = await self._profile_reader.fetch_snapshot(ctx.session, ctx.user.id)
        except Exception:
            logger.warning("ProfileUpdateHandler: no se pudo refrescar el perfil del turno", exc_info=True)
            return

        if not snapshot:
            return

        profile_text, summary = self._profile_context.build_prompt_and_summary(snapshot)
        nutritional_rules = getattr(ctx, "nutritional_rules_text", None)
        if nutritional_rules:
            profile_text = f"{profile_text}\n\n{nutritional_rules}"

        ctx.snapshot = snapshot
        ctx.profile_text = profile_text
        ctx.summary = summary

    async def handle(self, ctx: TurnContext) -> Tuple[Optional[BotReply], Optional[str]]:
        # Detector de absurdos para bloquear cosas locas
        ctx.has_absurd_profile_claim = self._profile_extractor.contains_absurd_claim(ctx.normalized.text)
        
        extracted_data = {}
        ext_result = None

        explicit_basic_data = self._extract_explicit_basic_fields(ctx.normalized.text)
        if explicit_basic_data:
            await self._profile_extractor.save_clean_data(
                ctx.user.id,
                explicit_basic_data,
                ctx.session,
                source_text=ctx.normalized.text,
                current_step=None,
            )
            extracted_data = explicit_basic_data
            ctx.extracted_data = extracted_data
            logger.info(
                "Deterministic basic profile update user=%s: %s",
                ctx.user.id,
                extracted_data,
            )
            await self._refresh_profile_context(ctx)
            basic_reply = await self._build_basic_profile_reply(ctx, extracted_data)
            if basic_reply is not None:
                return basic_reply, None

        # ── Prioridad 1: profile_intent del extractor (comprensión real) ──
        # Usa apply_profile_intent() que respeta operation, entity_code, strategy
        if not extracted_data and ctx.profile_intent and ctx.profile_intent.is_profile_update:
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

            if extracted_data:
                await self._refresh_profile_context(ctx)
                basic_reply = await self._build_basic_profile_reply(ctx, extracted_data)
                if basic_reply is not None:
                    return basic_reply, None

            # Continuar al fallback para generar respuesta contextual
            return await self._fallback_handler.handle(ctx)

        # ── Prioridad 2: Fast path del router (campo y valor claros, sin LLM) ──
        if not extracted_data and (
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
        elif not extracted_data:
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
                await self._refresh_profile_context(ctx)
                basic_reply = await self._build_basic_profile_reply(ctx, extracted_data)
                if basic_reply is not None:
                    return basic_reply, None

            # Bloqueo interactivo si hay duda médica
            if meta_flags.get("needs_health_clarification"):
                return BotReply(
                    text=meta_flags.get("clarification_prompt", "¿Podrías aclarar ese aspecto médico un poco más?"),
                    content_type="text",
                ), None

        # Si llegamos aquí, se actualizaron los datos o no hubo match exacto.
        # Continuamos con el flujo general para generar la respuesta contextual final.
        return await self._fallback_handler.handle(ctx)
