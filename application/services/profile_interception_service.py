"""
Nutribot Backend - Profile Interception Service
"""
from __future__ import annotations

from typing import Optional

from application.services.conversation_state_service import ConversationStateService
from application.services.onboarding_service import ONBOARDING_QUESTIONS, OnboardingService
from application.services.profile_context_service import ProfileContextService
from domain.entities import ConversationState
from domain.profile_snapshot import ProfileSnapshot
from domain.value_objects import ONBOARDING_PHASE_2, OnboardingStatus


class ProfileInterceptionService:
    def __init__(
        self,
        onboarding_service: OnboardingService,
        profile_context: ProfileContextService,
        state_service: ConversationStateService,
    ):
        self._onboarding_service = onboarding_service
        self._profile_context = profile_context
        self._state_service = state_service

    async def maybe_start_personalization_flow(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        summary: str,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        is_requesting_personalization: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], bool]:
        if onboarding_interception_happened or not is_requesting_personalization:
            return reply, onboarding_interception_happened

        next_step = await self._onboarding_service._find_next_missing_step(
            session,
            user_id,
            ignore_skips=True,
            treat_ninguna_as_missing=False,
            phase=[s for s in ONBOARDING_PHASE_2] if state.onboarding_status == OnboardingStatus.COMPLETED.value else None,
        )
        if next_step:
            pending_fields = self._profile_context.pending_fields(snapshot)
            pending_line = ", ".join(pending_fields) if pending_fields else "ninguno"
            step_label = self._profile_context.human_step_label(next_step)
            reply = (
                "Claro 😊 personalicemos tus recomendaciones.\n\n"
                f"Tengo registrado:\n{summary}\n\n"
                f"Nos falta: {pending_line}.\n\n"
                f"Empezamos por confirmar tu {step_label}?"
            )
            self._state_service.set_onboarding_in_progress(state, next_step)
            return reply, True

        if not is_asking_for_recommendation:
            reply = (
                "Ya tengo tu perfil completo 😊\n\n"
                f"{summary}\n\n"
                "Si quieres cambiar algun dato especifico (como tu peso o talla), solo dimelo directamente en cualquier momento."
            )
        return reply, onboarding_interception_happened

    async def maybe_intercept_for_missing_profile(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        reply: Optional[str],
        onboarding_interception_happened: bool,
        is_short_greeting: bool,
        is_asking_for_recommendation: bool,
    ) -> tuple[Optional[str], bool]:
        should_check = (
            not onboarding_interception_happened
            and state.onboarding_status != OnboardingStatus.COMPLETED.value
            and (is_short_greeting or is_asking_for_recommendation)
        )
        if not should_check:
            return reply, onboarding_interception_happened

        if is_asking_for_recommendation:
            missing_essential = self._profile_context.missing_essential_fields(snapshot)
            if not missing_essential:
                return reply, onboarding_interception_happened

            intro = "Claro, me encantaria darte una recomendacion a tu medida."
            known_parts = []
            if snapshot.measurements.age_years is not None:
                known_parts.append(f"Edad: {snapshot.measurements.age_years} anos")
            if snapshot.measurements.weight_kg is not None:
                known_parts.append(f"Peso: {snapshot.measurements.weight_kg}kg")
            if snapshot.measurements.height_cm is not None:
                h = snapshot.measurements.height_cm
                h_str = f"{h/100:.2f}m" if h > 10 else f"{h:.2f}m"
                known_parts.append(f"Talla: {h_str}")
            if snapshot.health.allergies:
                known_parts.append(f"Alergias: {', '.join(snapshot.health.allergies)}")
            if known_parts:
                intro += f" Ya tengo algunos datos: {', '.join(known_parts)}."

            missing_step = await self._onboarding_service._find_next_missing_step(session, user_id, phase=None)
            if not missing_step:
                return reply, onboarding_interception_happened

            step_name = "talla (estatura)" if missing_step == "altura_cm" else ("peso" if missing_step == "peso_kg" else missing_step)
            if "peso_kg" in missing_essential or "altura_cm" in missing_essential:
                reply = (
                    f"{intro}\n\n"
                    "Pero para que mi sugerencia sea 100% precisa y calcular tu IMC, "
                    f"solo me faltaria completar un par de datos mas. Te parece si empezamos por tu {step_name}?"
                )
            else:
                reply = (
                    f"{intro}\n\n"
                    "Solo me faltaria completar un pequeno detalle para ser mas preciso. "
                    f"Te parece si confirmamos tu {step_name}?"
                )

            self._state_service.set_onboarding_in_progress(state, missing_step)
            return reply, True

        if is_short_greeting:
            if state.onboarding_status == OnboardingStatus.NOT_STARTED.value:
                reply = (
                    "Hola 😊 Soy NutriBot, tu asistente de nutricion de EsSalud 🍏.\n\n"
                    "Puedo ayudarte con tips y recomendaciones segun tu perfil.\n\n"
                    "Si quieres, armamos tu perfil basico con 5 datos rapidos "
                    "(edad, peso, talla, alergias y objetivo). ¿Empezamos?"
                )
            else:
                reply = (
                    "Hola de nuevo 😊\n\n"
                    "Si te parece, completamos los datos que faltan para personalizar mejor tus recomendaciones.\n\n"
                    "¿Te animas? Es rapidito 🍏"
                )
            self._state_service.set_onboarding_invited(state)
            return reply, True

        return reply, onboarding_interception_happened

    async def maybe_suggest_phase2_field(
        self,
        *,
        session,
        state: ConversationState,
        user_id: int,
        snapshot: ProfileSnapshot,
        reply: Optional[str],
    ) -> Optional[str]:
        """Si el onboarding basico (Phase 1) ya esta completo, sugiere 1 dato de Phase 2 cada ~4 turnos utiles."""
        if state.onboarding_status != OnboardingStatus.COMPLETED.value:
            return reply
        if not reply:
            return reply
        if state.turns_since_last_prompt < 4:
            return reply

        next_phase2 = await self._onboarding_service._find_next_missing_step(
            session,
            user_id,
            phase=[s for s in ONBOARDING_PHASE_2],
            start_from_idx=0,
        )
        if not next_phase2:
            return reply

        step_label = self._profile_context.human_step_label(next_phase2)
        question = ONBOARDING_QUESTIONS.get(next_phase2, "")

        self._state_service.set_turns_since_last_prompt(state, 0)

        suggestion = (
            f"\n\nPor cierto, para que mis orientaciones sean aun mas precisas, "
            f"me podrias compartir tu {step_label}? "
            f"{question}\n"
            f"(Si prefieres no decirlo, no hay problema, solo ignora este mensaje.)"
        )
        return reply + suggestion
