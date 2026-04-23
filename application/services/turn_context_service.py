"""
Nutribot Backend - TurnContextService
Construye el contexto de turno antes de delegar a un handler.
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import User, ConversationState, NormalizedMessage
from domain.profile_snapshot import ProfileSnapshot
from domain.router import RouteResult, Intent
from domain.turn_context import TurnContext
from application.services.profile_read_service import ProfileReadService
from application.services.profile_context_service import ProfileContextService
from application.services.nutritional_rules_service import NutritionalRulesService
from application.services.conversation_memory_service import ConversationMemoryService


class TurnContextService:
    # Intents que se benefician de contexto nutricional derivado de reglas
    _NUTRITION_INTENTS = {
        Intent.NUTRITION_QUERY,
        Intent.RECOMMENDATION_REQUEST,
        Intent.IMAGE,
        Intent.AUDIO,
        Intent.AMBIGUOUS,
    }

    def __init__(
        self,
        profile_reader: ProfileReadService,
        profile_context: ProfileContextService,
        memory_service: ConversationMemoryService,
        nutritional_rules: NutritionalRulesService | None = None,
    ):
        self._profile_reader = profile_reader
        self._profile_context = profile_context
        self._memory_service = memory_service
        self._nutritional_rules = nutritional_rules or NutritionalRulesService()

    async def build(
        self,
        session: AsyncSession,
        user: User,
        state: ConversationState,
        state_snapshot: ConversationState,
        normalized: NormalizedMessage,
        route: RouteResult,
        rag_text: Optional[str] = None
    ) -> TurnContext:
        """Construye y ensambla todo el contexto necesario para el turno."""
        
        # 1. Cargar Snapshot
        snapshot = await self._profile_reader.fetch_snapshot(session, user.id)
        if not snapshot:
            snapshot = ProfileSnapshot.from_row({"usuario_id": user.id, "skipped_fields": {}})

        # 2. Reconstruir Prompt Text y Summary
        profile_text, summary = self._profile_context.build_prompt_and_summary(snapshot)

        # 3. Cargar History
        history = await self._memory_service.load_recent_history(session, user.id)
        
        # 4. Derivar Flags
        looks_like_profile_update = route.intent in (
            Intent.PROFILE_UPDATE,
            Intent.CORRECTION_PAST_FIELD,
            Intent.ANSWER_CURRENT_STEP,
        )
        is_asking_for_recommendation = route.intent in (
            Intent.NUTRITION_QUERY,
            Intent.RECOMMENDATION_REQUEST,
        )
        is_short_greeting = route.intent == Intent.GREETING
        is_requesting_personalization = route.intent == Intent.PERSONALIZE_REQUEST
        is_requesting_survey = route.intent == Intent.SURVEY_CONTINUE

        # 5. Cargar contexto de reglas nutricionales (solo para intents relevantes)
        nutritional_rules_text = None
        if route.intent in self._NUTRITION_INTENTS:
            rules_ctx = await self._nutritional_rules.resolve_nutritional_context(session, user.id)
            nutritional_rules_text = self._nutritional_rules.build_rules_prompt_context(rules_ctx)
            if nutritional_rules_text:
                profile_text = f"{profile_text}\n\n{nutritional_rules_text}"

        return TurnContext(
            session=session,
            user=user,
            state=state,
            state_snapshot=state_snapshot,
            normalized=normalized,
            route=route,
            history=history,
            snapshot=snapshot,
            profile_text=profile_text,
            summary=summary,
            rag_text=rag_text,
            nutritional_rules_text=nutritional_rules_text,
            looks_like_profile_update=looks_like_profile_update,
            is_asking_for_recommendation=is_asking_for_recommendation,
            is_short_greeting=is_short_greeting,
            is_requesting_personalization=is_requesting_personalization,
            is_requesting_survey=is_requesting_survey,
        )

