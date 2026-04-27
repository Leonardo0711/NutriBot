"""
Nutribot Backend - Turn Context
Encapsula todo el contexto de una interacción para que los Handlers puedan procesarlo.
"""
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import User, ConversationState, NormalizedMessage
from domain.profile_snapshot import ProfileSnapshot
from domain.router import RouteResult
from domain.profile_intent import ProfileIntentResult

@dataclass
class TurnContext:
    session: AsyncSession
    user: User
    state: ConversationState
    state_snapshot: ConversationState
    normalized: NormalizedMessage
    route: RouteResult
    
    # Datos de Perfil y de Memoria
    history: list[dict]
    snapshot: ProfileSnapshot
    profile_text: str
    summary: str
    
    # Contexto RAG para consultas generales
    rag_text: Optional[str] = None
    
    # Contexto nutricional derivado de reglas clínicas (tablas rel_*)
    nutritional_rules_text: Optional[str] = None
    
    # Flags precomputados por el enrutador / reglas
    looks_like_profile_update: bool = False
    is_asking_for_recommendation: bool = False
    is_short_greeting: bool = False
    is_requesting_personalization: bool = False
    is_requesting_survey: bool = False
    
    # Mutaciones de contexto
    has_absurd_profile_claim: bool = False
    onboarding_interception_happened: bool = False
    extracted_data: dict = field(default_factory=dict)
    
    # ── Intent Extraction Layer ──
    profile_intent: Optional[ProfileIntentResult] = None
    turn_kind: Optional[str] = None  # PROFILE_MAINTENANCE, ONBOARDING_RESPONSE, CONVERSATIONAL, etc.
    
    # Tracking de operaciones (para memoria delegada si fuera necesario)
    bot_reply_text: Optional[str] = field(default=None, init=False)


