"""
Nutribot Backend - Domain Entities
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .value_objects import MessageType, SessionMode


@dataclass
class User:
    id: int
    numero_whatsapp: str
    whatsapp_jid: Optional[str] = None
    creado_en: Optional[datetime] = None


@dataclass
class ConversationState:
    usuario_id: int
    mode: str = SessionMode.ACTIVE_CHAT.value
    awaiting_field_code: Optional[str] = None
    awaiting_question_code: Optional[str] = None
    last_provider_message_id: Optional[str] = None
    last_turn_at: Optional[datetime] = None
    last_form_prompt_at: Optional[datetime] = None
    turns_since_last_prompt: int = 0
    closure_score: Optional[int] = None
    reply_resolved_something: bool = False
    profile_completion_pct: int = 0
    usability_completion_pct: int = 0
    meaningful_interactions_count: int = 0
    last_openai_response_id: Optional[str] = None
    onboarding_status: str = "not_started"
    onboarding_step: Optional[str] = None
    onboarding_last_invited_at: Optional[datetime] = None
    onboarding_next_eligible_at: Optional[datetime] = None
    onboarding_skip_count: int = 0
    onboarding_updated_at: Optional[datetime] = None
    version: int = 1
    updated_at: Optional[datetime] = None


@dataclass
class NormalizedMessage:
    provider_message_id: str
    phone: str
    content_type: MessageType
    text: str
    image_base64: Optional[str] = None
    image_mimetype: Optional[str] = None
    used_audio: bool = False
    interactive_id: Optional[str] = None


@dataclass
class ProfileExtraction:
    usuario_id: int
    field_code: str
    raw_value: str
    confidence: float
    evidence_text: str
    status: str = "tentative"
    normalized_value: Optional[str] = None
    resolved_entity_type: Optional[str] = None
    resolved_entity_code: Optional[str] = None
    resolution_strategy: Optional[str] = None
    semantic_cache_hit: bool = False
    id: Optional[int] = None
    extracted_at: Optional[datetime] = None


@dataclass
class IncomingWebhookMessage:
    provider_message_id: str
    phone: str
    content_type: MessageType
    text_body: Optional[str] = None
    media_url: Optional[str] = None
    media_mimetype: Optional[str] = None
    interactive_id: Optional[str] = None
    interactive_text: Optional[str] = None

