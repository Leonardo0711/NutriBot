"""
Nutribot Backend — Domain Entities
Dataclasses que representan las entidades de negocio.
Son objetos puros sin dependencia de infraestructura.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .value_objects import MessageType, SessionMode


# ──────────────────────────────────────────────
# Entidad: Usuario
# ──────────────────────────────────────────────
@dataclass
class User:
    id: int
    numero_whatsapp: str
    whatsapp_jid: Optional[str] = None
    creado_en: Optional[datetime] = None


# ──────────────────────────────────────────────
# Entidad: Estado Conversacional (persistida)
# ──────────────────────────────────────────────
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
    last_openai_response_id: Optional[str] = None
    onboarding_status: str = "not_started"
    onboarding_step: Optional[str] = None
    onboarding_last_invited_at: Optional[datetime] = None
    onboarding_next_eligible_at: Optional[datetime] = None
    onboarding_skip_count: int = 0
    onboarding_updated_at: Optional[datetime] = None
    version: int = 1
    updated_at: Optional[datetime] = None


# ──────────────────────────────────────────────
# Entidad: Mensaje Normalizado (transiente)
# Resultado de media_service.normalize()
# ──────────────────────────────────────────────
@dataclass
class NormalizedMessage:
    """Mensaje ya normalizado, listo para enviar al LLM."""
    provider_message_id: str
    phone: str
    content_type: MessageType
    text: str  # Texto original o transcripción STT
    image_base64: Optional[str] = None  # Base64 de la imagen para Vision
    used_audio: bool = False  # True si el input original era voz


# ──────────────────────────────────────────────
# Entidad: Extracción de Perfil (persistida)
# ──────────────────────────────────────────────
@dataclass
class ProfileExtraction:
    usuario_id: int
    field_code: str
    raw_value: str
    confidence: float
    evidence_text: str
    status: str = "tentative"
    id: Optional[int] = None
    extracted_at: Optional[datetime] = None


# ──────────────────────────────────────────────
# Entidad: Mensaje Webhook Crudo (transiente)
# Resultado de parse_evolution_webhook()
# ──────────────────────────────────────────────
@dataclass
class IncomingWebhookMessage:
    """Representación mínima parseada del webhook de Evolution."""
    provider_message_id: str
    phone: str
    content_type: MessageType
    text_body: Optional[str] = None
    media_url: Optional[str] = None
    media_mimetype: Optional[str] = None
