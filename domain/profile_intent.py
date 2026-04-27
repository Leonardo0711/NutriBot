# domain/profile_intent.py
"""
Nutribot Backend — Profile Intent Domain Objects
Dataclasses que representan la intención de actualización de perfil
extraída del mensaje del usuario.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileIntentValue:
    """Un valor individual detectado en la intención de perfil."""
    raw_value: str
    normalized_value: str | None = None
    entity_type: str | None = None
    entity_code: str | None = None
    entity_label: str | None = None
    resolution_strategy: str | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProfileIntentResult:
    """
    Resultado de la extracción de intención de perfil.
    Encapsula si el mensaje es una actualización, qué campo,
    qué operación y los valores detectados.
    """
    is_profile_update: bool = False
    field_code: str | None = None
    operation: str | None = None  # ADD, REMOVE, REPLACE, CLEAR, CORRECTION, HISTORICAL_UPDATE, NOOP
    values: list[ProfileIntentValue] = field(default_factory=list)
    confidence: float = 0.0
    evidence_text: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    source: str = "NONE"  # EXPECTED_FIELD, FAST_NUMERIC, LLM_STRUCTURED, ROUTER_HINT

    @property
    def has_values(self) -> bool:
        return bool(self.values)

    @property
    def is_confident(self) -> bool:
        return self.is_profile_update and self.confidence >= 0.75 and not self.needs_clarification
