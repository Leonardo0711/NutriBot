"""Small presentation layer for more natural outbound WhatsApp replies."""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Optional

from config import get_settings


_TWILIO_MESSAGE_ID_RE = re.compile(r"\b(?:SM|MM)[A-Za-z0-9]{10,}\b")


@dataclass(frozen=True)
class HumanizedOutbound:
    text: str
    delay_seconds: float
    typing_message_id: Optional[str]


class ResponseHumanizer:
    """Adds controlled variation and humane pacing without changing intent."""

    _EXACT_VARIANTS: dict[str, tuple[str, ...]] = {
        "Entendido, seguimos conversando sin problema.": (
            "Entendido 😊 seguimos conversando sin problema.",
            "Claro, seguimos conversando sin problema 😊",
            "De acuerdo, continuamos sin problema.",
            "Perfecto, seguimos normal 😊",
        ),
        "Entendido, dejamos el formulario por ahora.": (
            "Entendido 😊 dejamos el formulario por ahora.",
            "No hay problema, lo dejamos por ahora.",
            "Claro, lo dejamos para otro momento 😊",
        ),
        "Muchas gracias por completar el formulario.": (
            "¡Muchas gracias por completar el formulario! 😊",
            "Gracias por completar el formulario, me ayuda mucho 😊",
            "¡Listo! Muchas gracias por completar el formulario.",
        ),
        "Listo 😊 ya completé los datos extra de tu perfil.": (
            "Listo 😊 ya completé esos datos extra de tu perfil.",
            "Perfecto, ya guardé esos datos extra de tu perfil 😊",
            "Gracias, ya actualicé esos datos extra de tu perfil.",
        ),
        "Listo 😊 ya completé esos datos extra de tu perfil.": (
            "Listo 😊 ya completé esos datos extra de tu perfil.",
            "Perfecto, ya guardé esos datos extra de tu perfil 😊",
            "Gracias, ya actualicé esos datos extra de tu perfil.",
        ),
        "Seguimos cuando quieras 😊": (
            "Seguimos cuando quieras 😊",
            "Cuando gustes, seguimos 😊",
            "Aquí estaré cuando quieras continuar 😊",
        ),
    }

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "humanize_outbox_enabled", True))

    @property
    def typing_enabled(self) -> bool:
        return bool(getattr(self._settings, "humanize_typing_enabled", True))

    def prepare(
        self,
        *,
        text: str,
        content_type: str,
        idempotency_key: str | None,
        phone: str | None,
    ) -> HumanizedOutbound:
        if not self.enabled:
            return HumanizedOutbound(text=text or "", delay_seconds=0.0, typing_message_id=None)

        key = f"{idempotency_key or ''}|{phone or ''}|{text or ''}"
        rng = random.Random(self._stable_seed(key))
        out_text = self._variant(text or "", rng) if content_type in {"text", "audio_tts"} else (text or "")
        delay = self._delay_seconds(out_text, content_type, rng)
        typing_id = self._typing_message_id(idempotency_key) if self.typing_enabled else None
        return HumanizedOutbound(text=out_text, delay_seconds=delay, typing_message_id=typing_id)

    @classmethod
    def _variant(cls, text: str, rng: random.Random) -> str:
        clean = (text or "").strip()
        variants = cls._EXACT_VARIANTS.get(clean)
        if not variants:
            return text
        return variants[rng.randrange(len(variants))]

    def _delay_seconds(self, text: str, content_type: str, rng: random.Random) -> float:
        max_delay = max(0.0, float(getattr(self._settings, "humanize_delay_max_seconds", 4.5)))
        if max_delay <= 0:
            return 0.0

        if content_type not in {"text", "interactive_buttons", "interactive_list"}:
            return min(max_delay, rng.uniform(0.2, 0.8))

        length = len((text or "").strip())
        if length <= 0:
            return 0.0
        if length <= 80:
            delay = rng.uniform(0.6, 1.5)
        elif length <= 260:
            delay = rng.uniform(1.1, 2.7)
        elif length <= 700:
            delay = rng.uniform(2.0, 4.0)
        else:
            delay = rng.uniform(3.0, 5.8)
        return round(min(max_delay, delay), 2)

    @staticmethod
    def _typing_message_id(idempotency_key: str | None) -> Optional[str]:
        if not idempotency_key:
            return None
        match = _TWILIO_MESSAGE_ID_RE.search(str(idempotency_key))
        return match.group(0) if match else None

    @staticmethod
    def _stable_seed(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:16], 16)
