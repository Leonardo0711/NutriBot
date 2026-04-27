"""
Nutribot Backend - SurveyService
Guided usability survey with structured interactive responses.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import timedelta
from typing import Optional, Any

from openai import AsyncOpenAI
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState, NormalizedMessage
from domain.reply_objects import BotReply
from domain.value_objects import MessageType, OnboardingStatus
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

CONSENT_STATE = "esperando_consentimiento_encuesta"

FORM_STATES_ORDER = [
    "esperando_correo",
    "esperando_asegurado_essalud",
    "esperando_p1",
    "esperando_p2",
    "esperando_p3",
    "esperando_p4",
    "esperando_p5",
    "esperando_p6",
    "esperando_p7",
    "esperando_audio_optin",
    "esperando_audio_prueba",
    "esperando_p8",
    "esperando_imagen_optin",
    "esperando_imagen_prueba",
    "esperando_p9",
    "esperando_p10",
    "esperando_nps",
    "esperando_comentario",
    "esperando_autorizacion",
]

FORM_QUESTIONS: dict[str, str] = {
    "esperando_correo": (
        "Si te parece, comparte tu correo."
    ),
    "esperando_asegurado_essalud": "¿Se encuentra asegurado en EsSalud?",
    "esperando_p1": "En una escala del 1 al 5, ¿considera que mi personalidad como chatbot resulta realista y atractiva?",
    "esperando_p2": "En una escala del 1 al 5, ¿diría que la explicación sobre mi propósito y alcance fue suficiente y comprensible?",
    "esperando_p3": "En una escala del 1 al 5, ¿le resultó sencillo navegar y utilizar mis funciones?",
    "esperando_p4": "En una escala del 1 al 5, ¿considera que comprendí adecuadamente sus consultas?",
    "esperando_p5": "En una escala del 1 al 5, desde su experiencia, ¿mis respuestas le fueron útiles, apropiadas e informativas?",
    "esperando_p6": "En una escala del 1 al 5, ¿percibió que traté de forma adecuada cualquier error o equivocación?",
    "esperando_p7": "En una escala del 1 al 5, ¿le resultó fácil y sencillo interactuar conmigo?",
    "esperando_audio_optin": "Por si acaso: tambien tengo modo audio 🎧.",
    "esperando_audio_prueba": "Por si acaso: tambien tengo modo audio 🎧.",
    "esperando_p8": "En una escala del 1 al 5, ¿considera que mis respuestas en audio fueron claras?",
    "esperando_imagen_optin": "Por si acaso: tambien tengo reconocimiento de imagen 🖼️.",
    "esperando_imagen_prueba": "Por si acaso: tambien tengo reconocimiento de imagen 🖼️.",
    "esperando_p9": "En una escala del 1 al 5, ¿logré interpretar correctamente lo que muestra la imagen que me enviaste?",
    "esperando_p10": "En una escala del 1 al 5, ¿me enfoqué únicamente en responderte preguntas sobre nutrición?",
    "esperando_nps": "En una escala del 1 al 10, ¿qué tan probable es que me recomiendes a un amigo o familiar?",
    "esperando_comentario": "En esta primera interacción, ¿qué te gustó o no te gustó de mí?",
    "esperando_autorizacion": "¿Autorizas el uso anónimo y agregado de tus respuestas para investigación? (Si autorizo / No autorizo)",
}

EMAIL_REASK_COPY = (
    "Tu correo ayuda a enviarte avisos de campañas de salud y nutrición cerca de ti.\n\n"
    "Si deseas, compártelo ahora. Si no, escribe *no* y seguimos con la encuesta."
)

_SCALE_PREFIX = {
    "esperando_p1": "survey:p1",
    "esperando_p2": "survey:p2",
    "esperando_p3": "survey:p3",
    "esperando_p4": "survey:p4",
    "esperando_p5": "survey:p5",
    "esperando_p6": "survey:p6",
    "esperando_p7": "survey:p7",
    "esperando_p8": "survey:p8",
    "esperando_p9": "survey:p9",
    "esperando_p10": "survey:p10",
    "esperando_nps": "survey:nps",
}


class SurveyResponseExtractor:
    """Structured-first extractor for survey answers."""

    _SKIP_EXACT = {"paso", "saltar", "skip", "no quiero", "siguiente", "prefiero no", "omitir", "nopi"}
    _CANCEL_PHRASES = {"cancelar todo", "detener bot", "salir de todo", "stop survey"}
    _YES_WORDS = re.compile(r"(?:\b|^)(si|sí|sii|sip|yes|claro|dale|ok|de acuerdo|acepto|autorizo|ya|yap|yaa)(?:\b|$)", re.IGNORECASE)
    _NO_WORDS = re.compile(r"(?:\b|^)(no|nop|nopi|noo|nooo|nel|rechazo|no autorizo)(?:\b|$)", re.IGNORECASE)
    _WHY_PATTERN = re.compile(r"(para que|por que|que uso|que finalidad)", re.IGNORECASE)
    _YES_SHORT_EXACT = {
        "si", "sÃ­", "sii", "sip", "yes", "claro", "dale", "ok", "okay", "de acuerdo",
        "acepto", "autorizo", "si claro", "si normal", "si acepto", "si autorizo",
        "ya", "yap", "yaa",
    }
    _NO_SHORT_EXACT = {
        "no", "nop", "nopi", "noo", "nooo", "nel", "rechazo", "no autorizo",
        "no gracias", "no deseo", "no quiero",
    }
    _ACK_TOKENS = {
        "si", "sÃ­", "no", "ok", "okay", "dale", "claro", "gracias", "listo", "lista",
        "perfecto", "perfecta", "entendido", "entendida", "yap", "ya", "yaa",
        "jaja", "jajaja", "jajajaja", "de", "acuerdo",
    }
    _QUERY_TOKENS = {
        "como", "que", "cual", "cuando", "donde", "por", "porque", "para", "sobre",
        "puedo", "debo", "quiero", "seria", "mejor", "explicame", "explica", "dame",
        "ayudame", "ayuda",
    }
    _NUTRITION_HINTS = (
        "menu", "receta", "dieta", "imc", "calorias", "desayuno", "almuerzo",
        "cena", "peso", "talla", "proteina", "carbohidratos", "grasa",
        "nutricion", "plan alimentario", "porciones", "macros",
    )
    _USEFUL_CHAT_HINTS = (
        "quiero", "necesito", "ayudame", "ayudame con", "me ayudas", "me recomiendas",
        "recomendacion", "recomendaciones", "como bajo", "como subir", "como mejorar",
        "que puedo comer", "que debo comer", "puedo comer", "entrenamiento", "ejercicio",
        "sobrepeso", "obesidad", "azucar", "glucosa", "presion", "hipertension",
    )
    _QUESTION_STARTS = (
        "como", "que", "cual", "cuanto", "puedo", "debo", "quiero", "me ayudas",
        "podrias", "podrías", "necesito",
    )
    _GENERAL_CHAT_MARKERS = (
        "sobre ",
        "y sobre ",
        "entonces ",
        "por cierto ",
        "ademas ",
        "tambien ",
    )
    _INTERRUPTIBLE_STATES = {
        "esperando_asegurado_essalud",
        "esperando_p1",
        "esperando_p2",
        "esperando_p3",
        "esperando_p4",
        "esperando_p5",
        "esperando_p6",
        "esperando_p7",
        "esperando_p8",
        "esperando_p9",
        "esperando_p10",
        "esperando_nps",
        "esperando_comentario",
        "esperando_autorizacion",
        "esperando_audio_optin",
        "esperando_audio_prueba",
        "esperando_imagen_optin",
        "esperando_imagen_prueba",
    }
    _MEDIA_READY_WORDS = {
        "listo",
        "lista",
        "ok",
        "okay",
        "dale",
        "continuar",
        "continua",
        "continuemos",
        "pasemos",
        "ya",
        "yap",
    }
    _NUMBER_WORDS = {
        "uno": 1,
        "una": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
    }

    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self._client = openai_client
        self._model = model

    @staticmethod
    def _contains_nutrition_hint(text: str) -> bool:
        low = (text or "").lower()
        for hint in SurveyResponseExtractor._NUTRITION_HINTS:
            if " " in hint and hint in low:
                return True
            if " " not in hint and re.search(rf"\b{re.escape(hint)}\b", low):
                return True
        return False

    @classmethod
    def _looks_like_useful_chat_question(cls, text: str) -> bool:
        low = (text or "").strip().lower()
        if not low:
            return False
        if cls._extract_binary_answer(low) is not None:
            return False

        if cls._contains_nutrition_hint(low):
            return True

        has_question_shape = ("?" in low) or any(low.startswith(prefix + " ") for prefix in cls._QUESTION_STARTS)
        has_useful_hint = any(hint in low for hint in cls._USEFUL_CHAT_HINTS)
        has_general_chat_marker = any(marker in low for marker in cls._GENERAL_CHAT_MARKERS)

        if has_question_shape and has_useful_hint:
            return True
        # Preguntas largas de chat (aunque no empiecen con "como/que" ni lleven signos)
        # se tratan como interrupcion para responder primero y luego retomar encuesta.
        if has_useful_hint and len(low.split()) >= 4:
            return True
        if has_question_shape and len(low.split()) >= 5:
            return True
        if has_general_chat_marker and len(low.split()) >= 6:
            return True
        if cls._looks_like_free_form_interrupt(low):
            return True
        return False

    @classmethod
    def _extract_binary_answer(cls, text: str) -> Optional[str]:
        token = cls._normalize_token(text)
        if not token:
            return None
        if token in cls._YES_SHORT_EXACT:
            return "Si"
        if token in cls._NO_SHORT_EXACT:
            return "No"
        return None

    @classmethod
    def _looks_like_free_form_interrupt(cls, text: str) -> bool:
        token = cls._normalize_token(text)
        if not token:
            return False
        words = token.split()
        if len(words) < 5:
            return False
        meaningful = [w for w in words if w not in cls._ACK_TOKENS]
        if len(meaningful) < 4:
            return False
        if "?" in text:
            return True
        if any(w in cls._QUERY_TOKENS for w in meaningful):
            return True
        return False

    @staticmethod
    def _normalize_token(value: str) -> str:
        txt = unicodedata.normalize("NFKD", str(value or ""))
        txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
        txt = re.sub(r"[^a-z0-9\s]", " ", txt.lower())
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    @classmethod
    def _extract_scale_value(cls, raw_text: str) -> Optional[int]:
        v = (raw_text or "").strip()
        if not v:
            return None
        m = re.search(r"\b(\d{1,2})\b", v)
        if m:
            return int(m.group(1))
        norm = cls._normalize_token(v)
        if not norm:
            return None
        for token in norm.split():
            val = cls._NUMBER_WORDS.get(token)
            if val is not None:
                return val
        return None

    def extract_structured(
        self,
        state_name: str,
        interactive_id: Optional[str],
        user_text: str,
    ) -> Optional[dict]:
        if not interactive_id:
            return None

        scale_prefix = _SCALE_PREFIX.get(state_name)
        if scale_prefix and interactive_id.startswith(scale_prefix + ":"):
            value = interactive_id.split(":")[-1]
            return {"intent": "ANSWER", "value": value}

        if state_name == "esperando_audio_optin":
            if interactive_id == "survey:audio_optin:yes":
                return {"intent": "ANSWER", "value": "yes"}
            if interactive_id == "survey:audio_optin:no":
                return {"intent": "ANSWER", "value": "no"}

        if state_name == "esperando_imagen_optin":
            if interactive_id == "survey:image_optin:yes":
                return {"intent": "ANSWER", "value": "yes"}
            if interactive_id == "survey:image_optin:no":
                return {"intent": "ANSWER", "value": "no"}

        if state_name == "esperando_autorizacion":
            if interactive_id == "survey:auth:yes":
                return {"intent": "ANSWER", "value": "Si"}
            if interactive_id == "survey:auth:no":
                return {"intent": "ANSWER", "value": "No"}

        if state_name == CONSENT_STATE:
            if interactive_id == "survey:consent:yes":
                return {"intent": "ANSWER", "value": "Si"}
            if interactive_id == "survey:consent:no":
                return {"intent": "ANSWER", "value": "No"}

        return None

    def try_fast_extract(self, state_name: str, user_text: str) -> Optional[dict]:
        v = (user_text or "").strip()
        vl = v.lower()
        if vl in self._CANCEL_PHRASES:
            return {"intent": "CANCEL", "value": None}
        if vl in self._SKIP_EXACT:
            return {"intent": "SKIP", "value": None}
        if self._WHY_PATTERN.search(vl):
            return {"intent": "WHY", "value": None}

        if state_name == "esperando_correo":
            if self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl):
                return {"intent": "INTERRUPT", "value": None}
            if self._NO_WORDS.search(vl):
                return {"intent": "SKIP", "value": None}
            match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", v)
            if match:
                return {"intent": "ANSWER", "value": match.group(0)}
            return None

        if state_name == CONSENT_STATE and (
            self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl)
        ):
            return {"intent": "INTERRUPT", "value": None}

        if state_name in self._INTERRUPTIBLE_STATES and (
            self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl)
        ):
            return {"intent": "INTERRUPT", "value": None}

        if state_name in {"esperando_audio_prueba", "esperando_imagen_prueba"}:
            if self._NO_WORDS.search(vl):
                return {"intent": "SKIP", "value": None}
            token = SurveyService._normalize_token(vl)
            if token in self._MEDIA_READY_WORDS:
                return {"intent": "ANSWER", "value": "LISTO"}
            if "listo" in vl or "continu" in vl or "pasemos" in vl:
                return {"intent": "ANSWER", "value": "LISTO"}
            return None

        if state_name in ("esperando_asegurado_essalud", "esperando_audio_optin", "esperando_imagen_optin", "esperando_autorizacion", CONSENT_STATE):
            if self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl):
                return {"intent": "INTERRUPT", "value": None}
            binary = self._extract_binary_answer(vl)
            if binary == "No":
                return {"intent": "ANSWER", "value": "No"}
            if binary == "Si":
                return {"intent": "ANSWER", "value": "Si"}
            return None

        if state_name.startswith("esperando_p"):
            score = self._extract_scale_value(v)
            if score is None:
                if self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl):
                    return {"intent": "INTERRUPT", "value": None}
                return {"intent": "ANSWER", "value": v}
            if 1 <= score <= 5:
                return {"intent": "ANSWER", "value": str(score)}
            return {"intent": "ANSWER", "value": v}

        if state_name == "esperando_nps":
            score = self._extract_scale_value(v)
            if score is None:
                if self._looks_like_useful_chat_question(vl) or self._looks_like_free_form_interrupt(vl):
                    return {"intent": "INTERRUPT", "value": None}
                return {"intent": "ANSWER", "value": v}
            if 1 <= score <= 10:
                return {"intent": "ANSWER", "value": str(score)}
            return {"intent": "ANSWER", "value": v}

        if state_name == "esperando_comentario":
            if self._looks_like_useful_chat_question(vl):
                return {"intent": "INTERRUPT", "value": None}
            if vl in {"no", "nada", "ninguno"}:
                return {"intent": "ANSWER", "value": None}
            return {"intent": "ANSWER", "value": v}

        return None

    async def extract(self, state_name: str, user_text: str, interactive_id: Optional[str]) -> dict:
        structured = self.extract_structured(state_name, interactive_id, user_text)
        if structured is not None:
            return structured

        fast = self.try_fast_extract(state_name, user_text)
        if fast is not None:
            return fast

        # Fallback LLM only when deterministic parser is not enough.
        if not self._client:
            return {"intent": "ANSWER", "value": user_text}
        try:
            prompt = (
                f'Respuesta usuario a estado "{state_name}": "{user_text}". '
                'Clasifica en ANSWER, WHY, SKIP, INTERRUPT o CANCEL. '
                'Reglas: si el estado espera numero (p1-p10/nps) y el mensaje no es claramente un numero valido, '
                'pero contiene una consulta o pedido de ayuda, clasifica como INTERRUPT. '
                'Si el usuario cambia de tema o pide consejo/comida/ejercicio/salud, clasifica INTERRUPT. '
                'Retorna JSON con keys intent y value.'
            )
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "Analista de encuestas."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            logger.exception("Survey extractor fallback failed")
            return {"intent": "ANSWER", "value": user_text}


class SurveyService:
    REINVITE_COOLDOWN = timedelta(hours=6)
    _FEATURE_STATES = {
        "esperando_audio_optin",
        "esperando_audio_prueba",
        "esperando_imagen_optin",
        "esperando_imagen_prueba",
    }
    _NUMBER_WORDS = {
        "cero": 0,
        "uno": 1,
        "una": 1,
        "dos": 2,
        "tres": 3,
        "cuatro": 4,
        "cinco": 5,
        "seis": 6,
        "siete": 7,
        "ocho": 8,
        "nueve": 9,
        "diez": 10,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def __init__(self, openai_client: AsyncOpenAI, openai_model: str):
        self.extractor = SurveyResponseExtractor(openai_client, openai_model)

    @staticmethod
    def _normalize_token(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        base = unicodedata.normalize("NFKD", raw)
        without_accents = "".join(ch for ch in base if not unicodedata.combining(ch))
        return without_accents.lower().strip()

    @staticmethod
    def _parse_int(value: Any, minimum: int, maximum: int) -> Optional[int]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        digit_match = re.search(r"\b(\d{1,2})\b", raw)
        if digit_match:
            number = int(digit_match.group(1))
            if minimum <= number <= maximum:
                return number
            return None

        normalized = SurveyService._normalize_token(raw)
        if normalized in SurveyService._NUMBER_WORDS:
            number = SurveyService._NUMBER_WORDS[normalized]
            if minimum <= number <= maximum:
                return number
            return None

        for token in re.split(r"[\s,.;:!?/\\-]+", normalized):
            if not token:
                continue
            mapped = SurveyService._NUMBER_WORDS.get(token)
            if mapped is None:
                continue
            if minimum <= mapped <= maximum:
                return mapped
            return None

        try:
            number = int(raw)
            if minimum <= number <= maximum:
                return number
        except Exception:
            return None
        return None

    @classmethod
    def _is_profile_capture_active(cls, state: ConversationState) -> bool:
        if state.awaiting_field_code:
            return True
        if state.onboarding_status == OnboardingStatus.IN_PROGRESS.value:
            return True
        if state.onboarding_step and state.onboarding_status != OnboardingStatus.COMPLETED.value:
            return True
        return False

    @classmethod
    def _is_reinvite_cooldown_active(cls, state: ConversationState) -> bool:
        if not state.last_form_prompt_at:
            return False
        now_peru = get_now_peru()
        try:
            delta = now_peru - state.last_form_prompt_at
        except Exception:
            return False
        return delta < cls.REINVITE_COOLDOWN

    @staticmethod
    def _normalize_email(value: Any) -> Optional[str]:
        if not value:
            return None
        candidate = str(value).strip()
        if re.search(r"^[\w\.-]+@[\w\.-]+\.\w+$", candidate):
            return candidate
        return None

    @staticmethod
    def _normalize_auth(value: Any) -> Optional[bool]:
        if value is None:
            return None
        text_value = SurveyService._normalize_token(value)
        if text_value in {"si", "yes", "si autorizo", "autorizo", "acepto"}:
            return True
        if text_value in {"no", "no autorizo", "rechazo"}:
            return False
        return None

    @staticmethod
    def _wants_finish_media_test(value: Any, raw_text: str) -> bool:
        token = SurveyService._normalize_token(value if value is not None else raw_text)
        if token in {"listo", "lista", "ok", "okay", "dale", "continuar", "continua", "continuemos", "pasemos", "ya", "yap"}:
            return True
        low = str(raw_text or "").lower()
        return ("listo" in low) or ("continu" in low) or ("pasemos" in low)

    @staticmethod
    def _is_audio_message(normalized: NormalizedMessage) -> bool:
        return normalized.content_type in (MessageType.AUDIO, MessageType.PTT) or bool(normalized.used_audio)

    @staticmethod
    def _is_image_message(normalized: NormalizedMessage) -> bool:
        return normalized.content_type == MessageType.IMAGE or bool(normalized.image_base64)

    @staticmethod
    def _is_multimedia_message(normalized: NormalizedMessage) -> bool:
        return (
            normalized.content_type in (MessageType.IMAGE, MessageType.AUDIO, MessageType.PTT)
            or bool(normalized.image_base64)
            or bool(normalized.used_audio)
        )

    def _is_state_answered(self, state_name: str, parciales: dict) -> bool:
        field_key = state_name.replace("esperando_", "")
        if state_name == "esperando_correo":
            if parciales.get("correo"):
                return True
            return bool(parciales.get("correo_opt_out_final"))
        if state_name == "esperando_asegurado_essalud":
            return self._normalize_auth(parciales.get("asegurado_essalud")) is not None
        if state_name == "esperando_autorizacion":
            return self._normalize_auth(parciales.get("autorizacion")) is not None
        if state_name.startswith("esperando_p"):
            return self._parse_int(parciales.get(field_key), 1, 5) is not None
        if state_name == "esperando_nps":
            return self._parse_int(parciales.get("nps"), 1, 10) is not None
        if state_name == "esperando_comentario":
            return "comentario" in parciales
        return False

    def _next_unanswered_state_from(self, start_state: Optional[str], parciales: dict) -> Optional[str]:
        if not start_state:
            return None
        try:
            start_idx = FORM_STATES_ORDER.index(start_state)
        except ValueError:
            return start_state
        for state_name in FORM_STATES_ORDER[start_idx:]:
            if state_name in self._FEATURE_STATES:
                return state_name
            if not self._is_state_answered(state_name, parciales):
                return state_name
        return None

    def _missing_media_feedback_states(self, parciales: dict) -> list[str]:
        pending: list[str] = []
        if self._parse_int(parciales.get("p8"), 1, 5) is None:
            pending.append("esperando_audio_optin")
        if self._parse_int(parciales.get("p9"), 1, 5) is None:
            pending.append("esperando_imagen_optin")
        return pending

    @staticmethod
    def _merge_prefix(*parts: str) -> str:
        clean_parts: list[str] = []
        seen: set[str] = set()
        for part in parts:
            txt = str(part or "").strip()
            if not txt:
                continue
            key = SurveyService._normalize_token(txt)
            if key in seen:
                continue
            seen.add(key)
            clean_parts.append(txt)
        return "\n\n".join(clean_parts)

    def _advance_feature_state_by_usage(
        self,
        *,
        current_state: str,
        parciales: dict,
        uso_audio: bool,
        uso_imagen: bool,
    ) -> tuple[Optional[str], str]:
        if current_state in {"esperando_audio_optin", "esperando_audio_prueba"}:
            if uso_audio:
                return (
                    self._next_unanswered_state_from("esperando_p8", parciales),
                    "Excelente 🙌 como ya usaste el modo audio, ahora si cuentame del 1 al 5 como te fue.",
                )
            parciales["p8_no_aplica"] = True
            parciales.pop("p8", None)
            return (
                self._next_unanswered_state_from("esperando_imagen_optin", parciales),
                "Tip NutriBot: tambien tengo modo audio 🎧. Cuando quieras probarlo, solo enviame una consulta por voz.",
            )

        if current_state in {"esperando_imagen_optin", "esperando_imagen_prueba"}:
            if uso_imagen:
                return (
                    self._next_unanswered_state_from("esperando_p9", parciales),
                    "Excelente 🙌 como ya usaste imagen, ahora si cuentame del 1 al 5 como te fue.",
                )
            parciales["p9_no_aplica"] = True
            parciales.pop("p9", None)
            return (
                self._next_unanswered_state_from("esperando_p10", parciales),
                "Tip NutriBot: tambien tengo reconocimiento de imagen 🖼️. Cuando quieras probarlo, enviame una foto de comida o etiqueta nutricional.",
            )

        return current_state, ""

    def _auto_skip_feature_states(
        self,
        *,
        next_state: Optional[str],
        parciales: dict,
        uso_audio: bool,
        uso_imagen: bool,
    ) -> tuple[Optional[str], str]:
        info_parts: list[str] = []
        state_cursor = next_state
        guard = 0
        while state_cursor in self._FEATURE_STATES and guard < 8:
            guard += 1
            advanced_state, info = self._advance_feature_state_by_usage(
                current_state=state_cursor,
                parciales=parciales,
                uso_audio=uso_audio,
                uso_imagen=uso_imagen,
            )
            if info:
                info_parts.append(info)
            if advanced_state == state_cursor:
                break
            state_cursor = advanced_state
        return state_cursor, self._merge_prefix(*info_parts)

    def _build_question_reply(self, state_name: str, prefix: str = "") -> BotReply:
        question = FORM_QUESTIONS.get(state_name, "")
        lines: list[str] = []
        if prefix:
            lines.append(prefix)
        if question:
            lines.append(question)

        has_prefix = bool(prefix and str(prefix).strip())
        if state_name in {"esperando_asegurado_essalud", "esperando_autorizacion"}:
            if not has_prefix:
                lines.append("Responde: Si o No.")
        elif state_name == "esperando_correo":
            lines.append("Si no deseas compartirlo, escribe *no*.")

        deduped_lines: list[str] = []
        seen_keys: set[str] = set()
        for part in lines:
            clean = str(part or "").strip()
            if not clean:
                continue
            key = self._normalize_token(clean)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_lines.append(clean)

        text_msg = "\n\n".join(deduped_lines).strip()
        return BotReply(text=text_msg, content_type="text")

    def _build_consent_reply(self) -> BotReply:
        text_msg = (
            "¿Me podrías ayudar a seguir mejorando contestando algunas preguntas?\n\n"
            "Si estás de acuerdo, empezamos."
        )
        return BotReply(text=text_msg, content_type="text")

    async def _load_active_progress(self, session: AsyncSession, usuario_id: int):
        result = await session.execute(
            text(
                """
                SELECT fp.*
                FROM formulario_en_progreso fp
                JOIN formularios f ON f.id = fp.formulario_id
                WHERE fp.usuario_id = :uid
                  AND f.activo = TRUE
                ORDER BY fp.actualizado_en DESC
                LIMIT 1
                """
            ),
            {"uid": usuario_id},
        )
        return result.fetchone()

    async def _persist_progress(
        self,
        session: AsyncSession,
        progress_id: int,
        parciales: dict,
        next_state: str,
        audio_test_requested: bool,
        audio_test_completed: bool,
        audio_test_declined: bool,
        image_test_requested: bool,
        image_test_completed: bool,
        image_test_declined: bool,
        uso_audio: bool,
        uso_imagen: bool,
    ) -> None:
        progress_update_stmt = text(
                """
                UPDATE formulario_en_progreso
                SET respuestas_parciales = :p,
                    estado_actual = :next,
                    completado_en = CASE
                        WHEN :is_completed THEN :now
                        ELSE completado_en
                    END,
                    audio_test_requested = :audio_test_requested,
                    audio_test_completed = :audio_test_completed,
                    audio_test_declined = :audio_test_declined,
                    image_test_requested = :image_test_requested,
                    image_test_completed = :image_test_completed,
                    image_test_declined = :image_test_declined,
                    uso_audio = :uso_audio,
                    uso_imagen = :uso_imagen,
                    actualizado_en = :now
                WHERE id = :pid
                """
            ).bindparams(bindparam("p", type_=JSONB))
        await session.execute(
            progress_update_stmt,
            {
                "p": parciales,
                "next": next_state,
                "is_completed": next_state == "completado",
                "audio_test_requested": audio_test_requested,
                "audio_test_completed": audio_test_completed,
                "audio_test_declined": audio_test_declined,
                "image_test_requested": image_test_requested,
                "image_test_completed": image_test_completed,
                "image_test_declined": image_test_declined,
                "uso_audio": uso_audio,
                "uso_imagen": uso_imagen,
                "now": get_now_peru(),
                "pid": progress_id,
            },
        )

    async def _persist_final_results(
        self,
        session: AsyncSession,
        progress,
        parciales: dict,
        state: ConversationState,
    ) -> None:
        correo = self._normalize_email(parciales.get("correo"))
        correo_proporcionado = correo is not None
        nps = self._parse_int(parciales.get("nps"), 1, 10)
        autorizo = self._normalize_auth(parciales.get("autorizacion"))
        comentario = parciales.get("comentario")
        asegurado_essalud = self._normalize_auth(parciales.get("asegurado_essalud"))
        asegurado_essalud_text = None if asegurado_essalud is None else ("Si" if asegurado_essalud else "No")

        audio_evaluado = self._parse_int(parciales.get("p8"), 1, 5) is not None
        audio_no_aplica = bool(parciales.get("p8_no_aplica"))
        imagen_evaluada = self._parse_int(parciales.get("p9"), 1, 5) is not None
        imagen_no_aplica = bool(parciales.get("p9_no_aplica"))

        resp_result = await session.execute(
            text(
                """
                INSERT INTO respuestas_formulario (
                    formulario_id, usuario_id, correo, correo_proporcionado,
                    asegurado_essalud, comentario, autorizo_uso_investigacion, puntaje_nps,
                    audio_evaluado, audio_no_aplica, imagen_evaluada, imagen_no_aplica
                )
                VALUES (
                    :fid, :uid, :correo, :correo_proporcionado,
                    :asegurado_essalud, :comentario, :autorizo, :nps,
                    :audio_evaluado, :audio_no_aplica, :imagen_evaluada, :imagen_no_aplica
                )
                ON CONFLICT (formulario_id, usuario_id) DO UPDATE SET
                    correo = EXCLUDED.correo,
                    correo_proporcionado = EXCLUDED.correo_proporcionado,
                    asegurado_essalud = EXCLUDED.asegurado_essalud,
                    comentario = EXCLUDED.comentario,
                    autorizo_uso_investigacion = EXCLUDED.autorizo_uso_investigacion,
                    puntaje_nps = EXCLUDED.puntaje_nps,
                    audio_evaluado = EXCLUDED.audio_evaluado,
                    audio_no_aplica = EXCLUDED.audio_no_aplica,
                    imagen_evaluada = EXCLUDED.imagen_evaluada,
                    imagen_no_aplica = EXCLUDED.imagen_no_aplica
                RETURNING id
                """
            ),
            {
                "fid": progress.formulario_id,
                "uid": state.usuario_id,
                "correo": correo,
                "correo_proporcionado": correo_proporcionado,
                "asegurado_essalud": asegurado_essalud_text,
                "comentario": comentario,
                "autorizo": autorizo,
                "nps": nps,
                "audio_evaluado": audio_evaluado,
                "audio_no_aplica": audio_no_aplica,
                "imagen_evaluada": imagen_evaluada,
                "imagen_no_aplica": imagen_no_aplica,
            },
        )
        respuesta_id = resp_result.scalar()
        if not respuesta_id:
            return

        eval_values = {
            "respuesta_id": respuesta_id,
            "p1": self._parse_int(parciales.get("p1"), 1, 5),
            "p2": self._parse_int(parciales.get("p2"), 1, 5),
            "p3": self._parse_int(parciales.get("p3"), 1, 5),
            "p4": self._parse_int(parciales.get("p4"), 1, 5),
            "p5": self._parse_int(parciales.get("p5"), 1, 5),
            "p6": self._parse_int(parciales.get("p6"), 1, 5),
            "p7": self._parse_int(parciales.get("p7"), 1, 5),
            "p8": self._parse_int(parciales.get("p8"), 1, 5),
            "p9": self._parse_int(parciales.get("p9"), 1, 5),
            "p10": self._parse_int(parciales.get("p10"), 1, 5),
        }
        await session.execute(
            text(
                """
                INSERT INTO evaluacion_usabilidad (
                    respuesta_formulario_id,
                    p1_personalidad_realista_atractiva,
                    p2_explico_proposito_alcance,
                    p3_facil_navegar,
                    p4_me_entendio_bien,
                    p5_respuestas_utiles_apropiadas_informativas,
                    p6_manejo_errores_equivocaciones,
                    p7_muy_facil_de_usar,
                    p8_audio_claridad,
                    p9_reconocio_contexto_imagen,
                    p10_se_enfoco_en_nutricion
                )
                VALUES (
                    :respuesta_id,
                    :p1, :p2, :p3, :p4, :p5, :p6, :p7, :p8, :p9, :p10
                )
                ON CONFLICT (respuesta_formulario_id) DO UPDATE SET
                    p1_personalidad_realista_atractiva = EXCLUDED.p1_personalidad_realista_atractiva,
                    p2_explico_proposito_alcance = EXCLUDED.p2_explico_proposito_alcance,
                    p3_facil_navegar = EXCLUDED.p3_facil_navegar,
                    p4_me_entendio_bien = EXCLUDED.p4_me_entendio_bien,
                    p5_respuestas_utiles_apropiadas_informativas = EXCLUDED.p5_respuestas_utiles_apropiadas_informativas,
                    p6_manejo_errores_equivocaciones = EXCLUDED.p6_manejo_errores_equivocaciones,
                    p7_muy_facil_de_usar = EXCLUDED.p7_muy_facil_de_usar,
                    p8_audio_claridad = EXCLUDED.p8_audio_claridad,
                    p9_reconocio_contexto_imagen = EXCLUDED.p9_reconocio_contexto_imagen,
                    p10_se_enfoco_en_nutricion = EXCLUDED.p10_se_enfoco_en_nutricion
                """
            ),
            eval_values,
        )

    async def process(
        self,
        session: AsyncSession,
        state: ConversationState,
        normalized: NormalizedMessage,
        projected_interactions_count: Optional[int] = None,
    ) -> Optional[BotReply]:
        if state.awaiting_question_code == CONSENT_STATE:
            return await self._handle_consent_response(session, state, normalized)

        if state.mode == "collecting_usability":
            return await self._process_form_response(session, state, normalized)

        if state.mode == "active_chat":
            if self._is_profile_capture_active(state):
                return None
            if self._is_reinvite_cooldown_active(state):
                return None
            effective = projected_interactions_count if projected_interactions_count is not None else state.meaningful_interactions_count
            if effective >= 5 and state.usability_completion_pct < 100:
                active_form = await self._get_active_form(session)
                if not active_form:
                    # Evita invitar a un formulario inexistente.
                    state.meaningful_interactions_count = 0
                    return None
                state.awaiting_question_code = CONSENT_STATE
                state.meaningful_interactions_count = 0
                state.last_form_prompt_at = get_now_peru()
                return self._build_consent_reply()
        return None

    async def _handle_consent_response(
        self,
        session: AsyncSession,
        state: ConversationState,
        normalized: NormalizedMessage,
    ) -> Optional[BotReply]:
        # Si llega multimedia, no forzamos consentimiento: devolvemos control al chat normal.
        if self._is_multimedia_message(normalized):
            state.awaiting_question_code = None
            state.mode = "active_chat"
            state.meaningful_interactions_count = 0
            return None

        result = await self.extractor.extract(CONSENT_STATE, normalized.text, normalized.interactive_id)
        intent = result.get("intent", "ANSWER").upper()
        value = self._normalize_token(result.get("value"))

        if intent == "INTERRUPT":
            state.awaiting_question_code = None
            state.mode = "active_chat"
            state.meaningful_interactions_count = 0
            state.last_form_prompt_at = get_now_peru()
            return None

        if value in {"si", "yes"}:
            state.awaiting_question_code = None
            return await self._try_start_form(session, state)

        if value in {"no"}:
            state.awaiting_question_code = None
            state.mode = "active_chat"
            state.meaningful_interactions_count = 0
            state.last_form_prompt_at = get_now_peru()
            return BotReply(text="Entendido, seguimos conversando sin problema.", content_type="text")

        return self._build_consent_reply()

    async def _try_start_form(self, session: AsyncSession, state: ConversationState) -> Optional[BotReply]:
        form = await self._get_active_form(session)
        if not form:
            state.meaningful_interactions_count = 0
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return BotReply(
                text="Gracias por aceptar 😊. Por ahora no tengo un formulario activo; seguimos con tu chat normal.",
                content_type="text",
            )

        progress = await self._load_active_progress(session, state.usuario_id)
        if progress and progress.formulario_id == form.id:
            if progress.estado_actual == "completado":
                state.meaningful_interactions_count = 0
                return None
            
            parciales = dict(progress.respuestas_parciales or {})
            resumption_state = progress.estado_actual
            resumption_prefix = "Retomemos el formulario pendiente."

            # Mejora: Si no proporcionó correo, volver a pedirlo al retomar, pero luego saltar a donde estaba
            if not parciales.get("correo"):
                # Guardamos donde estaba para volver despues
                if resumption_state != "esperando_correo":
                    parciales["return_to_state"] = resumption_state
                
                resumption_state = "esperando_correo"
                parciales["correo_decline_count"] = 0 # Reset para que vuelva a pedir
                await self._persist_progress(
                    session=session,
                    progress_id=progress.id,
                    parciales=parciales,
                    next_state=resumption_state,
                    audio_test_requested=bool(getattr(progress, "audio_test_requested", False)),
                    audio_test_completed=bool(getattr(progress, "audio_test_completed", False)),
                    audio_test_declined=bool(getattr(progress, "audio_test_declined", False)),
                    image_test_requested=bool(getattr(progress, "image_test_requested", False)),
                    image_test_completed=bool(getattr(progress, "image_test_completed", False)),
                    image_test_declined=bool(getattr(progress, "image_test_declined", False)),
                    uso_audio=bool(getattr(progress, "uso_audio", False)),
                    uso_imagen=bool(getattr(progress, "uso_imagen", False)),
                )
                resumption_prefix = "¡Hola de nuevo! Retomemos el formulario pendiente. Empecemos por tu correo y luego seguimos donde nos quedamos."

            state.mode = "collecting_usability"
            state.awaiting_question_code = resumption_state
            state.meaningful_interactions_count = 0
            return self._build_question_reply(resumption_state, prefix=resumption_prefix)

        first_state = FORM_STATES_ORDER[0]
        await session.execute(
            text(
                """
                INSERT INTO formulario_en_progreso (
                    formulario_id, usuario_id, estado_actual, respuestas_parciales,
                    completado_en, audio_test_requested, audio_test_completed, audio_test_declined,
                    image_test_requested, image_test_completed, image_test_declined, actualizado_en
                )
                VALUES (
                    :fid, :uid, :state, '{}'::jsonb,
                    NULL, FALSE, FALSE, FALSE,
                    FALSE, FALSE, FALSE, :now
                )
                ON CONFLICT (usuario_id) DO UPDATE SET
                    formulario_id = EXCLUDED.formulario_id,
                    estado_actual = EXCLUDED.estado_actual,
                    respuestas_parciales = '{}'::jsonb,
                    completado_en = NULL,
                    audio_test_requested = FALSE,
                    audio_test_completed = FALSE,
                    audio_test_declined = FALSE,
                    image_test_requested = FALSE,
                    image_test_completed = FALSE,
                    image_test_declined = FALSE,
                    actualizado_en = :now
                """
            ),
            {"fid": form.id, "uid": state.usuario_id, "state": first_state, "now": get_now_peru()},
        )
        state.mode = "collecting_usability"
        state.awaiting_question_code = first_state
        state.meaningful_interactions_count = 0
        return self._build_question_reply(first_state)

    async def _get_active_form(self, session: AsyncSession):
        result = await session.execute(
            text(
                """
                SELECT id, codigo, version
                FROM formularios
                WHERE activo = TRUE
                ORDER BY version DESC, id DESC
                LIMIT 1
                """
            )
        )
        return result.fetchone()

    async def get_current_question_reply(self, session: AsyncSession, state: ConversationState) -> Optional[BotReply]:
        """Retorna la respuesta/pregunta actual basada en el estado, sin procesar nada."""
        current_state = state.awaiting_question_code
        if not current_state or current_state not in FORM_QUESTIONS:
            return None
        return self._build_question_reply(current_state)

    async def _process_form_response(
        self,
        session: AsyncSession,
        state: ConversationState,
        normalized: NormalizedMessage,
    ) -> Optional[BotReply]:
        current_state = state.awaiting_question_code
        if not current_state or current_state not in FORM_QUESTIONS:
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return None

        progress = await self._load_active_progress(session, state.usuario_id)
        if not progress:
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return None

        parciales = dict(progress.respuestas_parciales or {})
        if self._is_multimedia_message(normalized) and current_state not in {"esperando_audio_prueba", "esperando_imagen_prueba"}:
            await self._persist_progress(
                session=session,
                progress_id=progress.id,
                parciales=parciales,
                next_state=current_state,
                audio_test_requested=bool(getattr(progress, "audio_test_requested", False)),
                audio_test_completed=bool(getattr(progress, "audio_test_completed", False)),
                audio_test_declined=bool(getattr(progress, "audio_test_declined", False)),
                image_test_requested=bool(getattr(progress, "image_test_requested", False)),
                image_test_completed=bool(getattr(progress, "image_test_completed", False)),
                image_test_declined=bool(getattr(progress, "image_test_declined", False)),
                uso_audio=bool(getattr(progress, "uso_audio", False)),
                uso_imagen=bool(getattr(progress, "uso_imagen", False)),
            )
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return None

        result = await self.extractor.extract(current_state, normalized.text, normalized.interactive_id)
        intent = str(result.get("intent", "ANSWER")).upper()
        value = result.get("value")
        audio_test_requested = bool(getattr(progress, "audio_test_requested", False))
        audio_test_completed = bool(getattr(progress, "audio_test_completed", False))
        audio_test_declined = bool(getattr(progress, "audio_test_declined", False))
        image_test_requested = bool(getattr(progress, "image_test_requested", False))
        image_test_completed = bool(getattr(progress, "image_test_completed", False))
        image_test_declined = bool(getattr(progress, "image_test_declined", False))
        uso_audio = bool(getattr(progress, "uso_audio", False))
        uso_imagen = bool(getattr(progress, "uso_imagen", False))

        if intent == "CANCEL":
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return BotReply(text="Entendido, dejamos el formulario por ahora.", content_type="text")
        if intent == "INTERRUPT":
            interrupt_next_state = current_state
            if current_state == "esperando_audio_prueba" and not self._is_audio_message(normalized):
                audio_test_declined = True
                parciales["p8_no_aplica"] = True
                parciales.pop("p8", None)
                interrupt_next_state = "esperando_imagen_optin"
            elif current_state == "esperando_imagen_prueba" and not self._is_image_message(normalized):
                image_test_declined = True
                parciales["p9_no_aplica"] = True
                parciales.pop("p9", None)
                interrupt_next_state = "esperando_p10"

            await self._persist_progress(
                session=session,
                progress_id=progress.id,
                parciales=parciales,
                next_state=interrupt_next_state,
                audio_test_requested=bool(getattr(progress, "audio_test_requested", False)),
                audio_test_completed=bool(getattr(progress, "audio_test_completed", False)),
                audio_test_declined=audio_test_declined,
                image_test_requested=bool(getattr(progress, "image_test_requested", False)),
                image_test_completed=bool(getattr(progress, "image_test_completed", False)),
                image_test_declined=image_test_declined,
                uso_audio=bool(getattr(progress, "uso_audio", False)),
                uso_imagen=bool(getattr(progress, "uso_imagen", False)),
            )
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return None

        transition_prefix = ""

        if current_state in self._FEATURE_STATES:
            next_state, transition_prefix = self._advance_feature_state_by_usage(
                current_state=current_state,
                parciales=parciales,
                uso_audio=uso_audio,
                uso_imagen=uso_imagen,
            )
            if not next_state:
                await self._persist_progress(
                    session=session,
                    progress_id=progress.id,
                    parciales=parciales,
                    next_state="completado",
                    audio_test_requested=audio_test_requested,
                    audio_test_completed=audio_test_completed,
                    audio_test_declined=audio_test_declined,
                    image_test_requested=image_test_requested,
                    image_test_completed=image_test_completed,
                    image_test_declined=image_test_declined,
                    uso_audio=uso_audio,
                    uso_imagen=uso_imagen,
                )
                await self._persist_final_results(session, progress, parciales, state)
                state.usability_completion_pct = 100
                state.mode = "active_chat"
                state.awaiting_question_code = None
                return BotReply(text="Muchas gracias por completar el formulario.", content_type="text")

            next_state, auto_info = self._auto_skip_feature_states(
                next_state=next_state,
                parciales=parciales,
                uso_audio=uso_audio,
                uso_imagen=uso_imagen,
            )
            next_state = self._next_unanswered_state_from(next_state, parciales)
            transition_prefix = self._merge_prefix(transition_prefix, auto_info)

            await self._persist_progress(
                session=session,
                progress_id=progress.id,
                parciales=parciales,
                next_state=next_state,
                audio_test_requested=audio_test_requested,
                audio_test_completed=audio_test_completed,
                audio_test_declined=audio_test_declined,
                image_test_requested=image_test_requested,
                image_test_completed=image_test_completed,
                image_test_declined=image_test_declined,
                uso_audio=uso_audio,
                uso_imagen=uso_imagen,
            )
            state.awaiting_question_code = next_state
            return self._build_question_reply(next_state, prefix=transition_prefix)

        if current_state not in self._FEATURE_STATES:
            field_key = current_state.replace("esperando_", "")
            cleaned_value = value
            if current_state == "esperando_correo":
                if intent == "WHY":
                    return BotReply(
                        text=(
                            "Te lo pido para avisarte campañas de salud y nutrición cerca de ti.\n\n"
                            "Puedes compartir tu correo o escribir *no* para continuar."
                        ),
                        content_type="text",
                    )
                if intent == "SKIP":
                    decline_count = int(parciales.get("correo_decline_count") or 0) + 1
                    parciales["correo_decline_count"] = decline_count
                    if decline_count == 1:
                        await self._persist_progress(
                            session=session,
                            progress_id=progress.id,
                            parciales=parciales,
                            next_state=current_state,
                            audio_test_requested=audio_test_requested,
                            audio_test_completed=audio_test_completed,
                            audio_test_declined=audio_test_declined,
                            image_test_requested=image_test_requested,
                            image_test_completed=image_test_completed,
                            image_test_declined=image_test_declined,
                            uso_audio=uso_audio,
                            uso_imagen=uso_imagen,
                        )
                        state.awaiting_question_code = current_state
                        return BotReply(text=EMAIL_REASK_COPY, content_type="text")
                    cleaned_value = None
                normalized_email = self._normalize_email(cleaned_value)
                if cleaned_value is not None and not normalized_email:
                    return BotReply(
                        text=(
                            "Ese correo no parece valido. Ejemplo: tunombre@gmail.com\n\n"
                            "Si prefieres no compartirlo, escribe *no* y continuamos."
                        ),
                        content_type="text",
                    )
                cleaned_value = normalized_email
                parciales["correo_decline_count"] = 0 if normalized_email else int(parciales.get("correo_decline_count") or 0)
            elif intent in {"WHY", "SKIP"}:
                if current_state == "esperando_comentario":
                    cleaned_value = None
                else:
                    return self._build_question_reply(
                        current_state,
                        prefix="Necesito tu respuesta para continuar.",
                    )

            if current_state.startswith("esperando_p") or current_state == "esperando_nps":
                min_v, max_v = (1, 10) if current_state == "esperando_nps" else (1, 5)
                parsed = self._parse_int(cleaned_value, min_v, max_v)
                if parsed is None:
                    if max_v == 5:
                        hint = "La respuesta debe ser un numero del 1 al 5."
                    else:
                        hint = "La respuesta debe ser un numero del 1 al 10."
                    return self._build_question_reply(current_state, prefix=hint)
                cleaned_value = str(parsed)

            if current_state == "esperando_asegurado_essalud":
                insured = self._normalize_auth(cleaned_value)
                if insured is None:
                    return self._build_question_reply(current_state)
                cleaned_value = "Si" if insured else "No"

            if current_state == "esperando_autorizacion":
                auth = self._normalize_auth(cleaned_value)
                if auth is None:
                    return self._build_question_reply(current_state)
                cleaned_value = "Si" if auth else "No"

            parciales[field_key] = cleaned_value
            
            # Logica de "volver al estado original" si venimos de una reanudacion
            return_to = parciales.pop("return_to_state", None)
            forced_final_email = bool(parciales.pop("correo_required_for_finalize", False))
            if return_to:
                next_state = return_to
            elif current_state == "esperando_correo" and forced_final_email:
                # Si el correo se pidió solo para finalizar, no volver a recorrer
                # toda la encuesta. Cerrar al terminar este paso.
                if not cleaned_value:
                    parciales["correo_opt_out_final"] = True
                next_state = None
            else:
                idx = FORM_STATES_ORDER.index(current_state)
                next_state = FORM_STATES_ORDER[idx + 1] if idx + 1 < len(FORM_STATES_ORDER) else None
            
            next_state, auto_info = self._auto_skip_feature_states(
                next_state=next_state,
                parciales=parciales,
                uso_audio=uso_audio,
                uso_imagen=uso_imagen,
            )
            transition_prefix = self._merge_prefix(transition_prefix, auto_info)

        if next_state is None:
            # Antes de marcar como completado, verificar si falta el correo
            correo_missing = not parciales.get("correo")
            allow_finish_without_correo = bool(parciales.get("correo_opt_out_final"))
            if correo_missing and not allow_finish_without_correo:
                next_state = "esperando_correo"
                parciales["correo_required_for_finalize"] = True
                transition_prefix = self._merge_prefix(transition_prefix, "Para finalizar y guardar tus respuestas, necesitamos que nos proporciones tu correo.")
            else:
                pending_media_states = self._missing_media_feedback_states(parciales)
                if pending_media_states:
                    resume_state = pending_media_states[0]
                    await self._persist_progress(
                        session=session,
                        progress_id=progress.id,
                        parciales=parciales,
                        next_state=resume_state,
                        audio_test_requested=audio_test_requested,
                        audio_test_completed=audio_test_completed,
                        audio_test_declined=audio_test_declined,
                        image_test_requested=image_test_requested,
                        image_test_completed=image_test_completed,
                        image_test_declined=image_test_declined,
                        uso_audio=uso_audio,
                        uso_imagen=uso_imagen,
                    )
                    state.mode = "active_chat"
                    state.awaiting_question_code = None
                    state.usability_completion_pct = 85
                    return BotReply(
                        text=(
                            "Gracias 😊 Ya guardé tus respuestas actuales.\n\n"
                            "Recuerda que cuando quieras puedes enviarme audios o imágenes para ayudarte mejor. Por ejemplo:\n"
                            "🎤 Audio: Puedes enviarme un audio corto contándome qué almorzaste.\n"
                            "📸 Imagen: Puedes enviarme una foto de la etiqueta de un producto o de tu plato de comida para que lo analice."
                        ),
                        content_type="text",
                    )

                await self._persist_progress(
                    session=session,
                    progress_id=progress.id,
                    parciales=parciales,
                    next_state="completado",
                    audio_test_requested=audio_test_requested,
                    audio_test_completed=audio_test_completed,
                    audio_test_declined=audio_test_declined,
                    image_test_requested=image_test_requested,
                    image_test_completed=image_test_completed,
                    image_test_declined=image_test_declined,
                    uso_audio=uso_audio,
                    uso_imagen=uso_imagen,
                )
                await self._persist_final_results(session, progress, parciales, state)
                state.usability_completion_pct = 100
                state.mode = "active_chat"
                state.awaiting_question_code = None
                return BotReply(text="Muchas gracias por completar el formulario.", content_type="text")

        await self._persist_progress(
            session=session,
            progress_id=progress.id,
            parciales=parciales,
            next_state=next_state,
            audio_test_requested=audio_test_requested,
            audio_test_completed=audio_test_completed,
            audio_test_declined=audio_test_declined,
            image_test_requested=image_test_requested,
            image_test_completed=image_test_completed,
            image_test_declined=image_test_declined,
            uso_audio=uso_audio,
            uso_imagen=uso_imagen,
        )
        state.awaiting_question_code = next_state
        return self._build_question_reply(next_state, prefix=transition_prefix)
