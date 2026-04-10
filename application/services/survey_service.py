"""
Nutribot Backend — SurveyService
Evalúa la compuerta de cierre y gestiona el formulario de usabilidad.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

# ─── Definición de la máquina de estados del formulario ───
FORM_STATES_ORDER = [
    "esperando_correo",
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
]

FORM_QUESTIONS: dict[str, str] = {
    "esperando_correo": "📧 ¡Me encantaría seguir en contacto! ¿Me podrías compartir tu correo electrónico?",
    "esperando_p1": "Del 1 al 5, ¿sientes que mi personalidad fue realista y atractiva? 😊",
    "esperando_p2": "Del 1 al 5, ¿expliqué bien mi propósito y alcance? 🎯",
    "esperando_p3": "Del 1 al 5, ¿fui fácil de navegar? 🗺️",
    "esperando_p4": "Del 1 al 5, ¿sientes que te entendí bien? 🧠",
    "esperando_p5": "Del 1 al 5, ¿mis respuestas te parecieron útiles, apropiadas e informativas? 🥗",
    "esperando_p6": "Del 1 al 5, ¿manejé bien los errores o equivocaciones? ⚙️",
    "esperando_p7": "Del 1 al 5, ¿fui muy fácil de usar? ✨",
    "esperando_p8": "Del 1 al 5, (si escuchaste audios) ¿mis respuestas por ese medio se hicieron con claridad? 🎙️",
    "esperando_p9": "Del 1 al 5, (si enviaste fotos) ¿pude reconocer bien el contexto de lo que me mostraste? 📸",
    "esperando_p10": "Del 1 al 5, ¿me enfoqué solo en responderte preguntas sobre nutrición? 🥦",
    "esperando_nps": "⭐ En una escala del 1 al 10, ¿qué tan probable es que me recomiendes a un amigo o familiar?",
    "esperando_comentario": "💭 (Opcional) ¿Qué te gustó o no te gustó de mí en esta primera interacción?",
    "esperando_autorizacion": "📋 Por último, ¿autorizas el uso anónimo y agregado de tus respuestas para fines de investigación científica y evaluación de mi herramienta? (Sí autorizo / No autorizo)",
}

PERSUASION_MESSAGES: dict[str, str] = {
    "esperando_correo": "Es para avisarte sobre campañas de salud, jornadas de nutrición cerca de ti y consejos exclusivos para tu perfil. ¡Es súper útil! ✉️ ¿Te animas?",
}
PERSUASION_DEFAULT = "Entiendo, pero tus respuestas son vitales para mejorar este servicio para todos. 🌟 ¿Te gustaría darnos este dato?"


class SurveyResponseExtractor:
    """Extrae intención y valor limpio de las respuestas del usuario a la encuesta."""

    _SKIP_EXACT = {"paso", "saltar", "skip", "no quiero", "siguiente", "prefiero no"}
    _CANCEL_PHRASES = ["cancelar todo", "detener bot", "salir de todo", "stop survey"]
    _YES_WORDS = re.compile(r"\b(s[ií1]|yes|yeah|claro|por supuesto|obvio|dale|ok[ay]*|chi|chii|afirmativo|autorizo|acepto|normal)\b", re.IGNORECASE)
    _NO_WORDS = re.compile(r"\b(no(?:\s+(?:autorizo|acepto|quiero))?|nop[e]?|nel|negativo|rechazo)\b", re.IGNORECASE)
    _NO_SE_WORDS = re.compile(r"\b(no\s*s[eé]|nose|ni idea)\b", re.IGNORECASE)
    _WHY_PATTERN = re.compile(r"(\bpara\s*qu[eé]\b|\bpor\s*qu[eé]\b|\bqu[eé]\s*uso\b|\bpara\s*que\s*sirve\b|\bpor\s*que\s*lo\s*pide|\bpara\s*que\s*es\b|\bcomo\s*para\s*que\b|\bpara\b\?)", re.IGNORECASE)

    _NUTRITION_HINTS = (
        "menu", "menÃº", "receta", "dieta", "imc", "calorias", "calorÃ­as",
        "desayuno", "almuerzo", "cena", "peso", "talla", "proteina", "proteÃ­na",
        "carbohidratos", "grasa", "nutricion", "nutriciÃ³n", "pescado", "marino",
    )

    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self._client = openai_client
        self._model = model

    def try_fast_extract(self, state_name: str, user_text: str) -> Optional[dict]:
        v = user_text.strip()
        vl = v.lower()

        if any(c in vl for c in self._CANCEL_PHRASES):
            return {"intent": "CANCEL", "value": None}
        if self._WHY_PATTERN.search(vl):
            return {"intent": "WHY", "value": None}
        if any(vl.startswith(s) or vl == s for s in self._SKIP_EXACT):
            return {"intent": "SKIP", "value": None}
        if state_name == "esperando_correo":
            # Si rechaza compartir correo, lo tratamos como SKIP para activar
            # la logica de persuasion (una vez) y luego continuar con usabilidad.
            if self._NO_WORDS.search(vl) or "prefiero no" in vl or "no compartir" in vl or "no dar" in vl:
                return {"intent": "SKIP", "value": None}
            match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", v)
            if match:
                return {"intent": "ANSWER", "value": match.group(0)}
            if any(hint in vl for hint in self._NUTRITION_HINTS):
                return {"intent": "INTERRUPT", "value": None}
            return None

        if state_name in ("esperando_asegurado", "esperando_autorizacion"):
            if self._YES_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "Sí"}
            if self._NO_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "No"}
            if self._NO_SE_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "No sé"}
            return None

        if state_name.startswith("esperando_p"):
            digits = re.sub(r"\D", "", v)
            if digits:
                num = int(digits[0])
                if 1 <= num <= 5:
                    return {"intent": "ANSWER", "value": str(num)}
            return None

        if state_name == "esperando_nps":
            digits = re.sub(r"\D", "", v)
            if digits:
                num = int(digits)
                if 1 <= num <= 10:
                    return {"intent": "ANSWER", "value": str(num)}
            return None

        if state_name == "esperando_comentario":
            if any(hint in vl for hint in self._NUTRITION_HINTS):
                return {"intent": "INTERRUPT", "value": None}
            return {"intent": "ANSWER", "value": v if vl not in ("no", "nada", "ninguno") else None}

        return None

    async def extract_with_ai(self, state_name: str, user_text: str) -> dict:
        if any(hint in user_text.lower() for hint in self._NUTRITION_HINTS):
            return {"intent": "INTERRUPT", "value": None}

        field_type = "EMAIL" if state_name == "esperando_correo" else \
                     "BOOLEAN (Si/No/No se)" if state_name in ("esperando_asegurado", "esperando_autorizacion") else \
                     "NUMBER (1-5)" if state_name.startswith("esperando_p") else \
                     "NUMBER (1-10)" if state_name == "esperando_nps" else \
                     "TEXT (Comentario)"

        prompt = f"""Analiza la respuesta del usuario a: "{FORM_QUESTIONS.get(state_name, '')}".\nCategorízala:\n- ANSWER\n- WHY\n- SKIP\n- INTERRUPT\nSi es ANSWER, extrae valor puro ({field_type}).\nJSON: {{"intent": "CATEGORIA", "value": "VALOR"}}\nUSUARIO: "{user_text}" """
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "Analista de encuestas. Lenguaje natural con errores."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.error("Error en IA survey extractor: %s", e)
            return {"intent": "ANSWER", "value": user_text}

    async def extract(self, state_name: str, user_text: str) -> dict:
        fast = self.try_fast_extract(state_name, user_text)
        if fast is not None:
            return fast
        return await self.extract_with_ai(state_name, user_text)


class SurveyService:
    def __init__(self, openai_client: AsyncOpenAI, openai_model: str):
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.extractor = SurveyResponseExtractor(openai_client, openai_model)

    async def process(
        self,
        session: AsyncSession,
        state: ConversationState,
        user_text: str,
        projected_interactions_count: Optional[int] = None,
    ) -> Optional[str]:
        if state.mode in ("collecting_usability", "collecting_profile"):
            return await self._process_form_response(session, state, user_text)

        if state.mode == "closing":
            pending = await self._check_pending_form(session, state)
            if pending:
                return pending
            return None

        if state.mode == "active_chat":
            effective_count = projected_interactions_count if projected_interactions_count is not None else state.meaningful_interactions_count
            if effective_count >= 5:
                # GUARDIA DE SOLAPAMIENTO
                now = get_now_peru()
                diff = now - state.onboarding_updated_at if state.onboarding_updated_at else None
                if diff and diff.total_seconds() < 300:
                    return None

                if state.usability_completion_pct < 100:
                    prefix = "\n\nPD: ¡Muchas gracias por chatear conmigo! Antes de despedirnos, me encantaría saber si me permites hacerte algunas breves preguntas sobre cómo ha sido tu experiencia usándome. 😊\n\n¿Estás de acuerdo?"
                    return await self._try_start_form(session, state, prefix=prefix)

        return None

    async def _try_start_form(self, session: AsyncSession, state: ConversationState, prefix: str = "") -> Optional[str]:
        result = await session.execute(text("SELECT id FROM formularios WHERE activo = TRUE ORDER BY id LIMIT 1"))
        form = result.fetchone()
        if not form:
            return None

        result = await session.execute(
            text("SELECT id, estado_actual, respuestas_parciales FROM formulario_en_progreso WHERE usuario_id = :uid AND formulario_id = :fid"),
            {"uid": state.usuario_id, "fid": form.id},
        )
        progress = result.fetchone()

        if progress:
            current_state = progress.estado_actual
            if current_state == "completado":
                return None
            state.mode = "collecting_usability"
            state.awaiting_question_code = current_state
            state.meaningful_interactions_count = 0
            question = FORM_QUESTIONS.get(current_state, "")
            return f"¡Antes de irte! 😊 Me quedaron unas preguntitas pendientes de la última vez que me ayudan mucho a mejorar NutriBot. ¿Te parece si las completamos rápido?\n\n{question}"

        first_state = FORM_STATES_ORDER[0]
        await session.execute(
            text("""
                INSERT INTO formulario_en_progreso (formulario_id, usuario_id, estado_actual, respuestas_parciales)
                VALUES (:fid, :uid, :state, '{}')
                ON CONFLICT (usuario_id) DO UPDATE SET estado_actual = :state, actualizado_en = :now
            """),
            {"fid": form.id, "uid": state.usuario_id, "state": first_state, "now": get_now_peru()},
        )
        state.mode = "collecting_usability"
        state.awaiting_question_code = first_state
        state.turns_since_last_prompt = 0
        state.meaningful_interactions_count = 0
        question = FORM_QUESTIONS[first_state]
        return f"{prefix}\n\n{question}" if prefix else f"¡Muchas gracias por chatear conmigo! 🙏 Antes de despedirnos, me encantaría saber qué te pareció la experiencia. Son unas preguntitas rápidas.\n\n{question}"

    async def _check_pending_form(self, session: AsyncSession, state: ConversationState) -> Optional[str]:
        result = await session.execute(
            text("""
                SELECT fp.estado_actual, fp.formulario_id 
                FROM formulario_en_progreso fp
                JOIN formularios f ON f.id = fp.formulario_id
                WHERE fp.usuario_id = :uid AND f.activo = TRUE AND fp.estado_actual != 'completado'
            """),
            {"uid": state.usuario_id},
        )
        row = result.fetchone()
        if not row:
            state.mode = "active_chat"
            return None

        state.mode = "collecting_usability"
        state.awaiting_question_code = row.estado_actual
        question = FORM_QUESTIONS.get(row.estado_actual, "")
        return f"😊 ¡Hola de nuevo! Nos quedaron unas preguntitas pendientes, ¿te animas a completarlas?\n\n{question}"

    async def _process_form_response(self, session: AsyncSession, state: ConversationState, user_text: str) -> Optional[str]:
        current_state = state.awaiting_question_code

        if not current_state or current_state not in FORM_QUESTIONS:
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return None

        result_data = await self.extractor.extract(current_state, user_text)
        intent = result_data.get("intent", "ANSWER").upper()
        extracted_value = result_data.get("value")

        if intent == "CANCEL":
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return "¡Entendido! Lo dejamos por ahora. Si tienes alguna duda, estaré por aquí. 🍏"

        if intent == "INTERRUPT":
            state.mode = "active_chat"
            return None

        result = await session.execute(
            text("SELECT id, formulario_id, respuestas_parciales, uso_audio, uso_imagen FROM formulario_en_progreso WHERE usuario_id = :uid"),
            {"uid": state.usuario_id},
        )
        progress = result.fetchone()
        if not progress:
            state.mode = "active_chat"
            return None

        parciales = progress.respuestas_parciales or {}
        field_key = current_state.replace("esperando_", "")

        if intent in ("WHY", "SKIP"):
            persuasion_key = f"{field_key}_persuaded"

            if not parciales.get(persuasion_key):
                parciales[persuasion_key] = True
                await session.execute(
                    text("UPDATE formulario_en_progreso SET respuestas_parciales = :p WHERE usuario_id = :uid"),
                    {"p": json.dumps(parciales), "uid": state.usuario_id},
                )
                return PERSUASION_MESSAGES.get(current_state, PERSUASION_DEFAULT)

            if current_state == "esperando_correo":
                next_state = "esperando_p1"
                await session.execute(
                    text("UPDATE formulario_en_progreso SET estado_actual = :next WHERE usuario_id = :uid"),
                    {"next": next_state, "uid": state.usuario_id},
                )
                state.awaiting_question_code = next_state
                return f"Entiendo, no te preocupes. 😊 ¿Me podrías ayudar al menos con estas otras consultas breves por favor? Me ayudará mucho a darte un mejor servicio.\n\n{FORM_QUESTIONS[next_state]}"

            cleaned_value = None
        else:
            cleaned_value = extracted_value
            if current_state == "esperando_correo" and ("@" not in str(cleaned_value) or "." not in str(cleaned_value)):
                return "Hmm, ese correo no parece válido 🤔 ¿Podrías revisarlo? Ejemplo: tunombre@gmail.com"

            if (current_state.startswith("esperando_p") or current_state == "esperando_nps") and not str(cleaned_value).isdigit():
                max_val = "10" if current_state == "esperando_nps" else "5"
                return f"¡Muchas gracias por tu comentario! 😊 Pero para poder registrar tu experiencia, ¿me podrías dar un número del 1 al {max_val}?"

        if cleaned_value is not None:
            parciales[field_key] = cleaned_value

        current_idx = FORM_STATES_ORDER.index(current_state)
        next_state = None
        media_addon = ""

        for i in range(current_idx + 1, len(FORM_STATES_ORDER)):
            candidate = FORM_STATES_ORDER[i]
            if candidate == "esperando_p8" and not progress.uso_audio:
                media_addon += "\n\n🎙️ *¡Probemos el audio!* Tengo la función de audio disponible, ¿te gustaría probarla pronto? Me ayuda mucho a escucharte mejor y darte recetas exactas. 😊"
                continue
            if candidate == "esperando_p9" and not progress.uso_imagen:
                media_addon += "\n\n📸 *¡Reconozco fotos!* También tengo la función de reconocer fotos e imágenes de tus platos. ¿Te gustaría probarla? Me ayuda a ser más visual con tus porciones. 🥦"
                continue
            next_state = candidate
            break

        if next_state is None:
            await session.execute(
                text("UPDATE formulario_en_progreso SET respuestas_parciales = :p, estado_actual = 'completado', actualizado_en = :now WHERE id = :pid"),
                {"p": json.dumps(parciales), "now": get_now_peru(), "pid": progress.id},
            )
            state.usability_completion_pct = 100
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return f"¡Muchas gracias por tus respuestas! 🙏 Esto me ayuda a ser un mejor bot nutricional para todos. ¡Espero verte pronto! 🍏✨{media_addon}"
        else:
            await session.execute(
                text("UPDATE formulario_en_progreso SET respuestas_parciales = :p, estado_actual = :next, actualizado_en = :now WHERE id = :pid"),
                {"p": json.dumps(parciales), "next": next_state, "now": get_now_peru(), "pid": progress.id},
            )
            state.awaiting_question_code = next_state
            return FORM_QUESTIONS[next_state]
