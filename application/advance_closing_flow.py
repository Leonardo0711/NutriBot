"""
Nutribot Backend — AdvanceClosingFlowUseCase
Evalúa la compuerta de cierre y gestiona el formulario de usabilidad.
Diseño: natural, no intrusivo. Solo al despedirse, y si hay preguntas
pendientes se retoman amablemente en la siguiente conversación.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from domain.entities import ConversationState
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

# ─── Definición de la máquina de estados del formulario ───
FORM_STATES_ORDER = [
    "esperando_correo",
    "esperando_asegurado",
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

# Preguntas del formulario con sus textos amigables (Orden según imágenes, en 1ra persona)
FORM_QUESTIONS: dict[str, str] = {
    "esperando_correo": "📧 Para empezar, ¿me podrías compartir tu correo electrónico?",
    "esperando_asegurado": "🏥 ¿Eres asegurado/a de EsSalud? (Sí, No, o No sé)",
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

# Mensajes de persuasión por campo
PERSUASION_MESSAGES: dict[str, str] = {
    "esperando_correo": "Es para avisarte sobre campañas de salud, jornadas de nutrición cerca de ti y consejos exclusivos para tu perfil nutricional. ¡Es súper útil! ✉️ ¿Te animas a compartirlo?",
    "esperando_asegurado": "Saber si eres asegurado nos ayuda a darte información sobre servicios específicos de salud preventiva en EsSalud 🏥. ¿Te gustaría comentarlo?",
}
PERSUASION_DEFAULT = "Entiendo, pero tus respuestas son vitales para mejorar este servicio para todos. 🌟 ¿Te gustaría darnos este dato?"


# ═══════════════════════════════════════════════════════════════
# Clase: SurveyResponseExtractor
# Separa la extracción de datos en un "fast path" (regex) y
# un "slow path" (LLM) para minimizar la latencia.
# ═══════════════════════════════════════════════════════════════
class SurveyResponseExtractor:
    """Extrae intención y valor limpio de las respuestas del usuario a la encuesta."""

    # Palabras que claramente indican un salto/rechazo
    _SKIP_EXACT = {"paso", "saltar", "skip", "no quiero", "siguiente", "prefiero no"}
    # Cancelación global (frases extremas)
    _CANCEL_PHRASES = ["cancelar todo", "detener bot", "salir de todo", "stop survey"]
    # Palabras afirmativas (para booleanos)
    _YES_WORDS = re.compile(
        r"\b(s[ií1]|yes|yeah|claro|por supuesto|obvio|dale|ok[ay]*|chi|chii|afirmativo|autorizo|acepto|normal)\b",
        re.IGNORECASE,
    )
    _NO_WORDS = re.compile(
        r"\b(no(?:\s+(?:autorizo|acepto|quiero))?|nop[e]?|nel|negativo|rechazo)\b",
        re.IGNORECASE,
    )
    _NO_SE_WORDS = re.compile(r"\b(no\s*s[eé]|nose|ni idea)\b", re.IGNORECASE)
    # Regex para detectar preguntas de "por qué"
    _WHY_PATTERN = re.compile(
        r"(\bpara\s*qu[eé]\b|\bpor\s*qu[eé]\b|\bqu[eé]\s*uso\b|\bpara\s*que\s*sirve\b|\bpor\s*que\s*lo\s*pide|\bpara\s*que\s*es\b|\bcomo\s*para\s*que\b|\bpara\b\?)",
        re.IGNORECASE,
    )

    def __init__(self, openai_client: AsyncOpenAI, model: str):
        self._client = openai_client
        self._model = model

    # ─── Fast Path: respuestas obvias sin IA ───
    def try_fast_extract(self, state_name: str, raw: str) -> Optional[dict]:
        """
        Intenta extraer la intención y el valor sin llamar al LLM.
        Retorna {"intent": ..., "value": ...} o None si necesita IA.
        """
        v = raw.strip()
        vl = v.lower()

        # Cancelación global
        if any(w in vl for w in self._CANCEL_PHRASES) and len(vl) < 30:
            return {"intent": "CANCEL", "value": None}

        # Pregunta de "¿por qué?" / "¿para qué?"
        if self._WHY_PATTERN.search(vl) and len(vl) < 60:
            return {"intent": "WHY", "value": None}

        # Skip exacto
        if vl in self._SKIP_EXACT:
            return {"intent": "SKIP", "value": None}

        # ─── Por tipo de campo ───
        if state_name == "esperando_correo":
            match = re.search(r"[\w.+-]+@[\w.-]+\.\w{2,}", v)
            if match:
                return {"intent": "ANSWER", "value": match.group(0).lower()}
            return None  # Necesita IA para limpiar

        if state_name in ("esperando_asegurado",):
            if self._NO_SE_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "No se"}
            if self._YES_WORDS.search(vl) and not self._NO_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "Si"}
            if self._NO_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "No"}
            return None

        if state_name == "esperando_autorizacion":
            if self._YES_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "Si"}
            if self._NO_WORDS.search(vl) and "todo bien" not in vl:
                return {"intent": "ANSWER", "value": "No"}
            return None

        if state_name.startswith("esperando_p"):
            digits = re.sub(r"\D", "", v)
            if digits:
                num = int(digits)
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
            # Cualquier texto es un comentario válido
            return {"intent": "ANSWER", "value": v if vl not in ("no", "nada", "ninguno") else None}

        return None

    # ─── Slow Path: extracción con IA ───
    async def extract_with_ai(self, state_name: str, user_text: str) -> dict:
        """Usa el LLM para clasificar la intención y extraer el valor."""
        field_type = "EMAIL" if state_name == "esperando_correo" else \
                     "BOOLEAN (Si/No/No se)" if state_name in ("esperando_asegurado", "esperando_autorizacion") else \
                     "NUMBER (1-5)" if state_name.startswith("esperando_p") else \
                     "NUMBER (1-10)" if state_name == "esperando_nps" else \
                     "TEXT (Comentario)"

        prompt = f"""Analiza la respuesta del usuario a: "{FORM_QUESTIONS.get(state_name, '')}".

Categorízala:
- ANSWER: Responde (errores ortográficos, símbolos, frases coloquiales como 'si normal', 'chi', 'claro que si' ALL cuentan como respuestas).
- WHY: Pregunta por el motivo.
- SKIP: Rechaza explícitamente participar.
- INTERRUPT: Cambia de tema radicalmente.

Si es ANSWER, extrae valor puro ({field_type}):
- EMAIL: Solo la dirección (ej: "ok, x@y.com" -> x@y.com)
- BOOLEAN: 'Si', 'No' o 'No se'
- NUMBER: Solo el dígito
- TEXT: Texto limpio

JSON: {{"intent": "CATEGORIA", "value": "VALOR"}}

USUARIO: "{user_text}" """

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

    # ─── Método principal ───
    async def extract(self, state_name: str, user_text: str) -> dict:
        """Intenta fast-path y cae a IA si es necesario."""
        fast = self.try_fast_extract(state_name, user_text)
        if fast is not None:
            logger.debug("Survey fast-path: state=%s → %s", state_name, fast)
            return fast

        logger.debug("Survey slow-path (AI): state=%s, text=%s", state_name, user_text[:50])
        return await self.extract_with_ai(state_name, user_text)


# ═══════════════════════════════════════════════════════════════
# Closure Score
# ═══════════════════════════════════════════════════════════════
CLOSURE_EVAL_PROMPT = """Evalúa si el siguiente mensaje del usuario indica intención de terminar/despedirse de la conversación.
Responde SOLO con un número entero del 0 al 100 (0 = no se despide, 100 = claramente se despide).
Ejemplos de despedida: "gracias chau", "ya me voy", "hasta luego", "bye", "eso era todo"
Ejemplos de NO despedida: "gracias por el dato" seguido de otra pregunta, "ok gracias, y otra cosa..."

Mensaje: """


async def evaluate_closure_score(client: AsyncOpenAI, model: str, user_text: str) -> int:
    """Evalúa la intención de despedida del usuario (0-100)."""
    try:
        response = await client.responses.create(
            model=model,
            input=CLOSURE_EVAL_PROMPT + user_text,
            instructions="Responde SOLO con un número entero del 0 al 100.",
        )
        score_text = response.output_text.strip()
        digits = re.sub(r"\D", "", score_text)
        return min(100, max(0, int(digits))) if digits else 0
    except Exception:
        logger.exception("Error evaluando closure score")
        return 0


# ═══════════════════════════════════════════════════════════════
# Orquestador principal del flujo de cierre
# ═══════════════════════════════════════════════════════════════
async def advance_closing_flow(
    session: AsyncSession,
    state: ConversationState,
    user_text: str,
    openai_client: AsyncOpenAI,
    openai_model: str,
) -> Optional[str]:
    """
    Evalúa la compuerta de cierre y gestiona el formulario.
    """
    # ─── Si ya está en modo collecting → procesar respuesta del formulario ───
    if state.mode in ("collecting_usability", "collecting_profile"):
        return await _process_form_response(session, state, user_text, openai_client, openai_model)

    # ─── Si está en closing pero sin formulario activo → verificar pendientes ───
    if state.mode == "closing":
        pending = await _check_pending_form(session, state)
        if pending:
            return pending
        return None

    # ─── active_chat → evaluar si se despide ───
    if state.mode == "active_chat":
        score = await evaluate_closure_score(openai_client, openai_model, user_text)
        state.closure_score = score
        logger.info(
            "Closure score=%d para user=%d (turns=%d)",
            score, state.usuario_id, state.turns_since_last_prompt,
        )

        if score >= 60:
            # Intentar iniciar formulario
            addon = await _try_start_form(session, state)
            return addon

    return None


async def _try_start_form(session: AsyncSession, state: ConversationState) -> Optional[str]:
    """Intenta iniciar el formulario de usabilidad si no lo ha completado."""
    # Verificar si hay un formulario activo
    result = await session.execute(
        text("SELECT id FROM formularios WHERE activo = TRUE ORDER BY id LIMIT 1")
    )
    form = result.fetchone()
    if not form:
        return None

    # Verificar si ya completó el formulario
    result = await session.execute(
        text("""
            SELECT estado_actual FROM formulario_en_progreso
            WHERE usuario_id = :uid AND formulario_id = :fid
        """),
        {"uid": state.usuario_id, "fid": form.id},
    )
    progress = result.fetchone()

    if progress and progress.estado_actual == "completado":
        return None  # Ya completó

    if progress:
        # Tiene progreso parcial → retomar
        current_state = progress.estado_actual
        state.mode = "collecting_usability"
        state.awaiting_question_code = current_state
        question = FORM_QUESTIONS.get(current_state, "")
        return f"¡Antes de irte! 😊 Me quedaron unas preguntitas pendientes de la última vez que me ayudan mucho a mejorar NutriBot. ¿Te parece si las completamos rápido?\n\n{question}"

    # No tiene progreso → crear nuevo
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
    question = FORM_QUESTIONS[first_state]
    return f"¡Muchas gracias por chatear conmigo! 🙏 Antes de despedirnos, me encantaría saber qué te pareció la experiencia. Son unas preguntitas rápidas.\n\n{question}"


async def _check_pending_form(session: AsyncSession, state: ConversationState) -> Optional[str]:
    """Verifica si hay un formulario pendiente y retoma amablemente."""
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


# ═══════════════════════════════════════════════════════════════
# Procesador de respuestas del formulario
# ═══════════════════════════════════════════════════════════════
async def _process_form_response(
    session: AsyncSession,
    state: ConversationState,
    user_text: str,
    openai_client: AsyncOpenAI,
    openai_model: str,
) -> Optional[str]:
    """Procesa la respuesta del usuario al formulario activo."""
    current_state = state.awaiting_question_code

    if not current_state or current_state not in FORM_QUESTIONS:
        state.mode = "active_chat"
        state.awaiting_question_code = None
        return None

    # ─── Extracción (fast-path + fallback IA) ───
    extractor = SurveyResponseExtractor(openai_client, openai_model)
    result_data = await extractor.extract(current_state, user_text)

    intent = result_data.get("intent", "ANSWER").upper()
    extracted_value = result_data.get("value")

    logger.info("advance_closing: intent=%s, extracted=%s for user=%s", intent, extracted_value, state.usuario_id)

    # Cancelación total
    if intent == "CANCEL":
        state.mode = "active_chat"
        state.awaiting_question_code = None
        return "¡Entendido! Lo dejamos por ahora. Si tienes alguna duda, estaré por aquí. 🍏"

    # Interrupción (cambio de tema)
    if intent == "INTERRUPT":
        state.mode = "active_chat"
        return None

    # ─── Obtener progreso del formulario ───
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

    # ─── WHY / SKIP → Persuasión ───
    if intent in ("WHY", "SKIP"):
        persuasion_key = f"{field_key}_persuaded"

        if not parciales.get(persuasion_key):
            parciales[persuasion_key] = True
            await session.execute(
                text("UPDATE formulario_en_progreso SET respuestas_parciales = :p WHERE usuario_id = :uid"),
                {"p": json.dumps(parciales), "uid": state.usuario_id},
            )
            return PERSUASION_MESSAGES.get(current_state, PERSUASION_DEFAULT)

        # Ya persuadido → saltar bloque identidad
        if current_state in ("esperando_correo", "esperando_asegurado"):
            next_state = "esperando_p1"
            await session.execute(
                text("UPDATE formulario_en_progreso SET estado_actual = :next WHERE usuario_id = :uid"),
                {"next": next_state, "uid": state.usuario_id},
            )
            state.awaiting_question_code = next_state
            return f"Entiendo, no te preocupes. 😊 ¿Me podrías ayudar al menos a mejorar NutriBot con unas preguntitas rápidas sobre tu experiencia? Me ayudará mucho a darte un mejor servicio.\n\n{FORM_QUESTIONS[next_state]}"

        cleaned_value = None
    else:
        # ─── ANSWER → usar valor extraído ───
        cleaned_value = extracted_value

        # Sanity checks finales
        if current_state == "esperando_correo" and ("@" not in str(cleaned_value) or "." not in str(cleaned_value)):
            return "Hmm, ese correo no parece válido 🤔 ¿Podrías revisarlo? Ejemplo: tunombre@gmail.com"

        if (current_state.startswith("esperando_p") or current_state == "esperando_nps") and not str(cleaned_value).isdigit():
            max_val = "10" if current_state == "esperando_nps" else "5"
            return f"¡Muchas gracias por tu comentario! 😊 Pero para poder registrar tu experiencia, ¿me podrías dar un número del 1 al {max_val}?"

    # ─── Guardar respuesta ───
    if cleaned_value is not None:
        parciales[field_key] = cleaned_value

    # ─── Avanzar al siguiente estado ───
    current_idx = FORM_STATES_ORDER.index(current_state)
    next_state = None
    media_addon = ""

    for i in range(current_idx + 1, len(FORM_STATES_ORDER)):
        candidate = FORM_STATES_ORDER[i]
        if candidate == "esperando_p8" and not progress.uso_audio:
            media_addon += "\n\n🎙️ *PD:* Veo que aún no hemos probado enviarnos audios. ¡Es súper práctico para consultas largas! Si gustas, puedes probarlo en nuestra próxima charla. 😊"
            continue
        if candidate == "esperando_p9" and not progress.uso_imagen:
            media_addon += "\n\n📸 *PD:* Por cierto, ¿viste que puedes enviarme fotos de tu comida? Me ayuda mucho a darte consejos más visuales. ¡Anímate a probarlo pronto! 🥦"
            continue
        next_state = candidate
        break

    if next_state is None:
        # ─── Formulario completado ───
        user_phone = (await session.execute(
            text("SELECT numero_whatsapp FROM usuarios WHERE id = :uid"),
            {"uid": state.usuario_id},
        )).scalar()

        has_essential = all([
            parciales.get("correo") and "@" in parciales.get("correo"),
            parciales.get("edad"),
            parciales.get("asegurado"),
        ])

        if has_essential:
            await session.execute(
                text("""
                    UPDATE formulario_en_progreso
                    SET estado_actual = 'completado',
                        respuestas_parciales = :parciales,
                        completado_en = :now,
                        actualizado_en = :now
                    WHERE usuario_id = :uid
                """),
                {"uid": state.usuario_id, "parciales": json.dumps(parciales), "now": get_now_peru()},
            )

            await session.execute(
                text("""
                    INSERT INTO respuestas_formulario
                        (formulario_id, usuario_id, correo, telefono, edad,
                         asegurado_essalud, comentario, autorizo_uso_investigacion, puntaje_nps)
                    VALUES (:fid, :uid, :correo, :tel, :edad, :aseg, :com, :aut, :nps)
                    ON CONFLICT (formulario_id, usuario_id) DO NOTHING
                """),
                {
                    "fid": progress.formulario_id,
                    "uid": state.usuario_id,
                    "correo": parciales.get("correo"),
                    "tel": user_phone,
                    "edad": int(parciales["edad"]) if parciales.get("edad") else None,
                    "aseg": parciales.get("asegurado"),
                    "com": parciales.get("comentario"),
                    "aut": parciales.get("autorizacion", "").lower().startswith("s") if parciales.get("autorizacion") else None,
                    "nps": int(parciales["nps"]) if parciales.get("nps") else None,
                },
            )
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return "🎉 ¡Listo, muchas gracias por completar la encuesta! Tus respuestas nos ayudan mucho a mejorar NutriBot. ¡Que tengas un excelente día! 💪🥦"
        else:
            await session.execute(
                text("""
                    UPDATE formulario_en_progreso
                    SET estado_actual = 'parcialmente_completado',
                        respuestas_parciales = :parciales,
                        actualizado_en = :now
                    WHERE usuario_id = :uid
                """),
                {"uid": state.usuario_id, "parciales": json.dumps(parciales), "now": get_now_peru()},
            )
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return "¡Muchas gracias por tus respuestas! 😊 He guardado tus comentarios para seguir mejorando. En nuestra próxima charla, si gustas, terminaremos de completar un par de datitos que nos faltan. ¡Un abrazo! ✨"

    # ─── Avanzar estado ───
    await session.execute(
        text("""
            UPDATE formulario_en_progreso
            SET estado_actual = :next,
                respuestas_parciales = :parciales,
                actualizado_en = :now
            WHERE usuario_id = :uid
        """),
        {"uid": state.usuario_id, "next": next_state, "parciales": json.dumps(parciales), "now": get_now_peru()},
    )

    state.awaiting_question_code = next_state
    state.turns_since_last_prompt = 0
    question = FORM_QUESTIONS[next_state]
    prefix = f"{media_addon}\n\n" if media_addon else ""
    return f"✅ ¡Anotado! {prefix}Siguiente pregunta:\n\n{question}"
