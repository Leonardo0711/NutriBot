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

# Preguntas del formulario con sus textos amigables
FORM_QUESTIONS: dict[str, str] = {
    "esperando_correo": "📧 Para empezar, ¿me podrías compartir tu correo electrónico?",
    "esperando_asegurado": "🏥 ¿Eres asegurado/a de EsSalud? (Sí, No, o No sé)",
    "esperando_p1": "Del 1 al 5, ¿qué tan fácil te resultó usar el chatbot? 🤔",
    "esperando_p2": "Del 1 al 5, ¿qué tan claras fueron mis respuestas? 💬",
    "esperando_p3": "Del 1 al 5, ¿la información nutricional que te di fue útil? 🥗",
    "esperando_p4": "Del 1 al 5, ¿qué tan rápido respondí a tus preguntas? ⚡",
    "esperando_p5": "Del 1 al 5, ¿te sentiste cómodo/a hablando conmigo? 😊",
    "esperando_p6": "Del 1 al 5, ¿confías en la información nutricional que te doy? 🔬",
    "esperando_p7": "Del 1 al 5, ¿el chatbot se adaptó bien a tus necesidades? 🎯",
    "esperando_p8": "Del 1 al 5, ¿qué tan buena fue tu experiencia enviándome audios? 🎙️",
    "esperando_p9": "Del 1 al 5, ¿qué te pareció poder enviarme fotos de tu comida? 📸",
    "esperando_p10": "Del 1 al 5, ¿recomendarías este chatbot a amigos o familia? 👨‍👩‍👧‍👦",
    "esperando_nps": "Del 1 al 10, ¿qué probabilidad hay de que recomiendes NutriBot? ⭐",
    "esperando_comentario": "💭 ¿Tienes algún comentario o sugerencia para mejorar?",
    "esperando_autorizacion": "📋 Por último, ¿autorizas que usemos tus respuestas de forma anónima para mejorar NutriBot? (Sí o No)",
}

# ─── Validaciones por campo ───
def _validate_field(state_name: str, raw_value: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Valida la respuesta del usuario para el campo actual.
    Returns: (is_valid, cleaned_value, error_message)
    """
    v = raw_value.strip()

    if state_name == "esperando_correo":
        if re.match(r"^\S+@\S+\.\S+$", v):
            return True, v.lower(), None
        return False, None, "Hmm, ese correo no parece válido 🤔 ¿Podrías revisarlo? Ejemplo: tunombre@gmail.com"

    elif state_name == "esperando_asegurado":
        vl = v.lower().strip()
        if any(x in vl for x in ["sí", "si", "yes"]):
            return True, "Si", None
        elif "no sé" in vl or "no se" in vl or "nose" in vl:
            return True, "No se", None
        elif "no" in vl:
            return True, "No", None
        return False, None, "¿Podrías responder 'Sí', 'No' o 'No sé'? 🏥"

    elif state_name.startswith("esperando_p"):
        try:
            score = int(re.sub(r"\D", "", v))
            if 1 <= score <= 5:
                return True, str(score), None
        except ValueError:
            pass
        return False, None, "Solo necesito un número del 1 al 5 ✋"

    elif state_name == "esperando_nps":
        try:
            score = int(re.sub(r"\D", "", v))
            if 1 <= score <= 10:
                return True, str(score), None
        except ValueError:
            pass
        return False, None, "Solo necesito un número del 1 al 10 ⭐"

    elif state_name == "esperando_comentario":
        return True, v if v.lower() not in ("no", "nada", "ninguno") else None, None

    elif state_name == "esperando_autorizacion":
        vl = v.lower().strip()
        if any(x in vl for x in ["sí", "si", "yes"]):
            return True, "Si", None
        elif "no" in vl:
            return True, "No", None
        return False, None, "¿Sí o No? 📋"

    return True, v, None


# ─── Closure Score ───
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
        return f"¡Antes de irte! 😊 Me quedaron unas preguntitas pendientes de la vez pasada, ¿te parece si las completamos rápido?\n\n{question}"

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


async def _process_form_response(
    session: AsyncSession,
    state: ConversationState,
    user_text: str,
    openai_client: AsyncOpenAI,
    openai_model: str,
) -> Optional[str]:
    """Procesa la respuesta del usuario al formulario activo con detección de interrupciones."""
    current_state = state.awaiting_question_code
    vl = user_text.lower().strip()
    
    if not current_state or current_state not in FORM_QUESTIONS:
        state.mode = "active_chat"
        state.awaiting_question_code = None
        return None

    # Detectar si el usuario cambió de tema (SEMÁNTICO)
    interruption_prompt = f"""Analiza si el usuario está respondiendo a la pregunta de usabilidad '{FORM_QUESTIONS.get(current_state, '')}' o si ha cambiado de tema para preguntar otra cosa o pedir ayuda.
    Responde SOLO: 'ANSWER' o 'INTERRUPTION'.
    
    USUARIO: "{user_text}"
    """
    
    int_resp = await openai_client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "system", "content": "Eres un detector de cambios de tema."},
                  {"role": "user", "content": interruption_prompt}],
        max_tokens=5,
        temperature=0
    )
    is_interruption = "INTERRUPTION" in int_resp.choices[0].message.content.strip().upper()

    if is_interruption and len(vl) > 10:
        # Pausamos el formulario y devolvemos control al chat libre
        state.mode = "active_chat"
        # Mantenemos el question_code para retomar luego
        return None

    # Detectar cancelación global
    cancel_words = ["cancelar", "para", "ya no", "no quiero responder", "detener", "salir"]
    if any(w in vl for w in cancel_words) and len(vl) < 30:
        state.mode = "active_chat"
        state.awaiting_question_code = None
        return "¡Entendido! Lo dejamos por ahora. Si tienes alguna duda, estaré por aquí. 🍏"

    # Obtener el estado actual del formulario para revisar persuasiones
    result = await session.execute(
        text("""
            SELECT id, formulario_id, respuestas_parciales, uso_audio, uso_imagen
            FROM formulario_en_progreso WHERE usuario_id = :uid
        """),
        {"uid": state.usuario_id},
    )
    progress = result.fetchone()
    if not progress:
        state.mode = "active_chat"
        return None

    parciales = progress.respuestas_parciales or {}

    # Detectar rechazo/salto suave (solo para campos específicos cortos)
    skip_words = ["no", "paso", "saltar", "skip", "no quiero", "siguiente"]
    is_skip = vl in skip_words or (vl.startswith("no") and len(vl) <= 12)

    cleaned_value = None

    if is_skip:
        # Lógica de persuasión para campos clave
        field_key = current_state.replace("esperando_", "")
        persuasion_key = f"{field_key}_persuaded"

        if not parciales.get(persuasion_key):
            # Aún no persuadimos, intentarlo
            parciales[persuasion_key] = True
            
            # Actualizar DB con el intento
            await session.execute(
                text("UPDATE formulario_en_progreso SET respuestas_parciales = :p WHERE usuario_id = :uid"),
                {"p": json.dumps(parciales), "uid": state.usuario_id}
            )

            if current_state == "esperando_correo":
                return "Solo pedimos el correo para avisarte de próximas campañas de salud o nutrición cerca a ti. ✉️ ¿Te animas a compartirlo?"
            elif current_state == "esperando_asegurado":
                return "Saber si eres asegurado nos ayuda a darte información sobre servicios específicos de EsSalud 🏥. ¿Te gustaría comentarlo?"
            else:
                return "Entiendo, pero tus respuestas nos ayudan muchísimo a mejorar a NutriBot para todos. 🌟 ¿Te animas a darnos este dato?"
        
        # Si ya fue persuadido y estamos en el bloque de identidad, saltar al bloque de mejora
        is_identity_block = current_state in ("esperando_correo", "esperando_asegurado")
        if is_identity_block:
            next_state = "esperando_p1"
            await session.execute(
                text("UPDATE formulario_en_progreso SET estado_actual = :next WHERE usuario_id = :uid"),
                {"next": next_state, "uid": state.usuario_id}
            )
            state.awaiting_question_code = next_state
            return f"Entiendo, no te preocupes. 😊 ¿Me podrías ayudar al menos a mejorar NutriBot con unas preguntitas rápidas sobre tu experiencia? Solo te tomará un minuto.\n\n{FORM_QUESTIONS[next_state]}"
        
        is_valid = True
    else:
        # Validar respuesta normal (usando LLM para escalas numéricas si no es obvio)
        cleaned_value = None
        
        try:
            if current_state.startswith("esperando_p") or current_state == "esperando_nps":
                max_val = 10 if current_state == "esperando_nps" else 5
                extract_prompt = f"""Analiza la respuesta del usuario para una calificación de 1 a {max_val}.
                1. Si el usuario da un número claro (ej: 'un 5', 'pongo 4', '5 estrellas'), responde SOLO el número.
                2. Si el usuario solo da un comentario positivo o negativo sin número (ej: 'si mucho', 'fue genial', 'pésimo'), responde 'COMMENT_ONLY'.
                3. De lo contrario, responde 'NONE'.
                
                Responde SOLO una palabra: el número, 'COMMENT_ONLY' o 'NONE'.
                
                USUARIO: "{user_text}" """
                
                ext_resp = await openai_client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "system", "content": "Analista de encuestas nutricionales."},
                              {"role": "user", "content": extract_prompt}],
                    max_tokens=10,
                    temperature=0
                )
                raw_ext = ext_resp.choices[0].message.content.strip().upper()
                
                if raw_ext.isdigit() and 1 <= int(raw_ext) <= max_val:
                    is_valid, cleaned_value, error_msg = True, raw_ext, None
                elif "COMMENT_ONLY" in raw_ext:
                    # El usuario fue amable pero no dio número -> Agradecer y pedir número
                    return f"¡Muchas gracias por tu comentario! 😊 Me alegra saber que te sientes así. Para poder registrarlo oficialmente en mi sistema, ¿me podrías confirmar qué puntaje le darías del 1 al {max_val}?"
                else:
                    return f"¡Me alegra muchísimo oír eso! 😊 Pero para poder registrar tu experiencia, ¿me podrías dar un número del 1 al {max_val}?"
            else:
                is_valid, cleaned_value, error_msg = _validate_field(current_state, user_text)
                if not is_valid:
                    return error_msg
        except Exception as e:
            logger.error(f"Error en extracción semántica del formulario: {e}")
            # Fallback a validación tradicional
            is_valid, cleaned_value, error_msg = _validate_field(current_state, user_text)
            if not is_valid:
                return error_msg

    # Guardar respuesta exitosa (sea valor real o salto)
    field_key = current_state.replace("esperando_", "")
    if cleaned_value is not None:
        parciales[field_key] = cleaned_value

    # Avanzar al siguiente estado
    current_idx = FORM_STATES_ORDER.index(current_state)
    next_state = None

    for i in range(current_idx + 1, len(FORM_STATES_ORDER)):
        candidate = FORM_STATES_ORDER[i]
        # Saltar p8 si no usó audio
        if candidate == "esperando_p8" and not progress.uso_audio:
            continue
        # Saltar p9 si no usó imagen
        if candidate == "esperando_p9" and not progress.uso_imagen:
            continue
        next_state = candidate
        break

    if next_state is None:
        # Recuperar teléfono del usuario desde la DB directamente
        user_phone = (await session.execute(
            text("SELECT numero_whatsapp FROM usuarios WHERE id = :uid"),
            {"uid": state.usuario_id}
        )).scalar()

        # Formulario completado
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

        # Guardar respuestas finales
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
                "correo": parciales.get("correo", "no_proporcionado@na.com"),
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
        return "🎉 ¡Listo, muchas gracias por completar la encuesta! Tus respuestas nos ayudan mucho a mejorar. ¡Que tengas un excelente día! 💪🥦"

    # Avanzar estado
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
    return f"✅ ¡Anotado! Siguiente pregunta:\n\n{question}"
