"""
Nutribot Backend - SurveyService
Guided usability survey with structured interactive responses.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Any

from openai import AsyncOpenAI
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.interactive_message_factory import (
    build_scale_list,
    build_yes_no_buttons,
)
from domain.entities import ConversationState, NormalizedMessage
from domain.reply_objects import BotReply
from domain.value_objects import MessageType
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

CONSENT_STATE = "esperando_consentimiento_encuesta"

FORM_STATES_ORDER = [
    "esperando_correo",
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
    "esperando_correo": "Me encantaria seguir en contacto. Podrias compartir tu correo?",
    "esperando_p1": "Que tan realista y atractiva te parecio mi personalidad?",
    "esperando_p2": "Que tan bien explique mi proposito y alcance?",
    "esperando_p3": "Que tan facil te pareci de navegar?",
    "esperando_p4": "Que tan bien sentiste que te entendi?",
    "esperando_p5": "Que tan utiles e informativas te parecieron mis respuestas?",
    "esperando_p6": "Que tan bien maneje errores o equivocaciones?",
    "esperando_p7": "Que tan facil fui de usar?",
    "esperando_audio_optin": "Te gustaria probar mi modo audio para luego evaluarlo?",
    "esperando_audio_prueba": "Enviame un audio cortito para poder evaluar esa funcion.",
    "esperando_p8": "Que tan clara te parecio mi experiencia con audio?",
    "esperando_imagen_optin": "Te gustaria probar reconocimiento de imagenes para luego evaluarlo?",
    "esperando_imagen_prueba": "Enviame una foto o imagen para poder evaluar esa funcion.",
    "esperando_p9": "Que tan bien reconoci el contexto de la imagen?",
    "esperando_p10": "Que tan bien me enfoque solo en nutricion?",
    "esperando_nps": "Del 1 al 10, que tan probable es que me recomiendes?",
    "esperando_comentario": "(Opcional) Que te gusto o no te gusto?",
    "esperando_autorizacion": "Autorizas el uso anonimo y agregado de tus respuestas para investigacion? (Si autorizo / No autorizo)",
}

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

    _SKIP_EXACT = {"paso", "saltar", "skip", "no quiero", "siguiente", "prefiero no", "omitir"}
    _CANCEL_PHRASES = {"cancelar todo", "detener bot", "salir de todo", "stop survey"}
    _YES_WORDS = re.compile(r"(?:\b|^)(si|yes|claro|dale|ok|de acuerdo|acepto|autorizo)(?:\b|$)", re.IGNORECASE)
    _NO_WORDS = re.compile(r"(?:\b|^)(no|nop|nel|rechazo|no autorizo)(?:\b|$)", re.IGNORECASE)
    _WHY_PATTERN = re.compile(r"(para que|por que|que uso|que finalidad)", re.IGNORECASE)
    _NUTRITION_HINTS = (
        "menu", "receta", "dieta", "imc", "calorias", "desayuno", "almuerzo",
        "cena", "peso", "talla", "proteina", "carbohidratos", "grasa",
        "nutricion", "plan alimentario", "porciones", "macros",
    )

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
            if self._NO_WORDS.search(vl):
                return {"intent": "SKIP", "value": None}
            match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", v)
            if match:
                return {"intent": "ANSWER", "value": match.group(0)}
            if self._contains_nutrition_hint(vl):
                return {"intent": "INTERRUPT", "value": None}
            return None

        if state_name in ("esperando_audio_optin", "esperando_imagen_optin", "esperando_autorizacion", CONSENT_STATE):
            if self._YES_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "Si"}
            if self._NO_WORDS.search(vl):
                return {"intent": "ANSWER", "value": "No"}
            return None

        if state_name.startswith("esperando_p") or state_name == "esperando_nps":
            m = re.match(r"^\s*(\d{1,2})\s*$", v)
            if m:
                return {"intent": "ANSWER", "value": m.group(1)}
            return None

        if state_name == "esperando_comentario":
            if self._contains_nutrition_hint(vl):
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
    def __init__(self, openai_client: AsyncOpenAI, openai_model: str):
        self.extractor = SurveyResponseExtractor(openai_client, openai_model)

    @staticmethod
    def _parse_int(value: Any, minimum: int, maximum: int) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(str(value).strip())
            if minimum <= number <= maximum:
                return number
        except Exception:
            return None
        return None

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
        text_value = str(value).strip().lower()
        if text_value in {"si", "yes", "si autorizo", "autorizo", "acepto"}:
            return True
        if text_value in {"no", "no autorizo", "rechazo"}:
            return False
        return None

    @staticmethod
    def _is_audio_message(normalized: NormalizedMessage) -> bool:
        return normalized.content_type in (MessageType.AUDIO, MessageType.PTT) or bool(normalized.used_audio)

    @staticmethod
    def _is_image_message(normalized: NormalizedMessage) -> bool:
        return normalized.content_type == MessageType.IMAGE or bool(normalized.image_base64)

    def _build_question_reply(self, state_name: str, prefix: str = "") -> BotReply:
        question = FORM_QUESTIONS.get(state_name, "")
        if state_name in _SCALE_PREFIX and state_name != "esperando_nps":
            payload = build_scale_list(question, _SCALE_PREFIX[state_name], 1, 5, "Selecciona una calificacion")
            if prefix:
                payload["body"] = f"{prefix}\n\n{payload['body']}"
            return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

        if state_name == "esperando_nps":
            payload = build_scale_list(question, _SCALE_PREFIX[state_name], 1, 10, "Selecciona una calificacion")
            if prefix:
                payload["body"] = f"{prefix}\n\n{payload['body']}"
            return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

        if state_name == "esperando_audio_optin":
            payload = build_yes_no_buttons(
                body=question + "\n*(Tambien puedes responder Si o No)*",
                button_yes_id="survey:audio_optin:yes",
                button_no_id="survey:audio_optin:no",
                yes_label="Si, probar",
                no_label="Omitir",
            )
            if prefix:
                payload["body"] = f"{prefix}\n\n{payload['body']}"
            return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

        if state_name == "esperando_imagen_optin":
            payload = build_yes_no_buttons(
                body=question + "\n*(Tambien puedes responder Si o No)*",
                button_yes_id="survey:image_optin:yes",
                button_no_id="survey:image_optin:no",
                yes_label="Si, probar",
                no_label="Omitir",
            )
            if prefix:
                payload["body"] = f"{prefix}\n\n{payload['body']}"
            return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

        if state_name == "esperando_autorizacion":
            payload = build_yes_no_buttons(
                body=question,
                button_yes_id="survey:auth:yes",
                button_no_id="survey:auth:no",
                yes_label="Si autorizo",
                no_label="No autorizo",
            )
            if prefix:
                payload["body"] = f"{prefix}\n\n{payload['body']}"
            return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

        text_msg = f"{prefix}\n\n{question}" if prefix else question
        return BotReply(text=text_msg, content_type="text")

    def _build_consent_reply(self) -> BotReply:
        payload = build_yes_no_buttons(
            body=(
                "Muchas gracias por chatear conmigo. Me encantaria hacerte unas breves "
                "preguntas del formulario de satisfaccion.\n\n"
                "Este formulario es aparte de tu perfil nutricional.\n\n"
                "Estas de acuerdo?\n*(Tambien puedes escribir Si o No)*"
            ),
            button_yes_id="survey:consent:yes",
            button_no_id="survey:consent:no",
            yes_label="Si",
            no_label="No",
        )
        return BotReply(text=payload["body"], content_type="interactive_list", payload_json=payload)

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

        audio_evaluado = self._parse_int(parciales.get("p8"), 1, 5) is not None
        audio_no_aplica = bool(parciales.get("p8_no_aplica"))
        imagen_evaluada = self._parse_int(parciales.get("p9"), 1, 5) is not None
        imagen_no_aplica = bool(parciales.get("p9_no_aplica"))

        resp_result = await session.execute(
            text(
                """
                INSERT INTO respuestas_formulario (
                    formulario_id, usuario_id, correo, correo_proporcionado,
                    comentario, autorizo_uso_investigacion, puntaje_nps,
                    audio_evaluado, audio_no_aplica, imagen_evaluada, imagen_no_aplica
                )
                VALUES (
                    :fid, :uid, :correo, :correo_proporcionado,
                    :comentario, :autorizo, :nps,
                    :audio_evaluado, :audio_no_aplica, :imagen_evaluada, :imagen_no_aplica
                )
                ON CONFLICT (formulario_id, usuario_id) DO UPDATE SET
                    correo = EXCLUDED.correo,
                    correo_proporcionado = EXCLUDED.correo_proporcionado,
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
            effective = projected_interactions_count if projected_interactions_count is not None else state.meaningful_interactions_count
            if effective >= 5 and state.usability_completion_pct < 100:
                state.awaiting_question_code = CONSENT_STATE
                state.meaningful_interactions_count = 0
                return self._build_consent_reply()
        return None

    async def _handle_consent_response(
        self,
        session: AsyncSession,
        state: ConversationState,
        normalized: NormalizedMessage,
    ) -> Optional[BotReply]:
        result = await self.extractor.extract(CONSENT_STATE, normalized.text, normalized.interactive_id)
        intent = result.get("intent", "ANSWER").upper()
        value = str(result.get("value") or "").strip().lower()

        if intent == "INTERRUPT":
            state.awaiting_question_code = None
            state.mode = "active_chat"
            state.meaningful_interactions_count = 0
            return None

        if value in {"si", "yes"}:
            state.awaiting_question_code = None
            return await self._try_start_form(session, state)

        if value in {"no"}:
            state.awaiting_question_code = None
            state.mode = "active_chat"
            state.meaningful_interactions_count = 0
            return BotReply(text="Entendido, seguimos conversando sin problema.", content_type="text")

        return self._build_consent_reply()

    async def _try_start_form(self, session: AsyncSession, state: ConversationState) -> Optional[BotReply]:
        result = await session.execute(
            text("SELECT id FROM formularios WHERE activo = TRUE ORDER BY version DESC, id DESC LIMIT 1")
        )
        form = result.fetchone()
        if not form:
            state.meaningful_interactions_count = 0
            return None

        progress = await self._load_active_progress(session, state.usuario_id)
        if progress and progress.formulario_id == form.id:
            if progress.estado_actual == "completado":
                state.meaningful_interactions_count = 0
                return None
            state.mode = "collecting_usability"
            state.awaiting_question_code = progress.estado_actual
            state.meaningful_interactions_count = 0
            return self._build_question_reply(progress.estado_actual, prefix="Retomemos el formulario pendiente.")

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
        result = await self.extractor.extract(current_state, normalized.text, normalized.interactive_id)
        intent = str(result.get("intent", "ANSWER")).upper()
        value = result.get("value")

        if intent == "CANCEL":
            state.mode = "active_chat"
            state.awaiting_question_code = None
            return BotReply(text="Entendido, dejamos el formulario por ahora.", content_type="text")
        if intent == "INTERRUPT":
            # Mantenemos el modo y el estado actual para que el orquestador pueda re-anclar después.
            # Solo retornamos None para que el orquestador sepa que debe responder con el LLM.
            return None

        audio_test_requested = bool(getattr(progress, "audio_test_requested", False))
        audio_test_completed = bool(getattr(progress, "audio_test_completed", False))
        audio_test_declined = bool(getattr(progress, "audio_test_declined", False))
        image_test_requested = bool(getattr(progress, "image_test_requested", False))
        image_test_completed = bool(getattr(progress, "image_test_completed", False))
        image_test_declined = bool(getattr(progress, "image_test_declined", False))
        uso_audio = bool(getattr(progress, "uso_audio", False))
        uso_imagen = bool(getattr(progress, "uso_imagen", False))

        if current_state == "esperando_audio_optin":
            txt = str(value or "").lower()
            if txt in {"si", "yes"}:
                audio_test_requested = True
                next_state = "esperando_audio_prueba"
            elif txt in {"no"} or intent == "SKIP":
                audio_test_declined = True
                parciales["p8_no_aplica"] = True
                parciales.pop("p8", None)
                next_state = "esperando_imagen_optin"
            else:
                return self._build_question_reply(current_state)
        elif current_state == "esperando_audio_prueba":
            if self._is_audio_message(normalized):
                uso_audio = True
                audio_test_completed = True
                next_state = "esperando_p8"
            elif intent == "SKIP" or str(value or "").lower() in {"no"}:
                audio_test_declined = True
                parciales["p8_no_aplica"] = True
                parciales.pop("p8", None)
                next_state = "esperando_imagen_optin"
            else:
                return BotReply(
                    text="Para evaluar audio, enviame una nota de voz real. Si prefieres omitirlo, escribe 'omitir'.",
                    content_type="text",
                )
        elif current_state == "esperando_imagen_optin":
            txt = str(value or "").lower()
            if txt in {"si", "yes"}:
                image_test_requested = True
                next_state = "esperando_imagen_prueba"
            elif txt in {"no"} or intent == "SKIP":
                image_test_declined = True
                parciales["p9_no_aplica"] = True
                parciales.pop("p9", None)
                next_state = "esperando_p10"
            else:
                return self._build_question_reply(current_state)
        elif current_state == "esperando_imagen_prueba":
            if self._is_image_message(normalized):
                uso_imagen = True
                image_test_completed = True
                next_state = "esperando_p9"
            elif intent == "SKIP" or str(value or "").lower() in {"no"}:
                image_test_declined = True
                parciales["p9_no_aplica"] = True
                parciales.pop("p9", None)
                next_state = "esperando_p10"
            else:
                return BotReply(
                    text="Para evaluar imagen, enviame una foto real. Si prefieres omitirlo, escribe 'omitir'.",
                    content_type="text",
                )
        else:
            field_key = current_state.replace("esperando_", "")
            cleaned_value = value
            if intent in {"WHY", "SKIP"}:
                if current_state in {"esperando_correo", "esperando_comentario"}:
                    cleaned_value = None
                else:
                    return self._build_question_reply(current_state, prefix="Necesito tu respuesta para continuar.")

            if current_state == "esperando_correo":
                normalized_email = self._normalize_email(cleaned_value)
                if cleaned_value is not None and not normalized_email:
                    return BotReply(text="Ese correo no parece valido. Ejemplo: tunombre@gmail.com", content_type="text")
                cleaned_value = normalized_email

            if current_state.startswith("esperando_p") or current_state == "esperando_nps":
                min_v, max_v = (1, 10) if current_state == "esperando_nps" else (1, 5)
                parsed = self._parse_int(cleaned_value, min_v, max_v)
                if parsed is None:
                    return self._build_question_reply(current_state, prefix=f"La respuesta debe ser un numero entre {min_v} y {max_v}.")
                cleaned_value = str(parsed)

            if current_state == "esperando_autorizacion":
                if str(cleaned_value or "").lower() not in {"si", "no"}:
                    return self._build_question_reply(current_state)

            parciales[field_key] = cleaned_value
            idx = FORM_STATES_ORDER.index(current_state)
            next_state = FORM_STATES_ORDER[idx + 1] if idx + 1 < len(FORM_STATES_ORDER) else None

        if next_state is None:
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
        return self._build_question_reply(next_state)
