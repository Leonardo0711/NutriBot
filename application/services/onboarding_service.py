"""
Nutribot Backend — OnboardingService
Gestiona la secuencia inicial opt-in para recolectar el perfil del usuario utilizando OOP.
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities import ConversationState
from domain.value_objects import OnboardingStatus, OnboardingStep, ONBOARDING_STEPS_ORDER
from domain.utils import get_now_peru
from domain.parsers import parse_weight, parse_height
from application.services.profile_extraction_service import ProfileExtractionService

logger = logging.getLogger(__name__)

ONBOARDING_QUESTIONS: dict[str, str] = {
    OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂",
    OnboardingStep.PESO.value: "¿Y cuál es tu peso aproximado en kilos? ⚖️",
    OnboardingStep.ALTURA.value: "¿Y cuál es tu estatura aproximada? Puedes usar centímetros o metros (Ej. 1.70m, 170cm) 📐",
    OnboardingStep.TIPO_DIETA.value: "¿Sigues algún **tipo de dieta** especial? 🥗 (Ej: *Omnívora* si comes de todo, *Vegetariana*, *Vegana*, *Keto*, *Sin gluten*, o ninguna en particular)",
    OnboardingStep.ALERGIAS.value: "¿Tienes alguna **alergia** o intolerancia alimentaria? (Ej. Maní, mariscos, lactosa...) 🍎",
    OnboardingStep.ENFERMEDADES.value: "¿Padeces alguna condición de salud relevante? 🏥 (Ej: *Diabetes*, *Hipertensión*, *Hipotiroidismo* o ninguna)",
    OnboardingStep.RESTRICCIONES.value: "¿Tienes alguna **restricción alimentaria** por religión, ética o gusto personal? 🚫 (Ej: No como cerdo, no como carnes rojas, no me gusta el brócoli...)",
    OnboardingStep.OBJETIVO.value: "¿Cuál es tu **objetivo principal** al usar Nutribot? 🎯 (Ej. *Bajar de peso*, *Ganar masa muscular*, *Controlar mi glucemia*, o simplemente *Comer más sano*)",
    OnboardingStep.PROVINCIA.value: "¿En qué **provincia** de Perú te encuentras actualmente? 😊",
    OnboardingStep.DISTRITO.value: "¿Y en qué **distrito** vives específicamente? 🏠 (Para darte recomendaciones locales)"

}

class OnboardingService:
    def __init__(self, openai_client: AsyncOpenAI, openai_model: str, profile_extractor: ProfileExtractionService):
        self._openai_client = openai_client
        self._openai_model = openai_model
        self._profile_extractor = profile_extractor

    def _validate_onboarding_field(self, step: str, raw_value: str) -> tuple[bool, Optional[str], Optional[str]]:
        v = raw_value.strip()
        vl = v.lower()
        skip_words = ["no", "saltar", "ninguno", "ninguna", "paso", "no se", "omitir", "despues"]
        is_skip = any(vl == w or vl.startswith(w + " ") for w in skip_words)

        if step == OnboardingStep.EDAD.value:
            try:
                age = int(re.sub(r"\D", "", v))
                if 5 <= age <= 120:
                    return True, str(age), None
            except ValueError:
                pass
            return False, None, "¿Podrías darme tu edad en números enteros? (Ej. 30)"
        
        elif step == OnboardingStep.PESO.value:
            if is_skip: return True, None, None
            w = parse_weight(v)
            if w:
                return True, str(w), None
            return False, None, "No logré captar el peso. ¿Podrías decirlo en kilos o libras? (O escribe 'saltar' si prefieres no decirlo aún)"
        
        elif step == OnboardingStep.ALTURA.value:
            if is_skip: return True, None, None
            h = parse_height(v)
            if h:
                return True, str(h), None
            return False, None, "No logré captar la estatura. ¿Podrías decirlo en centímetros o metros? (O escribe 'saltar' si prefieres no decirlo aún)"
        else:
            if len(v) > 200:
                return False, None, "¡Uy, es un poco largo! ¿Podrías resumirlo un poquito más, por favor?"
            return True, v, None

    def _set_onboarding_state(self, state: ConversationState, status: OnboardingStatus, step: Optional[str], **kwargs):
        state.onboarding_status = status.value
        state.onboarding_step = step
        state.onboarding_updated_at = get_now_peru()
        
        if status == OnboardingStatus.INVITED:
            state.onboarding_last_invited_at = get_now_peru()
        
        if status == OnboardingStatus.SKIPPED:
            state.onboarding_next_eligible_at = get_now_peru() + timedelta(days=14)
            if "skip_count" in kwargs:
                state.onboarding_skip_count = kwargs["skip_count"]
        
        if status == OnboardingStatus.PAUSED:
            state.onboarding_next_eligible_at = get_now_peru() + timedelta(days=3)
        
        logger.info(
            "Onboarding state change: user=%s, status=%s → %s, step=%s",
            state.usuario_id, state.onboarding_status, status.value, step
        )

    async def advance_flow(
        self,
        user_text: str,
        state: ConversationState,
        session: AsyncSession,
        treat_ninguna_as_missing: bool = False,
        pre_extracted_data: Optional[dict] = None
    ) -> Optional[str]:
        if state.onboarding_status not in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            return None

        vl = user_text.lower().strip()
        
        if state.onboarding_step == OnboardingStep.INVITACION.value:
            prompt = f"""Analiza la respuesta del usuario a una invitación de 'Personalizar perfil nutricional'.
            Responde SOLO con una palabra: ACCEPT, REJECT o OTHER.
            USUARIO: "{user_text}"
            """
            resp = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "system", "content": "Eres un clasificador de intenciones binarias."},
                          {"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            intent = resp.choices[0].message.content.strip().upper()
            
            if "REJECT" in intent:
                # If they say NO but added more text (length > 15), treat as OTHER (pause)
                if len(user_text) > 15:
                    self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
                    return None
                
                self._set_onboarding_state(state, OnboardingStatus.SKIPPED, None, skip_count=state.onboarding_skip_count + 1)
                return "¡Entendido! Seguimos conversando libremente. Si alguna vez quieres personalizar tu perfil, solo dímelo. 😊 ¿En qué más puedo ayudarte hoy?"
            elif "OTHER" in intent:
                self._set_onboarding_state(state, OnboardingStatus.PAUSED, None)
                return None
            else:
                next_step = await self._find_next_missing_step(session, state.usuario_id)
                if next_step is None:
                    self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
                    return "¡Veo que ya tengo tu perfil nutricional completo! 😊 ¿En qué puedo ayudarte hoy?"

                self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
                res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
                p = res_p.mappings().fetchone() or {}
                known_parts = []
                if p.get("edad"): known_parts.append(f"Edad: {p['edad']} años")
                if p.get("peso_kg"): known_parts.append(f"Peso: {p['peso_kg']}kg")
                if p.get("altura_cm"): known_parts.append(f"Talla: {p['altura_cm']}cm")
                
                if known_parts:
                    return f"¡Genial! 😊 Ya tengo registrado: **{', '.join(known_parts)}**. Ahora necesito completar unos datos más.\n\n{ONBOARDING_QUESTIONS[next_step]}"
                return ONBOARDING_QUESTIONS[next_step]

        current_step = state.onboarding_step
        if not current_step:
            return None

        # Use pre_extracted_data if available, otherwise call LLM
        if pre_extracted_data is not None:
            extracted = pre_extracted_data
            logger.info("OnboardingService: Using pre-extracted data: %s", extracted)
        else:
            # Delegate extraction to the ProfileExtractionService
            extracted = await self._profile_extractor.extract_and_save(
                user_text=user_text,
                usuario_id=state.usuario_id,
                session=session,
                current_step=current_step
            )

        
        is_interruption = False

        if not extracted:
            skip_words = ["saltar", "paso", "omitir", "no", "ninguna", "ninguno", "siguiente"]
            is_skip = any(w in vl.split() or vl.startswith(w) for w in skip_words)
            if not is_skip:
                interruption_prompt = f"""Analiza si el usuario intentó responder a la pregunta '{ONBOARDING_QUESTIONS.get(current_step, '')}' pero dio una respuesta vaga, o si definitivamente cambió de tema.
                Responde SOLO: 'ANSWER' si intentaba responder.
                Responde SOLO: 'INTERRUPTION' si cambió de tema claramente.
                USUARIO: "{user_text}"
                """
                int_resp = await self._openai_client.chat.completions.create(
                    model=self._openai_model,
                    messages=[{"role": "system", "content": "Eres un clasificador de intención exacto."},
                              {"role": "user", "content": interruption_prompt}],
                    max_tokens=5,
                    temperature=0
                )
                is_interruption = "INTERRUPTION" in int_resp.choices[0].message.content.strip().upper()

        if is_interruption:
            self._set_onboarding_state(state, OnboardingStatus.PAUSED, state.onboarding_step)
            return None

        if not extracted:
            # Silence logic: only avoid the error message if it's a clear short negation/skip
            silent_negations = ["no", "nada", "no tengo", "ninguno", "ninguna", "sin", "no se", "no sé", "no se nada", "ningunaa", "ningunaaa", "ningun", "inguna", "ingun", "nada que ver", "nada que noo", "naa", "nadaa", "naaa", "noo", "nooo", "nada de eso"]
            is_silent_neg = any(s in vl for s in silent_negations) or len(vl) < 4
            is_explicit_skip = any(w in vl.split() or vl.startswith(w) for w in ["saltar", "paso", "omitir", "luego", "siguiente"])
            
            if not is_silent_neg and not is_explicit_skip:
                # Intelligence: explain WHY it wasn't captured if it's a long text
                if len(user_text) > 10:
                    explain_prompt = f"""El usuario intentó responder a la pregunta '{ONBOARDING_QUESTIONS.get(current_step, '')}' con el texto: '{user_text}'.
                    Nuestra IA de extracción NO captó ningún dato válido. 
                    Explica de forma muy breve y amable (máximo 15 palabras) por qué ese dato no parece válido o por qué no se puede registrar.
                    Ejemplo para 'Diabetes tipo 20': 'No reconozco ese tipo de diabetes, ¿podrías confirmarlo?'
                    Ejemplo para algo incoherente: 'No logré entender ese detalle, ¿puedes repetirlo más simple?'
                    """
                    exp_resp = await self._openai_client.chat.completions.create(
                        model=self._openai_model,
                        messages=[{"role": "user", "content": explain_prompt}],
                        max_tokens=40,
                        temperature=0.7
                    )
                    explanation = exp_resp.choices[0].message.content.strip()
                    return f"{explanation} 😊"
                
                return f"No logré captar ese dato para tu perfil. ¿Podrías decírmelo de forma más clara? 😊"


        if not extracted and ("saltar" in vl or "paso" in vl or "omitir" in vl or "luego" in vl or "siguiente" in vl):
            await self._mark_field_as_skipped(session, state.usuario_id, current_step)

        updated_cols = list(extracted.keys()) if extracted else []
        current_idx = -1
        for i, s in enumerate(ONBOARDING_STEPS_ORDER):
            if s.value == current_step:
                current_idx = i
                break

        was_current_answered = any(col in updated_cols for col in [current_step, "peso_kg" if current_step=="peso" else current_step])
        was_current_skipped = not extracted and any(w in vl.split() or vl.startswith(w) for w in ["saltar", "paso", "omitir", "luego", "siguiente"])
        
        # If the current step was NOT answered/skipped (i.e., it was a correction of a past field),
        # we should start looking for the next missing step starting FROM the current step.
        search_start_idx = current_idx + 1 if (was_current_answered or was_current_skipped) else current_idx

        next_step = await self._find_next_missing_step(
            session, 
            state.usuario_id, 
            treat_ninguna_as_missing=treat_ninguna_as_missing, 
            start_from_idx=search_start_idx,
            ignore_cols=updated_cols
        )


        if next_step:
            self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
            if updated_cols:
                transition = "¡Perfecto! Ya anoté esos detalles. ✍️"
                if len(updated_cols) == 1 and updated_cols[0] == "region":
                    transition = "¡Qué bueno! Me encanta esa zona. 📍"
                return f"{transition} Ahora, para seguir personalizando tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
            return f"Entendido. 😊 Siguiendo con tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
        else:
            res_final = await session.execute(text("SELECT provincia, distrito FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
            p_final = res_final.fetchone()
            if p_final and (not p_final.provincia or not p_final.distrito):
                force_step = OnboardingStep.PROVINCIA.value if not p_final.provincia else OnboardingStep.DISTRITO.value
                self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, force_step)
                return f"¡Casi terminamos! 🎯 Solo un pequeño detalle final: {ONBOARDING_QUESTIONS[force_step].lower()}"

            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
            return "¡Excelente, perfil completado! 🎯 Muchas gracias por tu tiempo. Esto me ayudará a darte recomendaciones mucho más precisas. ¿En qué más puedo ayudarte hoy?"

    async def _find_next_missing_step(self, session: AsyncSession, uid: int, ignore_skips: bool = False, treat_ninguna_as_missing: bool = False, skip_step: Optional[str] = None, start_from_idx: Optional[int] = None, ignore_cols: Optional[list[str]] = None) -> Optional[str]:
        res = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        p = res.mappings().fetchone()
        
        if not p:
            return OnboardingStep.EDAD.value

        skipped = p.get("skipped_fields", {})
        if not isinstance(skipped, dict):
            skipped = {}

        col_map = {
            OnboardingStep.EDAD.value: "edad",
            OnboardingStep.ALERGIAS.value: "alergias",
            OnboardingStep.ENFERMEDADES.value: "enfermedades",
            OnboardingStep.RESTRICCIONES.value: "restricciones_alimentarias",
            OnboardingStep.TIPO_DIETA.value: "tipo_dieta",
            OnboardingStep.OBJETIVO.value: "objetivo_nutricional",
            OnboardingStep.PESO.value: "peso_kg",
            OnboardingStep.ALTURA.value: "altura_cm",
            OnboardingStep.REGION.value: "region",
            OnboardingStep.PROVINCIA.value: "provincia",
            OnboardingStep.DISTRITO.value: "distrito"
        }

        base_idx = start_from_idx if start_from_idx is not None else 1
        for step in ONBOARDING_STEPS_ORDER[base_idx:]:
            if skip_step and step.value == skip_step:
                continue
                
            col = col_map.get(step.value)
            if not col: continue

            if ignore_cols and col in ignore_cols:
                continue
            
            if not ignore_skips and skipped.get(step.value):
                continue

            val = p.get(col)
            is_empty = val is None or (isinstance(val, str) and len(val.strip()) == 0)
            
            if treat_ninguna_as_missing and isinstance(val, str) and val.upper() == "NINGUNA":
                is_empty = True

            if is_empty:
                return step.value

        return None

    async def _mark_field_as_skipped(self, session: AsyncSession, uid: int, field: str):
        sql_init = """
            INSERT INTO perfil_nutricional (usuario_id, actualizado_en)
            VALUES (:uid, :upd)
            ON CONFLICT (usuario_id) DO NOTHING
        """
        await session.execute(text(sql_init), {"uid": uid, "upd": get_now_peru()})

        sql_skip = f"""
            UPDATE perfil_nutricional 
            SET skipped_fields = skipped_fields || jsonb_build_object(:field, true),
                actualizado_en = :upd
            WHERE usuario_id = :uid
        """
        await session.execute(text(sql_skip), {"uid": uid, "field": field, "upd": get_now_peru()})
