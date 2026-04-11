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
    FIELD_LABELS = {
        OnboardingStep.EDAD.value: "tu **edad**",
        OnboardingStep.PESO.value: "tu **peso**",
        OnboardingStep.ALTURA.value: "tu **talla (estatura)**",
        OnboardingStep.ALERGIAS.value: "si tienes alguna **alergia o restriccion**",
        OnboardingStep.TIPO_DIETA.value: "si sigues algun **tipo de dieta**",
        OnboardingStep.ENFERMEDADES.value: "si padeces alguna **condicion de salud**",
        OnboardingStep.RESTRICCIONES.value: "si tienes alguna **restriccion alimentaria**",
        OnboardingStep.OBJETIVO.value: "tu **objetivo nutricional**",
        OnboardingStep.PROVINCIA.value: "la **provincia** donde te encuentras",
        OnboardingStep.DISTRITO.value: "tu **distrito**",
    }

    SWITCHBOARD_SYSTEM_PROMPT = """Eres el Cerebro de Nutribot (Switchboard), encargado de clasificar la intención del usuario durante el registro de perfil.

REGLAS DE INTENCIÓN:
1. ANSWER: El usuario provee el dato solicitado (ej: '30 años', '80kg', 'no tengo alergias').
2. DOUBT: El usuario está confundido o hace una petición de chat/nutrición EN LUGAR de responder (ej: 'dame un menú', '¿para qué sirve?').
3. SKIP: El usuario pide saltar la pregunta.
4. GREETING: Saludo inicial.
5. RESET/STOP: Comandos de sistema.

REGLAS DE ORO (CRÍTICAS):
- PETICIONES DE COMIDA = DOUBT: Si el usuario pide un menú, receta o consejo (ej: 'Dame un menú marino', 'Dame dieta para bajar peso') MIENTRAS estás en un paso de perfil, clasifica SIEMPRE como DOUBT. NUNCA lo extraigas como 'Ninguna' o como dato de perfil.
- BLINDAJE DE ALERGIAS: Si el paso es ALERGIAS y el usuario pide comida, NUNCA devuelvas 'Ninguna'. Es preferible clasificar como DOUBT y pedir aclaración.

- COHERENCIA MÉDICA Y BIOLÓGICA (NUEVO):
  * RECHAZA datos absurdos (ej: 'alergia al aire', 'enfermedad de los marcianos').
  * RECHAZA métricas imposibles (ej: un adulto de 300cm, un bebé de 200kg, o una persona de 2 metros que pese 20kg).
  * Si el dato no tiene sentido biológico o médico, clasifica como DOUBT y pide aclaración en 'explanation'.

- Si tienes dudas entre ANSWER y DOUBT por una petición de comida o incoherencia, elige DOUBT.

EXAMPLES:
- Paso: PESO | Usuario: 'Peso 500 kilos' -> Intent: DOUBT, Data: {}, Explanation: '¡Wow! 😮 Quizás hubo un error al escribir. ¿Me confirmas tu peso real para calcular bien tu plan?'
- Paso: PESO | Usuario: '80kg y dame un menú' -> Intent: DOUBT, Data: {}, Explanation: '¡Perfecto! 📝 Ya casi llegamos al menú, solo confírmame primero el peso para que sea exacto.'
- Paso: PROVINCIA | Usuario: 'no deseo' -> Intent: SKIP, Data: {}, Explanation: 'Ningún problema, podemos seguir sin eso.'

REGLAS DE TONO Y FLEXIBILIDAD:
- EVITA FRASES BLOQUEANTES: Nunca digas "necesito esto para continuar" o "es obligatorio". Usa "Me ayudaría mucho a..." o "Para ser más preciso...".
- DETECCIÓN DE RECHAZO: Si el usuario dice "no quiero decirte", "ya me aburrí", "muchos datos", "no deseo", clasifica como SKIP.
- PETICIONES DE COMIDA DURANTE UBICACIÓN: Si estamos en PROVINCIA o DISTRITO y el usuario pide COMIDA/MENÚ, clasifica como SKIP para el campo actual para no bloquear.
- Si el usuario muestra FRUSTRACIÓN o ABURRIMIENTO, genera una 'explanation' muy breve y empática, y sugiere que pueden seguir con otra cosa.

FORMATO DE SALIDA (JSON):
{
  "intent": "ANSWER|DOUBT|SKIP|GREETING|RESET|STOP",
  "data": {"campo": "valor"} o {},
  "explanation": "Frase empática y breve (máx 12 palabras). Evita sonar repetitivo.",
  "confidence": 0.0-1.0
}
"""

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
        pre_extracted_data: Optional[dict] = None,
        history: Optional[list[dict]] = None
    ) -> Optional[str]:
        if state.onboarding_status not in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
            return None

        vl = user_text.lower().strip()
        current_step = state.onboarding_step

        # --- NEW Switchboard Logic (The Unified Brain) ---
        analysis = await self._analyze_turn(user_text, current_step, history)
        intent = analysis["intent"]
        
        if intent == "RESET" or user_text.strip() == "/reset":
            # Handle reset (clean up DB for this user) - Logic to be implemented or called
            await self._handle_system_reset(state.usuario_id, session)
            self._set_onboarding_state(state, OnboardingStatus.INVITED, OnboardingStep.INVITACION.value)
            return "¡Entendido! He borrado tus datos de perfil para que podamos empezar de cero cuando gustes. 🔄\n\n¿Quieres que empecemos ahora?"

        if current_step and current_step != OnboardingStep.INVITACION.value:
            # --- MODO OBSTINADO (Prioritarios) ---
            PRIORITARY_STEPS = [OnboardingStep.EDAD.value, OnboardingStep.PESO.value, OnboardingStep.ALTURA.value, OnboardingStep.ALERGIAS.value]
            is_food_request = any(w in vl for w in ["menu", "menú", "receta", "dieta", "comida", "desayuno", "almuerzo", "cena"])
            
            # REGLA DE ORO PARA UBICACIÓN (Provincia/Distrito): Cero insistencia si piden comida
            if current_step in (OnboardingStep.PROVINCIA.value, OnboardingStep.DISTRITO.value) and (intent == "DOUBT" or is_food_request):
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
                state.onboarding_step = await self._find_next_missing_step(session, state.usuario_id)
                if not state.onboarding_step:
                    state.onboarding_status = OnboardingStatus.COMPLETED.value
                return None # Devolvemos None para que el orchestrator responda la duda nutricional
            
            if current_step in PRIORITARY_STEPS and intent == "DOUBT":
                explanation = analysis.get("explanation")
                if explanation:
                    return f"{explanation}\n\n**{ONBOARDING_QUESTIONS.get(current_step, '')}**"

                prompts = {
                    OnboardingStep.EDAD.value: "tu **edad**",
                    OnboardingStep.PESO.value: "tu **peso**",
                    OnboardingStep.ALTURA.value: "tu **talla (estatura)**",
                    "alergias": "si tienes alguna **alergia o restricción**"
                }
                campo_lindo = prompts.get(current_step, self.FIELD_LABELS.get(current_step, current_step))
                return f"¡Me encantaría ayudarte con eso! 🍏 Pero para darte una recomendación que realmente te sirva y sea segura, primero necesito completar un detallito de tu perfil. 😊\n\n¿Me podrías decir {campo_lindo}, por favor?"
            # -------------------------------------

            if intent == "DOUBT":
                # Si tenemos una explicación específica de por qué falló la extracción o coherencia:
                explanation = analysis.get("explanation")
                
                # REGLA DE ORO PARA UBICACIÓN (Provincia/Distrito): Persuasión de un solo turno
                if current_step in (OnboardingStep.PROVINCIA.value, OnboardingStep.DISTRITO.value):
                    # Si muestra frustración o simplemente no quiere: SKIPEAMOS Y AVANZAMOS
                    if self._check_frustration(history, current_step) or any(w in vl for w in ["aburr", "harto", "no quiero", "no deseo", "basta", "dame", "nada"]):
                        await self._mark_field_as_skipped(session, state.usuario_id, current_step)
                        # Buscamos el siguiente campo pero NO devolvemos None aquí directamente
                        # Si es una duda nutricional, queremos que el orquestador responda.
                        # Si es simplemente un "no quiero", devolvemos el siguiente paso.
                        next_step = await self._find_next_missing_step(session, state.usuario_id)
                        if not next_step:
                            self._set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
                            return "¡Entendido! No te preocupes por eso. 😊 ¿En qué más puedo ayudarte hoy?"
                        
                        self._set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
                        if is_food_request or "dame" in vl:
                             return None # Deja que el orquestador responda el menú
                        return f"No hay problema, podemos saltarlo. 😊 Sigamos con otro detalle: **{ONBOARDING_QUESTIONS[next_step]}**"

                if explanation:
                    return f"{explanation}\n\n¿Seguimos con tu perfil? **{ONBOARDING_QUESTIONS.get(current_step, '')}**"

                # Si pide menu/receta en pleno onboarding, respondemos con control:
                # nunca negamos datos ya guardados y reenfocamos al paso faltante.
                if is_food_request:
                    res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
                    p = res_p.mappings().fetchone() or {}
                    known_parts = []
                    if p.get("edad"):
                        known_parts.append(f"Edad: {p['edad']} anos")
                    if p.get("peso_kg"):
                        known_parts.append(f"Peso: {p['peso_kg']}kg")
                    if p.get("altura_cm"):
                        h = float(p["altura_cm"])
                        h_str = f"{h/100:.2f}m" if h > 10 else f"{h:.2f}m"
                        known_parts.append(f"Talla: {h_str}")
                    if p.get("alergias"):
                        known_parts.append(f"Alergias: {p['alergias']}")

                    known_line = f"Tengo registrado: {', '.join(known_parts)}. " if known_parts else ""
                    campo_lindo = self.FIELD_LABELS.get(current_step, f"tu **{current_step}**")
                    question = ONBOARDING_QUESTIONS.get(current_step, "")
                    return (
                        f"Vamos bien. {known_line}Para darte una recomendacion 100% personalizada, "
                        f"solo me falta confirmar {campo_lindo}. {question}"
                    ).strip()

                # Let the general LLM handle the explanation for continuity
                return None
            
            if intent == "SKIP":
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
                # Fall through to advance step logic below
            
            if intent == "GREETING":
                # Let general LLM greet, we will append anchor later
                return None

            if intent == "STOP":
                self._set_onboarding_state(state, OnboardingStatus.PAUSED, current_step)
                return "De acuerdo, pausamos aquí. Si quieres seguir más tarde, solo dime 'continuar'. 👋"
            
            # -----------------------------------------------

        if current_step == OnboardingStep.INVITACION.value:
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
                if p.get("altura_cm"):
                    h = float(p["altura_cm"])
                    h_str = f"{h/100:.2f}m" if h > 10 else f"{h:.2f}m"
                    known_parts.append(f"Talla: {h_str}")
                
                if known_parts:
                    return f"¡Genial! 😊 Ya tengo registrado: **{', '.join(known_parts)}**. Ahora necesito completar unos datos más.\n\n{ONBOARDING_QUESTIONS[next_step]}"
                return ONBOARDING_QUESTIONS[next_step]

        current_step = state.onboarding_step
        if not current_step:
            return None

        # Process extracted data from Switchboard if intent was ANSWER
        if intent == "ANSWER" and analysis.get("data"):
            extracted = await self._profile_extractor.apply_cleaning_and_save(
                raw_extractions=analysis["data"],
                user_text=user_text,
                usuario_id=state.usuario_id,
                session=session,
                current_step=current_step
            )
        elif intent == "SKIP":
            extracted = {}
        else:
            extracted = {}

        if not extracted:
            if intent == "ANSWER":
                # Check frustration only when ANSWER fails
                if self._check_frustration(history, current_step):
                    campo_lindo = self.FIELD_LABELS.get(current_step, current_step)
                    return f"Veo que este punto es algo confuso. 😅 Si prefieres, podemos **saltarlo** por ahora y seguir con lo demás para no estancarnos. ¿Te parece?\n\nO si gustas, dime tu **{campo_lindo}** para continuar."

                return f"No logré captar ese detalle para tu perfil. 😅 ¿Podrías decírmelo de forma más simple?\n\n**{ONBOARDING_QUESTIONS.get(current_step, '')}**"
            elif intent == "SKIP":
                await self._mark_field_as_skipped(session, state.usuario_id, current_step)
            else:
                # GREETING and DOUBT are already handled at the top of advance_flow
                pass

        if not extracted and intent == "SKIP":
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
                
                # PD Recordatorio para campos opcionales
                pd_rem = ""
                if next_step in [OnboardingStep.ENFERMEDADES.value, OnboardingStep.RESTRICCIONES.value, OnboardingStep.OBJETIVO.value]:
                    pd_rem = "\n\n💡 *Recuerda:* Si en algún momento quieres contarme sobre tus alergias, restricciones o alguna condición de salud, solo dímelo. ¡Me ayuda a ser 100% exacto! 😊"

                return f"{transition} Ahora, para seguir personalizando tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}{pd_rem}"
            
            # Si no hubo extracción pero llegamos aquí, es que estamos repitiendo el paso actual
            if next_step == current_step:
                return f"Para poder ayudarte mejor, necesito este dato: **{ONBOARDING_QUESTIONS[current_step]}**"

            return f"Siguiendo con tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
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

    async def _analyze_turn(self, user_text: str, current_step: str, history: Optional[list[dict]]) -> dict:
        """
        The Switchboard: One LLM call to rule them all.
        Classifies intent and extracts raw data.
        """
        history_summary = ""
        if history:
            history_summary = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-5:]])

        current_q = ONBOARDING_QUESTIONS.get(current_step, "desconocida")
        
        prompt = f"""PASO ACTUAL: {current_step}
PREGUNTA: "{current_q}"

HISTORIAL:
{history_summary}

MENSAJE USUARIO: "{user_text}"
"""
        try:
            resp = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": self.SWITCHBOARD_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            import json
            analysis = json.loads(resp.choices[0].message.content)
            logger.info("Onboarding Switchboard: user=%s, intent=%s, data=%s", current_step, analysis.get("intent"), analysis.get("data"))
            return analysis
        except Exception as e:
            logger.error("Error in Switchboard: %s", e)
            return {"intent": "ANSWER", "data": {}, "explanation": None}

    async def _handle_system_reset(self, uid: int, session: AsyncSession):
        """Clean up user data and onboarding progress."""
        logger.info("System Reset triggered for user %s", uid)
        await session.execute(text("DELETE FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM memoria_chat WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM formulario_en_progreso WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM profile_extractions WHERE usuario_id = :uid"), {"uid": uid})
        await session.execute(text("DELETE FROM extraction_jobs WHERE usuario_id = :uid"), {"uid": uid})
        # Note: ConversationState is managed in the caller (advance_flow).
        # No commit here: this method runs inside an active transaction.

    def _check_frustration(self, history: Optional[list[dict]], current_step: str) -> bool:
        """
        Detecta si el usuario está estancado o mostrando signos de molestia.
        """
        if not history or len(history) < 2:
            return False
            
        last_user_msg = history[-1]["content"].lower() if history[-1]["role"] == "user" else ""
        frustration_keywords = ["aburr", "harto", "no quiero", "no deseo", "basta", "dame lo que", "muchos datos", "pesado", "stuck", "que fue", "no contestas"]
        if any(kw in last_user_msg for kw in frustration_keywords):
            return True

        # Regla de repetición (Assistant preguntó lo mismo 2 veces)
        assistant_msgs = [m["content"] for m in history if m["role"] == "assistant"]
        if len(assistant_msgs) >= 2:
            last_q = assistant_msgs[-1].lower()
            prev_q = assistant_msgs[-2].lower()
            q_text = ONBOARDING_QUESTIONS.get(current_step, "").lower()
            if q_text and q_text in last_q and q_text in prev_q:
                return True
            
        return False
