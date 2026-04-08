"""
Nutribot Backend — AdvanceOnboardingFlowUseCase
Gestiona la secuencia inicial opt-in para recolectar el perfil del usuario.

NOTA: Esta función SOLO modifica el objeto `state` en memoria.
La persistencia la hace `save_state()` en handle_incoming_message.py
para evitar conflictos de doble escritura.
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

logger = logging.getLogger(__name__)

ONBOARDING_QUESTIONS: dict[str, str] = {
    OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂",
    OnboardingStep.PESO.value: "¿Y cuál es tu peso aproximado en kilos? ⚖️",
    OnboardingStep.ALTURA.value: "¿Y cuál es tu estatura aproximada? Puedes usar centímetros o metros (Ej. 1.70m, 170cm) 📐",
    OnboardingStep.TIPO_DIETA.value: "¿Sigues algún tipo de dieta especial? (Ej: Vegana, Keto, Sin gluten, o ninguna) 🥗",
    OnboardingStep.ALERGIAS.value: "¿Tienes alguna alergia o intolerancia alimentaria? 🍎",
    OnboardingStep.ENFERMEDADES.value: "¿Padeces alguna condición de salud relevante? (Ej: Diabetes, Hipertensión) 🏥",
    OnboardingStep.OBJETIVO.value: "¿Cuál es tu principal objetivo nutricional? (Ej. comer más sano, bajar de peso, subir masa muscular, controlar mi azúcar...) 🎯",
    OnboardingStep.REGION.value: "Por último, ¿en qué región y provincia de Perú te encuentras? (Esto es para avisarte de campañas de salud cerca de ti) 📍"
}

def _parse_weight(val: str) -> Optional[float]:
    v = val.lower().strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilo|lb|libra)?", v)
    if not match: return None
    num = float(match.group(1))
    unit = match.group(2)
    if unit and ("lb" in unit or "lib" in unit):
        return round(num * 0.453592, 2)
    return round(num, 2)

def _parse_height(val: str) -> Optional[float]:
    v = val.lower().strip()
    
    # Check for ft/in like 5'4" first
    ft_in_match = re.search(r"(\d+)\s*'\s*(\d+)\s*\"", v)
    if ft_in_match:
        ft = int(ft_in_match.group(1))
        inches = int(ft_in_match.group(2))
        return round((ft * 30.48) + (inches * 2.54), 2)
    
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(cm|m|metro)?", v)
    if not match: return None
    num_str = match.group(1).replace(",", ".")
    num = float(num_str)
    unit = match.group(2)
    
    if not unit:
        # Guesses based on value
        if num < 3.0: return round(num * 100, 2)
        else: return round(num, 2)
    if "m" in unit and "cm" not in unit:
        return round(num * 100, 2)
    return round(num, 2)


def _validate_onboarding_field(step: str, raw_value: str) -> tuple[bool, Optional[str], Optional[str]]:
    v = raw_value.strip()
    vl = v.lower()
    
    skip_words = ["no", "saltar", "ninguno", "ninguna", "paso", "no se", "omitir", "despues"]
    is_skip = any(vl == w or vl.startswith(w + " ") for w in skip_words)

    if step == OnboardingStep.EDAD:
        try:
            age = int(re.sub(r"\D", "", v))
            if 5 <= age <= 120:
                return True, str(age), None
        except ValueError:
            pass
        return False, None, "¿Podrías darme tu edad en números enteros? (Ej. 30)"
    
    elif step == OnboardingStep.PESO:
        if is_skip: return True, None, None
        w = _parse_weight(v)
        if w and 10 <= w <= 300:
            return True, str(w), None
        return False, None, "No logré captar el peso. ¿Podrías decirlo en kilos o libras? (O escribe 'saltar' si prefieres no decirlo aún)"
    
    elif step == OnboardingStep.ALTURA:
        if is_skip: return True, None, None
        h = _parse_height(v)
        if h and 40 <= h <= 250:
            return True, str(h), None
        return False, None, "No logré captar la estatura. ¿Podrías decirlo en centímetros o metros? (O escribe 'saltar' si prefieres no decirlo aún)"

    else:
        # Texto libre
        if len(v) > 200:
            return False, None, "¡Uy, es un poco largo! ¿Podrías resumirlo un poquito más, por favor?"
        return True, v, None


def _set_onboarding_state(state: ConversationState, status: OnboardingStatus, step: Optional[str], **kwargs):
    """
    Modifica el objeto state EN MEMORIA.
    save_state() en handle_incoming_message.py se encarga de persistir.
    NO hace SQL directo — eso causaba conflictos de doble escritura.
    """
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


async def advance_onboarding_flow(
    user_text: str,
    state: ConversationState,
    session: AsyncSession,
    openai_client: AsyncOpenAI,
    openai_model: str,
    current_profile: Optional[dict] = None
) -> Optional[str]:
    """
    Controla la máquina de estados del onboarding opcional usando LLM para extracción robusta.
    Retorna el string que el bot debe decir, o None si el usuario abandonó el flujo.
    
    IMPORTANTE: Esta función SOLO modifica el objeto `state` — no hace SQL directo
    sobre conversation_state. El caller (handle_incoming_message) persiste todo
    con save_state() al final de la transacción.
    """
    if state.onboarding_status not in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
        logger.debug("advance_onboarding: skipping, status=%s", state.onboarding_status)
        return None

    # Normalizamos input
    vl = user_text.lower().strip()
    
    logger.info(
        "advance_onboarding: user=%s, status=%s, step=%s, text='%s'",
        state.usuario_id, state.onboarding_status, state.onboarding_step, user_text[:50]
    )
    
    # 1. Evaluar si está respondiendo a la Invitación Inicial (SEMÁNTICO)
    if state.onboarding_step == OnboardingStep.INVITACION.value:
        # Clasificador ultrarrápido de intención
        prompt = f"""Analiza la respuesta del usuario a una invitación de 'Personalizar perfil nutricional'.
        Responde SOLO con una palabra: ACCEPT, REJECT o OTHER.
        
        USUARIO: "{user_text}"
        
        Guía:
        - ACCEPT: si o variaciones (zi, chi, ya, dale, ok, bueno, por supu, claro, me gustaria, etc)
        - REJECT: no o variaciones (no quiero, paso, luego, no x ahora, naca, saltar, etc)
        - OTHER: si pregunta algo distinto o cambia de tema drásticamente."""
        
        resp = await openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "system", "content": "Eres un clasificador de intenciones binarias."},
                      {"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        intent = resp.choices[0].message.content.strip().upper()
        logger.info("advance_onboarding: invitation intent='%s' for text='%s'", intent, user_text[:30])

        if "REJECT" in intent:
            _set_onboarding_state(state, OnboardingStatus.SKIPPED, None, skip_count=state.onboarding_skip_count + 1)
            return "¡Entendido! Seguimos conversando libremente. Si alguna vez quieres personalizar tu perfil, solo dímelo. 😊 ¿En qué más puedo ayudarte hoy?"

        elif "OTHER" in intent and len(user_text) > 15:
            _set_onboarding_state(state, OnboardingStatus.PAUSED, None)
            return None

        else:
            # Asumimos ACCEPT si no es rechazo claro
            # Buscamos el primer paso realmente vacío (Siempre fresco)
            next_step = await _find_next_missing_step(session, state.usuario_id)
            logger.info("advance_onboarding: ACCEPT, next_missing_step=%s", next_step)
            
            if next_step is None: # Ya todo está lleno
                _set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
                return "¡Veo que ya tengo tu perfil nutricional completo! 😊 ¿En qué puedo ayudarte hoy?"

            _set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
            
            # Si ya hay datos previos, mencionarlos
            res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
            p = res_p.mappings().fetchone() or {}
            known_parts = []
            if p.get("edad"): known_parts.append(f"Edad: {p['edad']} años")
            if p.get("peso_kg"): known_parts.append(f"Peso: {p['peso_kg']}kg")
            if p.get("altura_cm"): known_parts.append(f"Talla: {p['altura_cm']}cm")
            
            if known_parts:
                return f"¡Genial! 😊 Ya tengo registrado: **{', '.join(known_parts)}**. Ahora necesito completar unos datos más.\n\n{ONBOARDING_QUESTIONS[next_step]}"
            
            return ONBOARDING_QUESTIONS[next_step]

    # Primero: ¿El usuario me está respondiendo o cambió de tema?
    # Agregamos bypass para mensajes que parecen datos (números + unidades)
    is_data_pattern = any(re.search(p, vl) for p in [r"\d+\s*kg", r"\d+\s*cm", r"\d+[\.,]\d+", r"\d+\s*años", r"^\d+$"])
    
    if is_data_pattern:
        is_interruption = False
        logger.debug("advance_onboarding: data pattern detected, skipping interruption check")
    else:
        interruption_prompt = f"""Analiza si el usuario está respondiendo a la pregunta '{ONBOARDING_QUESTIONS.get(state.onboarding_step, '')}' o si ha cambiado de tema para preguntar otra cosa.
        Si el usuario está dando un dato (números, medidas, etc.), responde siempre 'ANSWER'.
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

    if is_interruption and not any(w in vl for w in ["saltar", "paso", "no"]):
        # Abandono natural a mitad del formulario: pausamos
        logger.info("advance_onboarding: interruption detected for user=%s, pausing", state.usuario_id)
        _set_onboarding_state(state, OnboardingStatus.PAUSED, state.onboarding_step)
        return None # Devuelve control al Chat libre

    current_step = state.onboarding_step
    if not current_step:
        logger.warning("advance_onboarding: no current_step for user=%s, returning None", state.usuario_id)
        return None

    # Extracción robusta para campos numéricos
    cleaned_value = None
    err_msg = None
    
    if current_step in [OnboardingStep.EDAD.value, OnboardingStep.PESO.value, OnboardingStep.ALTURA.value]:
        # 1. Intento de extracción rápida por Regex (para casos como "90}")
        if current_step == OnboardingStep.EDAD.value:
            m = re.search(r"(\d+)", vl)
            if m: cleaned_value = m.group(1)
        elif current_step == OnboardingStep.PESO.value:
            w = _parse_weight(vl)
            if w: cleaned_value = str(w)
        elif current_step == OnboardingStep.ALTURA.value:
            h = _parse_height(vl)
            if h: cleaned_value = str(h)

        # 2. Si falló el regex, usar LLM para lenguaje natural ("pesos setenta kg", etc.)
        if cleaned_value is None:
            extractions = {
                OnboardingStep.EDAD.value: "Extract age as integer. Ignore typos, punctuation or extra characters (e.g. '90}' -> 90). Return only number or 'NONE'.",
                OnboardingStep.PESO.value: "Extract weight. Target kg. Handle 'quilos', 'kilos', 'lib', 'lbs'. If lbs, convert to kg (1 lb = 0.45kg). Ignore noise like braces or typos. Return only number or 'NONE'.",
                OnboardingStep.ALTURA.value: "Extract height. Target cm. Handle 'mts', 'metros', 'cms', 'centimetros', 'ft', 'in'. Ignore noise. Return only number or 'NONE'."
            }
            
            extract_resp = await openai_client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "system", "content": extractions[current_step]},
                          {"role": "user", "content": user_text}],
                max_tokens=10,
                temperature=0
            )
            val_str = extract_resp.choices[0].message.content.strip()
            logger.info("advance_onboarding: LLM extraction for %s: '%s' → '%s'", current_step, user_text[:30], val_str)
            
            if val_str.upper() != "NONE":
                try:
                    # Validamos rangos básicos
                    f_val = float(re.sub(r"[^\d\.]", "", val_str))
                    if current_step == OnboardingStep.EDAD.value and 5 <= f_val <= 120:
                        cleaned_value = str(int(f_val))
                    elif current_step == OnboardingStep.PESO.value and 20 <= f_val <= 400:
                        cleaned_value = str(round(f_val, 2))
                    elif current_step == OnboardingStep.ALTURA.value and 40 <= f_val <= 250:
                        cleaned_value = str(round(f_val, 2))
                except Exception:
                    logger.warning("advance_onboarding: failed to parse LLM value '%s'", val_str)

        if cleaned_value is None:
            # Solo si no es opcional o no dijo "saltar"
            if "saltar" in vl or "paso" in vl or "omitir" in vl or "no" in vl:
                cleaned_value = None # Se guarda NULL
            else:
                return f"No logré captar ese dato (intenté extraer {current_step}). ¿Podrías decírmelo de forma más clara? 😊"

    else:
        # Texto libre (Alergias, etc.) - ESTANDARIZACIÓN CON LLM
        standard_prompt = f"""Estandariza la respuesta del usuario para el campo '{current_step}'.
        Reglas:
        1. Si el usuario dice que no tiene nada, responde 'NINGUNA'.
        2. Si menciona una condición, responde SOLO el nombre en MAYÚSCULAS (ej: DIABETES, INTOLERANTE A LA LACTOSA).
        3. Si menciona varias, sepáralas por comas.
        4. No pongas frases como 'El usuario tiene...' ni explicaciones. Solo el valor canónico.
        
        USUARIO: "{user_text}"
        """
        std_resp = await openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "system", "content": "Analista de datos médicos/nutricionales experto en estandarización."},
                      {"role": "user", "content": standard_prompt}],
            max_tokens=40,
            temperature=0
        )
        cleaned_value = std_resp.choices[0].message.content.strip().upper()
        if any(w in cleaned_value for w in ["NINGUNA", "NO TENGO", "NO TENGO NINGUNA", "NO TENGO NADA"]):
            cleaned_value = "NINGUNA"

    # Guardar
    # Guardar
    if cleaned_value is not None:
        await _save_profile_field(session, state.usuario_id, current_step, cleaned_value)
        logger.info("advance_onboarding: saved %s=%s for user=%s", current_step, cleaned_value, state.usuario_id)
    elif "saltar" in vl or "paso" in vl or "omitir" in vl or "luego" in vl or "siguiente" in vl:
        # Omisión explícita
        await _mark_field_as_skipped(session, state.usuario_id, current_step)
        logger.info("advance_onboarding: marked %s as skipped for user=%s", current_step, state.usuario_id)
    elif "no" in vl and len(vl) < 10:
        # Un "no" corto suele ser omisión o "no tengo" dependiendo del campo
        if current_step in [OnboardingStep.ALERGIAS.value, OnboardingStep.ENFERMEDADES.value, OnboardingStep.TIPO_DIETA.value]:
            # En estos campos "no" suele significar "ninguna"
            await _save_profile_field(session, state.usuario_id, current_step, "NINGUNA")
        else:
            # En edad/peso/talla un "no" es omisión
            await _mark_field_as_skipped(session, state.usuario_id, current_step)

    # Avanzar al siguiente paso REALMENTE vacío (FORZAMOS RE-CONSULTA)
    next_step = await _find_next_missing_step(session, state.usuario_id)
    logger.info("advance_onboarding: after save, next_missing_step=%s for user=%s", next_step, state.usuario_id)

    if next_step:
        _set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
        
        # Obtener resumen de lo que ya sabemos para confirmar
        res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
        p = res_p.mappings().fetchone() or {}
        
        summary_parts = []
        if p.get("edad"): summary_parts.append(f"Edad: {p['edad']}")
        if p.get("peso_kg"): summary_parts.append(f"Peso: {p['peso_kg']}kg")
        if p.get("altura_cm"): summary_parts.append(f"Talla: {p['altura_cm']}cm")
        
        question = ONBOARDING_QUESTIONS[next_step]
        if summary_parts:
            # Solo mencionamos si ya hay algo guardado
            confirmation = " (Tengo registrado: " + ", ".join(summary_parts) + ")"
            return f"¡Entendido!{confirmation}. Ahora, {question[0].lower() + question[1:]}"
        
        return question
    else:
        _set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
        return "¡Excelente, perfil completado! 🎯 Muchas gracias por tu tiempo. Esto me ayudará a darte recomendaciones mucho más precisas. ¿En qué más puedo ayudarte hoy?"


async def _save_profile_field(session: AsyncSession, uid: int, field: str, value: str):
    """Upsert the specific profile column in perfil_nutricional"""
    
    col_map = {
        OnboardingStep.EDAD.value: "edad",
        OnboardingStep.ALERGIAS.value: "alergias",
        OnboardingStep.ENFERMEDADES.value: "enfermedades",
        OnboardingStep.TIPO_DIETA.value: "tipo_dieta",
        OnboardingStep.OBJETIVO.value: "objetivo_nutricional",
        OnboardingStep.PESO.value: "peso_kg",
        OnboardingStep.ALTURA.value: "altura_cm",
        OnboardingStep.REGION.value: "region"
    }
    
    db_col = col_map.get(field)
    if not db_col:
        return

    # To upsert dynamically, we must rely on safe parameterized queries.
    sql = f"""
        INSERT INTO perfil_nutricional (usuario_id, {db_col}, actualizado_en)
        VALUES (:uid, :val, :upd)
        ON CONFLICT (usuario_id) 
        DO UPDATE SET {db_col} = EXCLUDED.{db_col}, actualizado_en = EXCLUDED.actualizado_en
    """
    
    # Casting correct types con protección para None
    query_val = value
    try:
        if value is not None and db_col in ("peso_kg", "altura_cm"):
            query_val = float(re.sub(r"[^\d\.]", "", str(value)))
        elif value is not None and db_col == "edad":
            query_val = int(re.sub(r"\D", "", str(value)))
    except Exception:
        logger.warning(f"Error casting {value} for column {db_col}. Saving as None.")
        query_val = None

    await session.execute(
        text(sql),
        {"uid": uid, "val": query_val, "upd": get_now_peru()}
    )


async def _find_next_missing_step(session: AsyncSession, uid: int, cached_profile: Optional[dict] = None) -> Optional[str]:
    """Busca el primer paso de la lista que no tiene valor en el perfil. Siempre consulta DB si no se pasa cache."""
    p = cached_profile
    if p is None:
        res = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": uid})
        p = res.mappings().fetchone()
    
    if not p:
        logger.debug("_find_next_missing_step: no profile found for user=%s, returning 'edad'", uid)
        return OnboardingStep.EDAD.value

    # Metadata de campos omitidos
    skipped = p.get("skipped_fields", {})
    if not isinstance(skipped, dict):
        skipped = {}

    col_map = {
        OnboardingStep.EDAD.value: "edad",
        OnboardingStep.ALERGIAS.value: "alergias",
        OnboardingStep.ENFERMEDADES.value: "enfermedades",
        OnboardingStep.TIPO_DIETA.value: "tipo_dieta",
        OnboardingStep.OBJETIVO.value: "objetivo_nutricional",
        OnboardingStep.PESO.value: "peso_kg",
        OnboardingStep.ALTURA.value: "altura_cm",
        OnboardingStep.REGION.value: "region"
    }

    # Recorremos los pasos (saltando INVITACION que es el 0)
    for step in ONBOARDING_STEPS_ORDER[1:]:
        col = col_map.get(step.value)
        if not col: continue
        
        # Saltamos si el usuario ya decidió omitir este campo explícitamente
        if skipped.get(step.value):
            continue

        val = p.get(col)
        # Consideramos vacío si es None o string vacío (o placeholder que no sea estandarizado)
        if val is None or (isinstance(val, str) and len(val.strip()) == 0):
            logger.debug("_find_next_missing_step: user=%s, missing field %s (col=%s)", uid, step.value, col)
            return step.value

    logger.debug("_find_next_missing_step: user=%s, all fields filled!", uid)
    return None


async def _mark_field_as_skipped(session: AsyncSession, uid: int, field: str):
    """Marca un campo como omitido explícitamente en el JSONB."""
    # Asegurarse de que el registro existe primero con el upsert normal pero val=NULL
    sql_init = """
        INSERT INTO perfil_nutricional (usuario_id, actualizado_en)
        VALUES (:uid, :upd)
        ON CONFLICT (usuario_id) DO NOTHING
    """
    await session.execute(text(sql_init), {"uid": uid, "upd": get_now_peru()})

    # Ahora actualizamos el JSONB
    sql_skip = f"""
        UPDATE perfil_nutricional 
        SET skipped_fields = skipped_fields || jsonb_build_object(:field, true),
            actualizado_en = :upd
        WHERE usuario_id = :uid
    """
    await session.execute(text(sql_skip), {"uid": uid, "field": field, "upd": get_now_peru()})
