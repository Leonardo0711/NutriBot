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
from domain.parsers import parse_weight, parse_height, parse_age, standardize_text_list

logger = logging.getLogger(__name__)

ONBOARDING_QUESTIONS: dict[str, str] = {
    OnboardingStep.EDAD.value: "Para empezar, ¿cuántos años tienes? 🎂",
    OnboardingStep.PESO.value: "¿Y cuál es tu peso aproximado en kilos? ⚖️",
    OnboardingStep.ALTURA.value: "¿Y cuál es tu estatura aproximada? Puedes usar centímetros o metros (Ej. 1.70m, 170cm) 📐",
    OnboardingStep.TIPO_DIETA.value: "¿Sigues algún tipo de dieta especial? (Ej: Vegana, Keto, Sin gluten, o ninguna) 🥗",
    OnboardingStep.ALERGIAS.value: "¿Tienes alguna alergia o intolerancia alimentaria? 🍎",
    OnboardingStep.ENFERMEDADES.value: "¿Padeces alguna condición de salud relevante? (Ej: Diabetes, Hipertensión) 🏥",
    OnboardingStep.RESTRICCIONES.value: "¿Tienes alguna **restricción alimentaria** por religión, ética o gusto personal? (Ej: No como cerdo, no como carnes rojas, no me gusta el brócoli...) 🚫",
    OnboardingStep.OBJETIVO.value: "¿Cuál es tu principal objetivo nutricional? (Ej. comer más sano, bajar de peso, subir masa muscular, controlar mi azúcar...) 🎯",
    OnboardingStep.PROVINCIA.value: "¿En qué **provincia** de Perú te encuentras? 😊",
    OnboardingStep.DISTRITO.value: "¿Y en qué **distrito** vives? (Para recomendaciones locales) 🏠"
}


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
        w = parse_weight(v)
        if w:
            return True, str(w), None
        return False, None, "No logré captar el peso. ¿Podrías decirlo en kilos o libras? (O escribe 'saltar' si prefieres no decirlo aún)"
    
    elif step == OnboardingStep.ALTURA:
        if is_skip: return True, None, None
        h = parse_height(v)
        if h:
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
    current_profile: Optional[dict] = None,
    treat_ninguna_as_missing: bool = False
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

    # BYPASS CRÍTICO: Si el usuario está pidiendo una recomendación/menú clara,
    # pausamos el onboarding y NO llamamos al extractor para no sobreescribir datos por error.
    is_rec_intent = any(w in vl for w in ["menu", "receta", "dieta", "qué como", "que como", "comida saludable", "recomienda", "recomendación"])
    if is_rec_intent:
        logger.info("advance_onboarding: recommendation intent detected, bypassing extraction to protect data.")
        _set_onboarding_state(state, OnboardingStatus.PAUSED, state.onboarding_step)
        return None

    if is_interruption and not any(w in vl for w in ["saltar", "paso", "no"]):
        # Abandono natural a mitad del formulario: pausamos
        logger.info("advance_onboarding: interruption detected for user=%s, pausing", state.usuario_id)
        _set_onboarding_state(state, OnboardingStatus.PAUSED, state.onboarding_step)
        return None # Devuelve control al Chat libre

    current_step = state.onboarding_step
    if not current_step:
        logger.warning("advance_onboarding: no current_step for user=%s, returning None", state.usuario_id)
        return None

    # 2. Extracción Multi-campo (Usando la lógica central del SyncProfileProcessor)
    # Esto permite que si el usuario dice "no enfermedades pero sí alergia al maní", se guarden ambos correctamente.
    from application.sync_profile_processor import process_profile_sync
    extracted = await process_profile_sync(
        user_text, state.usuario_id, session, openai_client, openai_model,
        current_step=current_step
    )
    logger.info("advance_onboarding: multi-intent extraction results: %s", extracted)

    # Si no se extrajo nada y no es un "no/saltar", pedimos aclaración
    if not extracted and not ("saltar" in vl or "paso" in vl or "omitir" in vl or "no" in vl):
         return f"No logré captar ese dato para tu perfil. ¿Podrías decírmelo de forma más clara? 😊"

    # Si el usuario dijo explícitamente "saltar" o similar para el paso actual y no hubo extracción
    if not extracted and ("saltar" in vl or "paso" in vl or "omitir" in vl or "luego" in vl or "siguiente" in vl):
        await _mark_field_as_skipped(session, state.usuario_id, current_step)
        logger.info("advance_onboarding: marked %s as skipped for user=%s", current_step, state.usuario_id)

    # Avanzar al siguiente paso REALMENTE vacío (LINEALIDAD ESTRICTA + MEMORIA DE TURNO)
    # Ignoramos los campos que el usuario YA mencionó en este mensaje (extracted.keys())
    # Esto evita preguntar "¿Tienes restricciones?" justo después de que el usuario dijo "no tengo restricciones"
    updated_cols = list(extracted.keys()) if extracted else []
    
    current_idx = -1
    for i, s in enumerate(ONBOARDING_STEPS_ORDER):
        if s.value == current_step:
            current_idx = i
            break
            
    next_step = await _find_next_missing_step(
        session, 
        state.usuario_id, 
        treat_ninguna_as_missing=treat_ninguna_as_missing, 
        start_from_idx=current_idx + 1,
        ignore_cols=updated_cols
    )
    logger.info("advance_onboarding: after save, next_missing_step=%s for user=%s", next_step, state.usuario_id)

    if next_step:
        _set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, next_step)
        
        # Respuesta conversacional basada en si hubo extracciones previas
        if updated_cols:
            transition = "¡Perfecto! Ya anoté esos detalles. ✍️"
            if len(updated_cols) == 1 and updated_cols[0] == "region":
                transition = "¡Qué bueno! Me encanta esa zona. 📍"
            
            return f"{transition} Ahora, para seguir personalizando tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
        
        # Si no hubo extracción nueva pero avanzamos
        return f"Entendido. 😊 Siguiendo con tu perfil, {ONBOARDING_QUESTIONS[next_step][0].lower() + ONBOARDING_QUESTIONS[next_step][1:]}"
    else:
        # VALIDACIÓN FINAL: Antes de cerrar, aseguramos que Provincia y Distrito estén REALMENTE en la DB
        res_final = await session.execute(text("SELECT provincia, distrito FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": state.usuario_id})
        p_final = res_final.fetchone()
        if p_final and (not p_final.provincia or not p_final.distrito):
            # Forzamos que falten
            force_step = OnboardingStep.PROVINCIA.value if not p_final.provincia else OnboardingStep.DISTRITO.value
            _set_onboarding_state(state, OnboardingStatus.IN_PROGRESS, force_step)
            logger.warning("Final guard: found missing location for user=%s, force-prompting %s", state.usuario_id, force_step)
            return f"¡Casi terminamos! 🎯 Solo un pequeño detalle final: {ONBOARDING_QUESTIONS[force_step].lower()}"

        _set_onboarding_state(state, OnboardingStatus.COMPLETED, None)
        return "¡Excelente, perfil completado! 🎯 Muchas gracias por tu tiempo. Esto me ayudará a darte recomendaciones mucho más precisas. ¿En qué más puedo ayudarte hoy?"


async def _save_profile_field(session: AsyncSession, uid: int, field: str, value: str):
    """Upsert the specific profile column in perfil_nutricional"""
    
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
    
    # El valor ya viene normalizado del flujo anterior (parse_weight, etc.)
    # Solo aseguramos el casting final para la DB si es necesario, pero sin lógica de limpieza extra
    query_val = value
    if db_col in ("peso_kg", "altura_cm") and value is not None:
        try: query_val = float(value)
        except Exception: query_val = None
    elif db_col == "edad" and value is not None:
        try: query_val = int(value)
        except Exception: query_val = None

    await session.execute(
        text(sql),
        {"uid": uid, "val": query_val, "upd": get_now_peru()}
    )


async def _find_next_missing_step(session: AsyncSession, uid: int, cached_profile: Optional[dict] = None, ignore_skips: bool = False, treat_ninguna_as_missing: bool = False, skip_step: Optional[str] = None, start_from_idx: Optional[int] = None, ignore_cols: Optional[list[str]] = None) -> Optional[str]:
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
        OnboardingStep.RESTRICCIONES.value: "restricciones_alimentarias",
        OnboardingStep.TIPO_DIETA.value: "tipo_dieta",
        OnboardingStep.OBJETIVO.value: "objetivo_nutricional",
        OnboardingStep.PESO.value: "peso_kg",
        OnboardingStep.ALTURA.value: "altura_cm",
        OnboardingStep.REGION.value: "region",
        OnboardingStep.PROVINCIA.value: "provincia",
        OnboardingStep.DISTRITO.value: "distrito"
    }

    # Recorremos los pasos (saltando INVITACION que es el 0)
    # start_from_idx permite asegurar linealidad
    base_idx = start_from_idx if start_from_idx is not None else 1
    for step in ONBOARDING_STEPS_ORDER[base_idx:]:
        if skip_step and step.value == skip_step:
            continue
            
        col = col_map.get(step.value)
        if not col: continue

        # IGNORAR si ya se mencionó/actualizó en este turno (Memoria de turno)
        if ignore_cols and col in ignore_cols:
            continue
        
        # Saltamos si el usuario ya decidió omitir este campo explícitamente, 
        # A MENOS que se haya pedido ignorar los skips (personalización manual).
        if not ignore_skips and skipped.get(step.value):
            continue

        val = p.get(col)
        # Consideramos vacío si es None o string vacío
        is_empty = val is None or (isinstance(val, str) and len(val.strip()) == 0)
        
        # Si se solicita explícitamente (onboarding manual), tratamos "NINGUNA" como algo a completar
        if treat_ninguna_as_missing and isinstance(val, str) and val.upper() == "NINGUNA":
            is_empty = True

        if is_empty:
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
