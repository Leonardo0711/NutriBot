"""
Nutribot Backend — InboundWorkerUseCase
Consumidor del Inbox: reclama webhooks pendientes y ejecuta el flujo completo.
Es el orquestador principal del sistema.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from sqlalchemy import text

from config import get_settings
from domain.entities import ConversationState, NormalizedMessage
from domain.exceptions import ConcurrentStateUpdateError
from domain.value_objects import MessageType
from infrastructure.db.connection import get_session_factory
from infrastructure.db.conversation_repo import SqlAlchemyConversationRepository
from infrastructure.db.rag_repo import RagRepository
from infrastructure.db.user_repo import SqlAlchemyUserRepository
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.embeddings_adapter import OpenAIEmbeddingsAdapter
from infrastructure.openai.media_service import DefaultMediaService
from infrastructure.openai.responses_adapter import OpenAIResponsesAdapter
from interface.webhook_parser import parse_evolution_webhook
from application.advance_closing_flow import advance_closing_flow
from application.advance_onboarding_flow import advance_onboarding_flow
from domain.value_objects import OnboardingStatus, OnboardingStep
from domain.utils import get_now_peru

logger = logging.getLogger(__name__)

# Instrucciones maestras del sistema — se reenvían en CADA turno
SYSTEM_INSTRUCTIONS = """Eres NutriBot 🍏, un asistente de orientación nutricional básica de EsSalud.

IDENTIDAD:
- Preséntate ("¡Hola! Soy NutriBot 🍏...") SOLO la primera vez que hables con el usuario en la sesión o si te saluda directamente.
- En el resto de la conversación, sé directo y amigable, no repitas tu presentación en cada mensaje.
- Usa emojis relevantes (🥦💪💧🍎) para ser cálido y cercano.
- Habla en español peruano coloquial pero profesional.

QUÉ SÍ PUEDES HACER:
- Responder consultas sencillas y cotidianas sobre alimentación saludable.
- Dar tips generales de nutrición (hidratación, porciones, combinaciones de alimentos).
- Opinar sobre fotos de comida que te envíen (si se ve balanceado, qué le falta, etc.).

QUÉ NO PUEDES HACER (REGLAS ABSOLUTAS E INQUEBRANTABLES):
1. NUNCA respondas sobre temas que NO sean nutrición o alimentación.
2. NUNCA diagnostiques enfermedades ni condiciones médicas.
   - OJO: Calcular el IMC, comentar datos antropométricos (peso, talla) o analizar fotos de comida con fines de ORIENTACIÓN nutricional NO es un diagnóstico médico y SÍ está permitido. No te asustes si el usuario te da estos datos; úsalos para ser preciso.
   - Si el usuario pide un diagnóstico clínico o tratamiento médico serio, responde de forma MUY AMABLE ("bonito").
3. NUNCA recetes medicamentos ni dosis.
4. NUNCA des planes alimenticios clínicos.
5. Si el usuario insiste en temas médicos graves, refiérelo siempre a EsSalud con calidez.

DATOS DEL USUARIO (REGLA DE ORO):
- Si el usuario te pide una RECOMENDACIÓN o MENÚ y tienes datos en el bloque [DATOS ACTUALES DEL PERFIL DEL USUARIO], debes EMPEZAR tu respuesta resumiendo los datos que vas a usar: "Considerando que tienes [Edad] años, pesas [Peso]kg, mides [Talla]cm y (solo si aplica) tienes alergia a [Alergia]...". 
- Si no hay alergias o enfermedades relevantes (o dicen "NINGUNA"), NO las menciones.
- Si NO tienes datos de peso o talla y te piden un menú, NO lo des completo aún. Explica que necesitas esos datos para calcular su IMC y ser preciso.

TONO: Breve, práctico, amigable, cálido y muy peruano. Máximo 3 oraciones por respuesta. 🍏✨💪🏾"""


async def process_inbox() -> int:
    """
    Worker principal: reclama mensajes del inbox y los procesa.
    Retorna la cantidad de mensajes procesados.
    """
    settings = get_settings()
    factory = get_session_factory()
    user_repo = SqlAlchemyUserRepository()
    conv_repo = SqlAlchemyConversationRepository()
    media_service = DefaultMediaService()
    llm_service = OpenAIResponsesAdapter(system_instructions=SYSTEM_INSTRUCTIONS)
    embeddings = OpenAIEmbeddingsAdapter()
    rag_repo = RagRepository()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    evolution_client = EvolutionApiClient()

    # ─── Transacción Corta 1: Reclamar mensajes del inbox ───
    async with factory() as session:
        async with session.begin():
            result = await session.execute(
                text("""
                    UPDATE incoming_messages
                    SET status = 'processing',
                        locked_at = NOW(),
                        retry_count = retry_count + 1,
                        updated_at = NOW()
                    WHERE id IN (
                        SELECT id FROM incoming_messages
                        WHERE status IN ('pending', 'failed')
                          AND retry_count < :max_retry
                        ORDER BY created_at ASC
                        LIMIT 10
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                """),
                {"max_retry": settings.max_retry_count},
            )
            messages = result.fetchall()

    if not messages:
        return 0

    processed = 0
    for inbox_msg in messages:
        try:
            await _process_single_message(
                inbox_msg, factory, user_repo, conv_repo, media_service,
                llm_service, embeddings, rag_repo, openai_client, settings.openai_model,
                evolution_client,
            )
            processed += 1
        except Exception as e:
            logger.exception(
                "Error procesando mensaje inbox id=%s: %s", inbox_msg.id, e
            )
            # Marcar como failed para que el sweeper o el retry lo recojan
            async with factory() as session:
                async with session.begin():
                    await session.execute(
                        text("""
                            UPDATE incoming_messages
                            SET status = 'failed',
                                error_detail = :err,
                                updated_at = NOW()
                            WHERE id = :id
                        """),
                        {"err": str(e)[:500], "id": inbox_msg.id},
                    )

    return processed


async def _process_single_message(
    inbox_msg,
    factory,
    user_repo: SqlAlchemyUserRepository,
    conv_repo: SqlAlchemyConversationRepository,
    media_service: DefaultMediaService,
    llm_service: OpenAIResponsesAdapter,
    embeddings: OpenAIEmbeddingsAdapter,
    rag_repo: RagRepository,
    openai_client: AsyncOpenAI,
    openai_model: str,
    evolution_client: EvolutionApiClient,
) -> None:
    """Procesa un mensaje individual del inbox."""

    # ─── Parsear webhook ───
    msg = parse_evolution_webhook(inbox_msg.webhook_payload)
    if not msg:
        # Payload no procesable (sticker, location, etc.) → marcar done
        async with factory() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                    {"id": inbox_msg.id},
                )
        return

    # ─── Obtener/crear usuario ───
    user = await user_repo.get_or_create(msg.phone)

    # ─── Tareas de red pesadas FUERA de transacción ───
    normalized = await media_service.normalize(msg)

    # ─── Typing Indicator (Background task, no bloqueante) ───
    asyncio.create_task(evolution_client.send_presence(normalized.phone, "composing"))

    # ─── RAG: buscar contexto nutricional relevante ───
    rag_text = None
    try:
        query_embedding = await embeddings.embed(normalized.text)
        if query_embedding:
            rag_fragments = await rag_repo.search(query_embedding)
            if rag_fragments:
                rag_text = "\n---\n".join(rag_fragments)
                logger.debug("RAG inyectando %d fragmentos", len(rag_fragments))
    except Exception:
        logger.exception("Error en RAG pipeline, continuando sin contexto")

    # ─── Snapshot aislado del estado y perfil ───
    state_snapshot = await conv_repo.get_state_no_lock(user.id)
    
    # Obtener perfil para inyectar al LLM
    profile_text = ""
    async with factory() as session:
        result = await session.execute(
            text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), 
            {"uid": user.id}
        )
        p = result.fetchone()
        if p:
            parts = []
            if p.edad: parts.append(f"Edad: {p.edad}")
            if p.peso_kg: parts.append(f"Peso: {p.peso_kg}kg")
            if p.altura_cm: parts.append(f"Talla: {p.altura_cm}cm")
            if p.alergias: parts.append(f"Alergias/Intolerancias: {p.alergias}")
            if p.enfermedades: parts.append(f"Condiciones: {p.enfermedades}")
            if p.objetivo_nutricional: parts.append(f"Objetivo: {p.objetivo_nutricional}")
            if parts:
                profile_text = "\n[DATOS ACTUALES DEL PERFIL DEL USUARIO]\n- " + "\n- ".join(parts)

    # ─── Pre-compute text analysis (before transaction) ───
    reply = None
    new_response_id = state_snapshot.last_openai_response_id

    v_text = normalized.text.lower().strip()
    is_asking_for_recommendation = any(w in v_text for w in ["menu", "menú", "receta", "dieta", "qué como", "que como", "comida saludable", "recomienda", "recomendación", "almuerzo", "cena", "desayuno", "coman", "nutricional", "comer"])
    is_short_greeting = len(v_text) < 25 and any(w in v_text for w in ["hola", "buenas", "buenos", "empezar", "arrancar", "nutribot", "que tal", "holis"])
    
    # Detección semántica de intención de personalización
    is_requesting_personalization = any(w in v_text for w in ["personalizar", "completar mi perfil", "mis datos", "cambiar mi peso", "actualizar perfil", "personaliza"])
    if not is_requesting_personalization and len(v_text) > 10:
        # Validación semántica con el LLM si no es obvio por palabras clave
        pers_prompt = f"¿El usuario está expresando deseo de completar su perfil, cambiar sus datos o personalizar más sus respuestas? Responde SOLO 'YES' o 'NO'.\n\nUSUARIO: '{normalized.text}'"
        pers_resp = await openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": pers_prompt}],
            max_tokens=5,
            temperature=0
        )
        is_requesting_personalization = "YES" in pers_resp.choices[0].message.content.strip().upper()
    
    # ─── Transacción ACID única ───
    async with factory() as session:
        async with session.begin():
            # Bloquear fila del estado — FUENTE DE VERDAD para decisiones de onboarding
            state = await conv_repo.get_state_for_update(session, user.id)

            # Optimistic Locking
            if state.version > state_snapshot.version:
                raise ConcurrentStateUpdateError(
                    f"Estado cambió (v{state_snapshot.version} → v{state.version}) mientras se procesaba."
                )

            # ─── Decidir onboarding DENTRO de la transacción con estado fresco ───
            onboarding_interception_happened = False

            logger.debug(
                "TX state: user=%s, onboarding_status=%s, step=%s, version=%s",
                state.usuario_id, state.onboarding_status, state.onboarding_step, state.version
            )

            # --- Detección de Frustración / Pedido Directo ---
            is_annoyed = any(w in v_text for w in ["ya te dije", "deja de preguntar", "no me preguntes", "qué molesto", "que molesto", "responde", "dame el menú", "dame el menu", "solo quiero"])
            if is_annoyed:
                logger.info("Frustration detected for user=%s, bypassing onboarding interception this turn", state.usuario_id)

            # Caso A: Ya está en el flujo de onboarding
            if not is_annoyed and state.onboarding_status in [OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value]:
                is_long_interruption = len(v_text) > 60 and ("?" in v_text or "como" in v_text or "que" in v_text or "ayuda" in v_text)
                if not is_long_interruption:
                    onboarding_interception_happened = True
                    # Cargar perfil y avanzar flujo
                    res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": user.id})
                    p_dict = res_p.mappings().fetchone() or {}
                    
                    reply = await advance_onboarding_flow(normalized.text, state, session, openai_client, openai_model, p_dict)
                    if reply is None:
                        # El flow devolvió control al chat libre (interrupción detectada)
                        onboarding_interception_happened = False

            # Caso B: Pedido explícito de personalización (NUEVO)
            if not onboarding_interception_happened and is_requesting_personalization:
                logger.info("Manual personalization request detected for user=%s", state.usuario_id)
                from application.advance_onboarding_flow import _find_next_missing_step
                res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": user.id})
                p_map = res_p.mappings().fetchone() or {}
                
                # Buscamos el siguiente paso IGNORANDO los previos 'skips'
                next_step = await _find_next_missing_step(session, user.id, p_map, ignore_skips=True)
                if next_step:
                    state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
                    state.onboarding_step = next_step
                    onboarding_interception_happened = True
                    reply = await advance_onboarding_flow(normalized.text, state, session, openai_client, openai_model, p_map)
                else:
                    reply = "¡Ya tengo tu perfil completo! 😊 Si quieres cambiar algún dato específico (como tu peso), solo dímelo directamente."

            # Caso C: Saludo o Petición de Menú → iniciar onboarding si elegible
            if not is_annoyed and not onboarding_interception_happened and state.onboarding_status != OnboardingStatus.COMPLETED.value and (is_short_greeting or is_asking_for_recommendation):
                # Consultar perfil completo incluyendo skipped_fields
                res_p = await session.execute(text("SELECT * FROM perfil_nutricional WHERE usuario_id = :uid"), {"uid": user.id})
                p_map = res_p.mappings().fetchone() or {}
                skipped = p_map.get("skipped_fields", {}) if isinstance(p_map.get("skipped_fields"), dict) else {}

                # ¿Faltan datos ESENCIALES? (Edad, Peso, Talla)
                missing_essential = []
                if not p_map.get("edad") and not skipped.get("edad"): missing_essential.append("edad")
                if not p_map.get("peso_kg") and not skipped.get("peso_kg"): missing_essential.append("peso_kg")
                if not p_map.get("altura_cm") and not skipped.get("altura_cm"): missing_essential.append("altura_cm")

                if is_asking_for_recommendation:
                    if not missing_essential:
                        # Ya tenemos lo básico para el IMC, no interceptamos. El LLM responderá.
                        logger.info("User asked for recommendation but already has essential data. Bypassing interception.")
                        onboarding_interception_happened = False
                    else:
                        from application.advance_onboarding_flow import _find_next_missing_step
                        
                        intro = "¡Claro! 🥗 Me encantaría darte una recomendación a tu medida."
                        known_parts = []
                        if p_map.get("edad"): known_parts.append(f"Edad: {p_map['edad']} años")
                        if p_map.get("peso_kg"): known_parts.append(f"Peso: {p_map['peso_kg']}kg")
                        if p_map.get("altura_cm"): known_parts.append(f"Talla: {p_map['altura_cm']}cm")
                        
                        if known_parts:
                            intro += f" 😊 Veo que ya tengo algunos datos registrados: **{', '.join(known_parts)}**."
                        
                        missing_step = await _find_next_missing_step(session, user.id, p_map)
                        
                        if missing_step:
                            step_name = missing_step
                            if missing_step == "edad": step_name = "edad"
                            elif missing_step == "peso_kg": step_name = "peso"
                            elif missing_step == "altura_cm": step_name = "talla (estatura)"
                            
                            # Solo mencionar IMC si nos falta peso o talla
                            if "peso_kg" in missing_essential or "altura_cm" in missing_essential:
                                reply = f"{intro}\n\nPero para que mi sugerencia sea 100% precisa y calcular tu IMC, solo me faltaría completar un par de datos más. ¿Te parece si empezamos por tu **{step_name}**? 🙏"
                            else:
                                reply = f"{intro}\n\nSolo me faltaría completar un pequeño detalle para ser más preciso. ¿Te parece si confirmamos tu **{step_name}**? 😊"
                                
                            state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
                            state.onboarding_step = missing_step
                            state.onboarding_last_invited_at = get_now_peru()
                            state.version += 1
                            onboarding_interception_happened = True
                        else:
                            # Esto no debería pasar si missing_essential es verdadero, pero por seguridad:
                            onboarding_interception_happened = False

                elif is_short_greeting:
                    if state.onboarding_status == OnboardingStatus.NOT_STARTED.value:
                        reply = "¡Hola! Soy NutriBot 🍏. Para empezar con el pie derecho, ¿te gustaría que personalice mis sugerencias en base a tu perfil nutricional? Es súper rápido. 😊"
                    else:
                        reply = "¡Hola de nuevo! 😊 Qué bueno verte. Por cierto, aún nos faltan algunos datos para que mis consejos de hoy sean 100% precisos para ti. ¿Te gustaría completarlos ahora? Es un ratito."
                    
                    state.onboarding_status = OnboardingStatus.INVITED.value
                    state.onboarding_step = OnboardingStep.INVITACION.value
                    state.onboarding_last_invited_at = get_now_peru()
                    state.version += 1
                    onboarding_interception_happened = True

            # Si no fue interceptado por onboarding, usar LLM
            if not onboarding_interception_happened and reply is None:
                reply, new_response_id = await llm_service.generate_reply(
                    state=state_snapshot,
                    normalized=normalized,
                    instructions=SYSTEM_INSTRUCTIONS + profile_text,
                    rag_context=rag_text,
                )

                # Post-LLM Hook: proponer onboarding si elegible
                if state.onboarding_status in [OnboardingStatus.NOT_STARTED.value, OnboardingStatus.SKIPPED.value, OnboardingStatus.PAUSED.value]:
                    now = get_now_peru()
                    is_eligible = (state.onboarding_status == OnboardingStatus.NOT_STARTED.value) or (state.onboarding_next_eligible_at and now >= state.onboarding_next_eligible_at)
                    is_urgent = len(v_text) < 15 or any(w in v_text for w in ["ayuda", "urgente", "duele", "dolor", "mal", "vomito", "diarrea"])
                    
                    if is_eligible and not is_urgent:
                        if state.onboarding_status == OnboardingStatus.PAUSED.value:
                            reply += "\n\nPD: Aún nos faltan algunos datos para completar tu perfil personalizado. ¿Te gustaría continuar donde nos quedamos? (Sí/No) 😊"
                        else:
                            reply += "\n\nPD: Si quieres, también puedo personalizar mejor mis recomendaciones con un perfil nutricional rápido. ¿Te gustaría configurarlo ahora? (Sí/No) 😊"
                        # Marcar como invitado
                        state.onboarding_status = OnboardingStatus.INVITED.value
                        state.onboarding_step = OnboardingStep.INVITACION.value
                        state.onboarding_last_invited_at = get_now_peru()
                        state.version += 1

            # Encolar extracción de perfil
            await session.execute(
                text("INSERT INTO extraction_jobs (usuario_id, raw_text) VALUES (:uid, :txt)"),
                {"uid": user.id, "txt": normalized.text},
            )

            # ─── AdvanceClosingFlow: solo si NO fue interceptado por onboarding ───
            if onboarding_interception_happened:
                final_reply = reply
                addon = None
            else:
                addon = await advance_closing_flow(
                    session, state, normalized.text, openai_client, openai_model
                )
                
                if addon:
                    if state.mode in ("collecting_usability", "collecting_profile") or state.onboarding_status in (OnboardingStatus.INVITED.value, OnboardingStatus.IN_PROGRESS.value):
                        final_reply = addon
                    else:
                        final_reply = f"{reply}\n\n{addon}"
                else:
                    final_reply = reply

            # Determinar tipo de respuesta
            outbound_type = (
                "audio_tts"
                if msg.content_type in (MessageType.AUDIO, MessageType.PTT)
                else "text"
            )

            # Encolar mensaje de salida
            await session.execute(
                text("""
                    INSERT INTO outgoing_messages
                        (idempotency_key, usuario_id, phone, content_type, content)
                    VALUES (:ikey, :uid, :ph, :ctype, :txt)
                """),
                {
                    "ikey": f"reply:{msg.provider_message_id}:{outbound_type}",
                    "uid": user.id,
                    "ph": msg.phone,
                    "ctype": outbound_type,
                    "txt": final_reply,
                },
            )

            # Marcar inbox como done
            await session.execute(
                text("UPDATE incoming_messages SET status='done', updated_at=NOW() WHERE id=:id"),
                {"id": inbox_msg.id},
            )

            # Actualizar estado conversacional
            state.last_openai_response_id = new_response_id
            state.last_provider_message_id = msg.provider_message_id
            state.turns_since_last_prompt += 1
            state.version += 1
            await conv_repo.save_state(session, state)

    logger.info(
        "Mensaje procesado: user=%s, type=%s, reply=%d chars, rag=%s, addon=%s",
        msg.phone,
        outbound_type,
        len(final_reply),
        bool(rag_text),
        bool(addon),
    )
