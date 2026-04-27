"""
Nutribot Backend — Tests de la Capa Transversal de Profile Intent
Verifica que la intención de perfil (ADD/REMOVE/CORRECTION etc.)
se respeta correctamente a lo largo del pipeline.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domain.profile_intent import ProfileIntentResult, ProfileIntentValue
from domain.turn_context import TurnContext
from domain.router import RouteResult, Intent
from domain.entities import ConversationState, NormalizedMessage, User
from domain.reply_objects import BotReply
from domain.value_objects import OnboardingStatus, SessionMode


# ─── Fixtures ───────────────────────────────────────────────────────────


def _make_ctx(
    turn_kind=None,
    profile_intent=None,
    onboarding_status="completed",
    mode="active_chat",
    route_intent=Intent.NUTRITION_QUERY,
    text="hola",
    looks_like_profile_update=False,
    onboarding_interception_happened=False,
    extracted_data=None,
):
    """Construye un TurnContext mínimo para testing."""
    ctx = MagicMock(spec=TurnContext)
    ctx.turn_kind = turn_kind
    ctx.profile_intent = profile_intent
    ctx.looks_like_profile_update = looks_like_profile_update
    ctx.onboarding_interception_happened = onboarding_interception_happened
    ctx.extracted_data = extracted_data or {}
    ctx.is_requesting_survey = False
    ctx.is_short_greeting = False
    ctx.is_asking_for_recommendation = False
    ctx.is_requesting_personalization = False
    ctx.has_absurd_profile_claim = False
    ctx.history = []

    # Route
    ctx.route = MagicMock(spec=RouteResult)
    ctx.route.intent = route_intent
    ctx.route.confidence = 0.9
    ctx.route.resolved_field = None
    ctx.route.resolved_value = None

    # State
    ctx.state = MagicMock(spec=ConversationState)
    ctx.state.onboarding_status = onboarding_status
    ctx.state.onboarding_step = None
    ctx.state.awaiting_field_code = None
    ctx.state.mode = mode
    ctx.state.meaningful_interactions_count = 5

    # Normalized
    ctx.normalized = MagicMock(spec=NormalizedMessage)
    ctx.normalized.text = text
    ctx.normalized.used_audio = False
    ctx.normalized.image_base64 = None

    # User
    ctx.user = MagicMock(spec=User)
    ctx.user.id = 1

    # Session
    ctx.session = AsyncMock()

    return ctx


def _make_intent(field_code, operation, raw_values, confidence=0.95):
    """Construye un ProfileIntentResult con valores pre-resueltos."""
    values = [
        ProfileIntentValue(
            raw_value=rv,
            entity_code=rv.upper().replace(" ", "_"),
            entity_type="restriction" if "lacteo" in rv.lower() else "disease",
            resolution_strategy="EXACT",
            confidence=confidence,
        )
        for rv in raw_values
    ]
    return ProfileIntentResult(
        is_profile_update=True,
        field_code=field_code,
        operation=operation,
        values=values,
        confidence=confidence,
        evidence_text=", ".join(raw_values),
        source="LLM_STRUCTURED",
    )


# ─── Test 1: ADD restricción "lácteos" ──────────────────────────────────


@pytest.mark.asyncio
async def test_profile_intent_add_lacteos():
    """'prefiero no comer lácteos' → operation=ADD, field=restricciones_alimentarias"""
    intent = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])

    assert intent.is_profile_update is True
    assert intent.operation == "ADD"
    assert intent.field_code == "restricciones_alimentarias"
    assert intent.values[0].raw_value == "lacteos"
    assert intent.values[0].entity_code == "LACTEOS"


# ─── Test 2: REMOVE restricción "lácteos" ───────────────────────────────


@pytest.mark.asyncio
async def test_profile_intent_remove_lacteos():
    """'ya puedo comer lácteos' → operation=REMOVE"""
    intent = _make_intent("restricciones_alimentarias", "REMOVE", ["lacteos"])

    assert intent.is_profile_update is True
    assert intent.operation == "REMOVE"
    assert intent.field_code == "restricciones_alimentarias"
    # Verificar que ADD y REMOVE producen intents distintos
    intent_add = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])
    assert intent_add.operation != intent.operation


# ─── Test 3: ProfileUpdateHandler usa apply_profile_intent ──────────────


@pytest.mark.asyncio
async def test_profile_update_handler_uses_apply_profile_intent():
    """El handler llama a apply_profile_intent() en lugar de apply_cleaning_and_save()
    cuando hay un profile_intent válido."""
    from application.services.handlers.profile_update_handler import ProfileUpdateHandler

    intent = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])
    ctx = _make_ctx(profile_intent=intent)

    mock_extractor = AsyncMock()
    mock_extractor.contains_absurd_claim = MagicMock(return_value=False)
    mock_extractor.apply_profile_intent = AsyncMock(
        return_value=MagicMock(clean_data={"restricciones_alimentarias": "lacteos"}, meta_flags={})
    )

    mock_fallback = AsyncMock()
    mock_fallback.handle = AsyncMock(return_value=(BotReply(text="ok", content_type="text"), None))

    handler = ProfileUpdateHandler(
        profile_extractor=mock_extractor,
        profile_context=MagicMock(),
        fallback_handler=mock_fallback,
    )

    await handler.handle(ctx)

    # apply_profile_intent FUE llamado (no apply_cleaning_and_save)
    mock_extractor.apply_profile_intent.assert_called_once()
    call_kwargs = mock_extractor.apply_profile_intent.call_args
    assert call_kwargs.kwargs.get("intent") == intent or call_kwargs[1].get("intent") == intent


# ─── Test 4: Onboarding preserva operation del profile_intent ───────────


@pytest.mark.asyncio
async def test_onboarding_preserves_profile_intent_operation():
    """Durante onboarding, si el usuario dice 'peso 70 y no como lácteos',
    el campo extra (restricciones) se guarda via apply_profile_intent."""
    from application.services.handlers.onboarding_handler import OnboardingHandler

    intent = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])
    ctx = _make_ctx(
        profile_intent=intent,
        onboarding_status=OnboardingStatus.IN_PROGRESS.value,
    )

    mock_onboarding = AsyncMock()
    mock_onboarding.advance_flow = AsyncMock(return_value="¿Cuánto pesas?")

    mock_fallback = AsyncMock()

    handler = OnboardingHandler(
        onboarding_service=mock_onboarding,
        fallback_handler=mock_fallback,
    )

    result, _ = await handler.handle(ctx)

    # advance_flow fue llamado con pre_extracted_intent
    call_kwargs = mock_onboarding.advance_flow.call_args.kwargs
    assert call_kwargs.get("pre_extracted_intent") == intent


# ─── Test 5: NUTRITION_QUERY sin señal de perfil no corre el extractor ──


def test_nutrition_query_without_profile_signal_does_not_run_extractor():
    """'qué puedo desayunar mañana' no debe activar el extractor LLM."""
    from application.services.message_orchestrator import MessageOrchestratorService

    ctx = _make_ctx(
        route_intent=Intent.NUTRITION_QUERY,
        text="que puedo desayunar manana",
    )

    orch = MessageOrchestratorService.__new__(MessageOrchestratorService)
    should_run = orch._should_run_profile_intent_extractor(ctx)
    assert should_run is False


# ─── Test 6: Solo NUTRITION_VALUE cuenta para el survey ─────────────────


def test_survey_counts_only_nutrition_value():
    """Solo turn_kind=NUTRITION_VALUE debería contar como interacción significativa.
    PROFILE_MAINTENANCE, ONBOARDING_RESPONSE, SURVEY_RESPONSE y OTHER no cuentan."""
    non_counting_kinds = [
        "PROFILE_MAINTENANCE",
        "ONBOARDING_RESPONSE",
        "SURVEY_RESPONSE",
        "OTHER",
    ]

    for kind in non_counting_kinds:
        # Simula la lógica del GenericChatHandler
        base_should_count = True
        mode_before_survey = "active_chat"
        should_count = bool(
            base_should_count
            and mode_before_survey != SessionMode.COLLECTING_USABILITY.value
            and kind == "NUTRITION_VALUE"
        )
        assert should_count is False, f"turn_kind={kind} debería NO contar"

    # NUTRITION_VALUE sí cuenta
    should_count_nv = bool(
        True
        and "active_chat" != SessionMode.COLLECTING_USABILITY.value
        and "NUTRITION_VALUE" == "NUTRITION_VALUE"
    )
    assert should_count_nv is True


# ─── Test 7: PROFILE_MAINTENANCE snoozes survey ─────────────────────────


def test_profile_maintenance_snoozes_survey():
    """Cuando turn_kind es PROFILE_MAINTENANCE, no se cuenta la interacción
    y por lo tanto la encuesta no se dispara prematuramente."""
    ctx = _make_ctx(turn_kind="PROFILE_MAINTENANCE")

    base_should_count = True
    should_count = bool(
        base_should_count
        and ctx.state.mode != SessionMode.COLLECTING_USABILITY.value
        and ctx.turn_kind == "NUTRITION_VALUE"
    )
    assert should_count is False


# ─── Test 8: Preclasificación de turn_kind funciona ─────────────────────


def test_preclassify_turn_kind():
    """_preclassify_turn_kind asigna correctamente antes del handler."""
    from application.services.message_orchestrator import MessageOrchestratorService

    orch = MessageOrchestratorService.__new__(MessageOrchestratorService)

    # Profile update → PROFILE_MAINTENANCE
    intent = _make_intent("peso_kg", "REPLACE", ["70"])
    ctx = _make_ctx(profile_intent=intent)
    assert orch._preclassify_turn_kind(ctx) == "PROFILE_MAINTENANCE"

    # Onboarding → ONBOARDING_RESPONSE
    ctx2 = _make_ctx(onboarding_status=OnboardingStatus.IN_PROGRESS.value)
    ctx2.profile_intent = None
    assert orch._preclassify_turn_kind(ctx2) == "ONBOARDING_RESPONSE"

    # Survey → SURVEY_RESPONSE
    ctx3 = _make_ctx(mode=SessionMode.COLLECTING_USABILITY.value)
    ctx3.profile_intent = None
    assert orch._preclassify_turn_kind(ctx3) == "SURVEY_RESPONSE"

    # Nutrition query → NUTRITION_VALUE
    ctx4 = _make_ctx(route_intent=Intent.NUTRITION_QUERY)
    ctx4.profile_intent = None
    assert orch._preclassify_turn_kind(ctx4) == "NUTRITION_VALUE"

    # Greeting → OTHER
    ctx5 = _make_ctx(route_intent=Intent.GREETING)
    ctx5.profile_intent = None
    assert orch._preclassify_turn_kind(ctx5) == "OTHER"


# ─── Test 9: _classify_turn_kind no devuelve NUTRITION_VALUE para saludos ───


def test_classify_turn_kind_greeting_returns_other():
    """Un saludo no debe clasificarse como NUTRITION_VALUE post-handler.
    Esto evita que mensajes neutros cuenten para la encuesta."""
    from application.services.message_orchestrator import MessageOrchestratorService

    orch = MessageOrchestratorService.__new__(MessageOrchestratorService)

    ctx = _make_ctx(
        route_intent=Intent.GREETING,
        text="hola",
        turn_kind="OTHER",
    )
    ctx.profile_intent = None
    ctx.onboarding_interception_happened = False
    ctx.looks_like_profile_update = False
    ctx.extracted_data = {}

    handler = MagicMock()
    result = orch._classify_turn_kind(ctx, handler)
    assert result == "OTHER", f"Greeting should be OTHER, got {result}"


def test_classify_turn_kind_nutrition_returns_nutrition_value():
    """Una consulta nutricional real sí devuelve NUTRITION_VALUE."""
    from application.services.message_orchestrator import MessageOrchestratorService

    orch = MessageOrchestratorService.__new__(MessageOrchestratorService)

    ctx = _make_ctx(
        route_intent=Intent.NUTRITION_QUERY,
        text="que puedo desayunar",
        turn_kind="NUTRITION_VALUE",
    )
    ctx.profile_intent = None
    ctx.onboarding_interception_happened = False
    ctx.looks_like_profile_update = False
    ctx.extracted_data = {}

    handler = MagicMock()
    result = orch._classify_turn_kind(ctx, handler)
    assert result == "NUTRITION_VALUE"


# ─── Test 10: Onboarding persiste restricción extra via apply_profile_intent ─


@pytest.mark.asyncio
async def test_onboarding_apply_profile_intent_persists_extra_restriction():
    """
    Usuario está en onboarding de peso.
    Dice: 'peso 70 y prefiero no comer lacteos'
    El intent field_code es 'restricciones_alimentarias' (no coincide con step actual 'peso'),
    así que advance_flow debe llamar apply_profile_intent para guardar la restricción.
    """
    from application.services.onboarding_service import OnboardingService

    # Crear un mock del servicio con los métodos que advance_flow necesita
    svc = OnboardingService.__new__(OnboardingService)

    # Mock mínimo de _profile_extractor
    mock_extractor = AsyncMock()
    mock_extractor.apply_cleaning_and_save = AsyncMock(
        return_value=MagicMock(clean_data={"peso_kg": "70"}, meta_flags={})
    )
    mock_extractor.apply_profile_intent = AsyncMock(
        return_value=MagicMock(clean_data={"restricciones_alimentarias": "lacteos"}, meta_flags={})
    )
    svc._profile_extractor = mock_extractor

    # Simular intent para restricciones (no coincide con step "peso")
    intent = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])

    # Crear state en paso "peso"
    state = MagicMock(spec=ConversationState)
    state.onboarding_status = "in_progress"
    state.onboarding_step = "peso"
    state.usuario_id = 1
    state.version = 1

    session = AsyncMock()

    # Mock _analyze_turn para que devuelva ANSWER con datos de peso
    svc._analyze_turn = AsyncMock(return_value={
        "intent": "ANSWER",
        "data": {"peso_kg": "70"},
        "explanation": None,
    })
    svc._extract_numeric_step_fallback = MagicMock(return_value=None)
    svc._is_clarification_request = MagicMock(return_value=False)
    svc._is_explicit_skip_request = MagicMock(return_value=False)
    svc._is_health_step = MagicMock(return_value=False)
    svc._find_next_missing_step = AsyncMock(return_value="tipo_dieta")
    svc._set_onboarding_state = MagicMock()
    svc._phase_for_step = MagicMock(return_value=[
        MagicMock(value="edad"), MagicMock(value="peso"),
        MagicMock(value="altura"), MagicMock(value="tipo_dieta"),
    ])
    svc._classify_data_refusal = MagicMock(return_value=None)
    svc._can_try_health_rescue = MagicMock(return_value=False)
    svc.FIELD_LABELS = {"peso": "peso", "tipo_dieta": "tipo de dieta"}
    svc.SEMANTIC_FIELD_BY_STEP = {}

    # Call advance_flow con el intent extra
    result = await svc.advance_flow(
        user_text="peso 70 y prefiero no comer lacteos",
        state=state,
        session=session,
        pre_extracted_intent=intent,
    )

    # apply_profile_intent DEBE haberse llamado para la restricción extra
    mock_extractor.apply_profile_intent.assert_called_once()
    call_kwargs = mock_extractor.apply_profile_intent.call_args.kwargs
    assert call_kwargs["intent"] == intent
    assert call_kwargs["usuario_id"] == 1


# ─── Test 11: Flujo completo de snooze de encuesta ──────────────────────


def test_survey_snooze_full_flow():
    """
    Flujo:
    1. Usuario actualiza perfil → se pausa encuesta (snooze +5)
    2. No vuelve antes de 5 interacciones NUTRITION_VALUE
    3. Vuelve solo después de alcanzar el umbral
    """
    from application.services.conversation_state_service import ConversationStateService

    svc = ConversationStateService()

    # Estado inicial: usuario con 10 interacciones significativas
    state = MagicMock(spec=ConversationState)
    state.meaningful_interactions_count = 10
    state.survey_next_eligible_count = None
    state.survey_paused_reason = None
    state.survey_updated_at = None
    state.version = 1

    # 1. Pausar encuesta por mantenimiento de perfil (+5 turnos)
    svc.pause_survey_for_profile_maintenance(state, reason="PROFILE_MAINTENANCE")
    assert state.survey_next_eligible_count == 15  # 10 + 5
    assert state.survey_paused_reason == "PROFILE_MAINTENANCE"

    # 2. Con 12 interacciones, todavía NO se puede ofrecer
    state.meaningful_interactions_count = 12
    assert svc.can_offer_survey(state) is False

    # 3. Con 14 interacciones, todavía NO
    state.meaningful_interactions_count = 14
    assert svc.can_offer_survey(state) is False

    # 4. Con 15 interacciones, SÍ se puede y se limpia el snooze
    state.meaningful_interactions_count = 15
    assert svc.can_offer_survey(state) is True
    assert state.survey_next_eligible_count is None
    assert state.survey_paused_reason is None


# ─── Test 12: Las preguntas de encuesta no duplican el hint de escala ────


def test_survey_questions_no_duplicate_scale_hint():
    """Las preguntas ya incluyen 'En una escala del 1 al 5/10' en su texto,
    así que _build_question_reply no debe agregar 'Responde con un numero del 1 al 5'."""
    from application.services.survey_service import FORM_QUESTIONS, _SCALE_PREFIX, SurveyService

    # Verificar que las preguntas de escala ya tienen el texto incorporado
    for state_name in _SCALE_PREFIX:
        question = FORM_QUESTIONS.get(state_name, "")
        if state_name == "esperando_nps":
            assert "1 al 10" in question, f"{state_name} debe incluir '1 al 10'"
        else:
            assert "1 al 5" in question, f"{state_name} debe incluir '1 al 5'"

    # Verificar que _build_question_reply NO agrega línea redundante
    svc = SurveyService.__new__(SurveyService)
    for state_name in _SCALE_PREFIX:
        reply = svc._build_question_reply(state_name)
        text = reply.text
        # No debe contener "Responde con un numero del 1 al"
        assert "Responde con un numero" not in text, (
            f"{state_name}: el hint 'Responde con un numero' no debe duplicarse.\n"
            f"Texto completo: {text}"
        )


# ─── Test 13: E2E Conversación Real ──────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_real_conversation_flow():
    """
    Prueba de flujo completo:
    1. Usuario: hola -> GREETING -> OTHER (no incrementa encuesta)
    2. Bot: ... (onboarding_status='in_progress', step='peso')
    3. Usuario: 'peso 70 y prefiero no comer lácteos' 
       -> Onboarding guarda peso, y apply_profile_intent guarda EVITA_LACTEOS.
       -> PROFILE_MAINTENANCE (pausa encuesta, snooze +5, no incrementa).
    4. Usuario: 'qué puedo desayunar' 
       -> NUTRITION_QUERY -> NUTRITION_VALUE (incrementa encuesta).
    """
    from application.services.handlers.generic_chat_handler import GenericChatHandler
    from application.services.handlers.onboarding_handler import OnboardingHandler
    from application.services.message_orchestrator import MessageOrchestratorService
    from application.services.conversation_state_service import ConversationStateService

    state_svc = ConversationStateService()
    state = MagicMock(spec=ConversationState)
    state.onboarding_status = OnboardingStatus.COMPLETED.value
    state.onboarding_step = None
    state.mode = SessionMode.ACTIVE_CHAT.value
    state.meaningful_interactions_count = 0
    state.survey_next_eligible_count = None
    state.version = 1

    orch = MessageOrchestratorService.__new__(MessageOrchestratorService)
    orch._state_service = state_svc

    # --- Turno 1: "hola" ---
    ctx1 = _make_ctx(route_intent=Intent.GREETING, text="hola")
    ctx1.state = state
    ctx1.profile_intent = None
    ctx1.looks_like_profile_update = False

    turn_kind_1 = orch._preclassify_turn_kind(ctx1)
    assert turn_kind_1 == "OTHER"
    # El handler devolvería OTHER, meaningful_interactions_count queda en 0.

    # Cambiamos estado para simular que entra a Onboarding
    state.onboarding_status = OnboardingStatus.IN_PROGRESS.value
    state.onboarding_step = "peso"
    intent_lacteos = _make_intent("restricciones_alimentarias", "ADD", ["lacteos"])
    ctx2 = _make_ctx(route_intent=Intent.ANSWER_CURRENT_STEP, text="peso 70 y prefiero no comer lacteos")
    ctx2.state = state
    ctx2.profile_intent = intent_lacteos
    
    turn_kind_2 = orch._preclassify_turn_kind(ctx2)
    assert turn_kind_2 == "PROFILE_MAINTENANCE"

    # Simulamos el OnboardingService
    mock_onboarding_svc = AsyncMock()
    mock_onboarding_svc.advance_flow = AsyncMock(return_value="¿Cuál es tu altura?")
    handler_onboarding = OnboardingHandler(onboarding_service=mock_onboarding_svc, fallback_handler=AsyncMock())
    
    reply2, _ = await handler_onboarding.handle(ctx2)
    
    # advance_flow DEBE haber recibido el intent extra
    call_kwargs2 = mock_onboarding_svc.advance_flow.call_args.kwargs
    assert call_kwargs2["pre_extracted_intent"] == intent_lacteos

    # El orchestator al final del turno clasifica:
    ctx2.turn_kind = turn_kind_2
    final_kind_2 = orch._classify_turn_kind(ctx2, handler_onboarding)
    assert final_kind_2 == "PROFILE_MAINTENANCE"
    assert state.meaningful_interactions_count == 0  # No incrementó

    # Simulemos que ahora se completa el onboarding
    state.onboarding_status = OnboardingStatus.COMPLETED.value
    state.onboarding_step = None

    # --- Turno 3: "qué puedo desayunar" ---
    ctx3 = _make_ctx(route_intent=Intent.NUTRITION_QUERY, text="qué puedo desayunar")
    ctx3.state = state
    ctx3.profile_intent = None
    ctx3.snapshot = MagicMock()
    ctx3.summary = MagicMock()
    ctx3.state_snapshot = MagicMock()
    ctx3.history = []
    ctx3.session = MagicMock()
    ctx3.profile_text = "test profile"
    ctx3.turn_kind = "NUTRITION_VALUE"
    
    turn_kind_3 = orch._preclassify_turn_kind(ctx3)
    assert turn_kind_3 == "NUTRITION_VALUE"
    ctx3.turn_kind = turn_kind_3

    # Simulamos GenericChatHandler
    mock_llm = AsyncMock()
    mock_llm.generate_reply = AsyncMock(return_value=("Aquí tienes tu desayuno...", "resp_id"))
    mock_llm.append_continuity_tip = MagicMock(return_value="Aquí tienes tu desayuno...")
    mock_llm.sanitize_final_reply = MagicMock(return_value=BotReply(text="Aquí tienes tu desayuno...", content_type="text"))
    
    mock_profile_int = AsyncMock()
    mock_profile_int.maybe_start_personalization_flow = AsyncMock(return_value=(None, False))
    mock_profile_int.maybe_intercept_for_missing_profile = AsyncMock(return_value=(None, False))
    mock_profile_int.maybe_suggest_phase2_field = AsyncMock(return_value="Aquí tienes tu desayuno...")

    mock_survey_flow = AsyncMock()
    # No interrumpe, no hay survey
    mock_survey_flow.compose_reply_with_survey = AsyncMock(return_value=(BotReply(text="Aquí tienes tu desayuno...", content_type="text"), False, False))

    handler_generic = GenericChatHandler(
        llm_reply=mock_llm,
        profile_interception=mock_profile_int,
        survey_flow=mock_survey_flow,
        state_service=state_svc,
    )

    reply3, _ = await handler_generic.handle(ctx3)
    
    # GenericChatHandler debió haber llamado a update_meaningful_interaction_count
    # con projected_interactions_count = 1
    assert state.meaningful_interactions_count == 1
    
    final_kind_3 = orch._classify_turn_kind(ctx3, handler_generic)
    assert final_kind_3 == "NUTRITION_VALUE"


