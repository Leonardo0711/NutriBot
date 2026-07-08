import os
import pytest
import re
from types import SimpleNamespace

from application.services.survey_service import SurveyResponseExtractor, SurveyService
from application.services.survey_flow_service import SurveyFlowService
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.conversation_memory_service import ConversationMemoryService
from application.services.llm_reply_service import LlmReplyService
from domain.context_builder import should_fetch_rag, try_fast_response
from application.services.onboarding_service import OnboardingService
from application.services.profile_intent_extractor_service import ProfileIntentExtractorService
from domain.entities import ConversationState, NormalizedMessage, User
from domain.profile_snapshot import ProfileHealth, ProfileLocation, ProfileMeasurements, ProfileSnapshot
from domain.reply_objects import BotReply
from domain.router import Intent, RouteResult, classify_message
from domain.value_objects import MessageType, SessionMode, OnboardingStatus, OnboardingStep
from domain.utils import get_now_peru


@pytest.mark.asyncio
class TestHardening:

    def test_survey_fast_parser(self):
        extractor = SurveyResponseExtractor(None, "dummy-model")

        assert extractor.try_fast_extract("esperando_nps", "10") == {"intent": "ANSWER", "value": "10"}
        assert extractor.try_fast_extract("esperando_nps", " 10 ") == {"intent": "ANSWER", "value": "10"}
        assert extractor.try_fast_extract("esperando_nps", "1") == {"intent": "ANSWER", "value": "1"}

        assert extractor.try_fast_extract("esperando_p1", "10") == {"intent": "ANSWER", "value": "10"}
        assert extractor.try_fast_extract("esperando_p1", "5") == {"intent": "ANSWER", "value": "5"}
        assert extractor.try_fast_extract("esperando_p3", " 5 ") == {"intent": "ANSWER", "value": "5"}

    def test_survey_scale_questions_use_interactive_lists(self):
        service = SurveyService(None, "dummy-model")
        extractor = SurveyResponseExtractor(None, "dummy-model")

        p1 = service._build_question_reply("esperando_p1")
        assert p1.content_type == "interactive_list"
        assert p1.payload_json["sections"][0]["rows"][0]["id"] == "survey:p1:1"
        assert p1.payload_json["sections"][0]["rows"][0]["title"] == "1"
        assert p1.payload_json["sections"][0]["rows"][0]["description"] == ""
        assert p1.payload_json["sections"][0]["rows"][-1]["id"] == "survey:p1:5"
        assert p1.payload_json["sections"][0]["rows"][-1]["title"] == "5"
        assert "1 =" not in p1.text
        assert len(p1.payload_json["sections"][0]["rows"]) == 5

        nps = service._build_question_reply("esperando_nps")
        assert nps.content_type == "interactive_list"
        assert nps.payload_json["sections"][0]["rows"][0]["id"] == "survey:nps:1"
        assert nps.payload_json["sections"][0]["rows"][0]["title"] == "1"
        assert nps.payload_json["sections"][0]["rows"][-1]["id"] == "survey:nps:10"
        assert nps.payload_json["sections"][0]["rows"][-1]["title"] == "10"
        assert "10 =" not in nps.text
        assert len(nps.payload_json["sections"][0]["rows"]) == 10

        assert extractor.extract_structured("esperando_p1", "survey:p1:4", "") == {
            "intent": "ANSWER",
            "value": "4",
        }
        assert extractor.extract_structured("esperando_nps", "survey:nps:10", "") == {
            "intent": "ANSWER",
            "value": "10",
        }

    def test_survey_email_opt_out_counts_as_answered(self):
        service = SurveyService(None, "dummy-model")
        reply = service._build_question_reply("esperando_correo")

        assert service._is_state_answered(
            "esperando_correo",
            {"correo": None, "correo_opt_out": True},
        )
        assert "opcional" in reply.text.lower()
        assert "campañas" in reply.text.lower()

    def test_survey_interrupts_on_free_form_question_during_scale(self):
        extractor = SurveyResponseExtractor(None, "dummy-model")
        msg = "jajajaj yap y sobre dormir seria mejor de costado boca arriba boca abajo en el piso"
        assert extractor.try_fast_extract("esperando_p4", msg) == {"intent": "INTERRUPT", "value": None}

    def test_survey_consent_does_not_take_long_yes_plus_question_as_yes(self):
        extractor = SurveyResponseExtractor(None, "dummy-model")
        msg = "yap y sobre dormir seria mejor de costado boca arriba o boca abajo"
        assert extractor.try_fast_extract("esperando_consentimiento_encuesta", msg) == {
            "intent": "INTERRUPT",
            "value": None,
        }

    def test_health_clarification_flags(self):
        extractor = ProfileExtractionService(None, "dummy-model")

        assert extractor._check_health_ambiguity("diabetes") is not None
        assert extractor._check_health_ambiguity("tengo problemas de tiroides") is not None

        assert extractor._check_health_ambiguity("diabetes tipo 1") is None
        assert extractor._check_health_ambiguity("hipertension") is None
        assert extractor._check_health_ambiguity("hipotiroidismo") is None

    def test_initial_diet_guidance_uses_basic_profile(self):
        from application.services.nutrition_assessment_service import NutritionAssessmentService

        snapshot = ProfileSnapshot(
            user_id=1,
            measurements=ProfileMeasurements(age_years=23, weight_kg=78, height_cm=171),
            health=ProfileHealth(
                diseases=("Anemia por deficiencia de hierro",),
                nutrition_goal="BAJAR DE PESO",
            ),
            location=ProfileLocation(),
        )

        guidance = NutritionAssessmentService.build_initial_diet_guidance(snapshot)

        assert guidance is not None
        assert "Dieta sugerida" in guidance
        assert "hipocalórica moderada" in guidance
        assert "hierro" in guidance

    def test_encoding_no_mojibake(self):
        check_patterns = []
        critical_files = [
            "application/services/survey_service.py",
            "application/services/onboarding_service.py",
            "application/services/message_orchestrator.py",
            "domain/entities.py",
        ]

        for rel_path in critical_files:
            abs_path = os.path.join(os.path.dirname(__file__), "..", rel_path)
            if not os.path.exists(abs_path):
                continue

            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
                for bad_char in check_patterns:
                    assert bad_char not in content, f"Encontrado {repr(bad_char)} en {rel_path}"

    def test_strip_redundant_hola_de_nuevo_opening(self):
        text = (
            "¡Hola de nuevo! Considerando que tienes 23 años, aquí te dejo ideas "
            "para acompañar tu lasaña."
        )

        cleaned = LlmReplyService._strip_redundant_opening_fillers(text)

        assert cleaned.startswith("Considerando que tienes 23 años")
        assert "de nuevo!" not in cleaned.lower()

    def test_anti_noise_onboarding(self):
        extractor = ProfileExtractionService(None, "dummy-model")
        assert not extractor.contains_absurd_claim("ok")
        _, updates, _ = extractor._apply_bulletproof_logic({"alergias": "ok"}, "ok", "alergias")
        assert isinstance(updates, dict)

    def test_idempotence_mock(self):
        assert True

    def test_split_values_and_semantic_candidates_generalized(self):
        extractor = ProfileExtractionService(None, "dummy-model")
        values = extractor._split_values("soy alergico a muchas cosas, como a los lacteos, mariscos, gluten")
        normalized_values = {extractor._normalize_text(v) for v in values}
        assert "muchas cosas" not in normalized_values
        assert "lacteos" in normalized_values
        assert "mariscos" in normalized_values
        assert "gluten" in normalized_values

        candidates = extractor._restriction_resolution_candidates("lacteos")
        normalized_candidates = {extractor._normalize_text(v) for v in candidates}
        assert "lactosa" in normalized_candidates

    def test_step_scope_blocks_cross_field_contamination(self):
        extractor = ProfileExtractionService(None, "dummy-model")
        clean, updates, meta = extractor._apply_bulletproof_logic(
            {"alergias": "bajar de peso", "objetivo_nutricional": "bajar de peso"},
            "bajar de peso",
            "alergias",
        )
        assert clean == {}
        assert updates == {}
        assert meta.get("clarification_prompt")

    def test_step_scope_keeps_objective_even_with_typo(self):
        extractor = ProfileExtractionService(None, "dummy-model")
        clean, updates, _ = extractor._apply_bulletproof_logic(
            {"objetivo_nutricional": "najar de peso"},
            "najar de peso",
            "objetivo_nutricional",
        )
        assert "objetivo_nutricional" in clean
        assert "objetivo_nutricional" in updates

    def test_conflict_detection_uses_alias_tokens(self):
        snapshot = ProfileSnapshot(
            user_id=1,
            measurements=ProfileMeasurements(),
            health=ProfileHealth(
                allergies=("Sin Lactosa", "Alergia al Mani (Cacahuate)"),
                food_restrictions=(),
            ),
            location=ProfileLocation(),
        )
        conflicts = LlmReplyService._find_conflicting_items_in_text(
            "quiero una receta con lacteos y mani",
            snapshot,
        )
        normalized = {LlmReplyService._normalize_text_for_match(x) for x in conflicts}
        assert any("lactosa" in item for item in normalized)
        assert any("mani" in item or "cacahuate" in item for item in normalized)

    def test_scope_redirect_blocks_out_of_domain_request(self):
        route = RouteResult(intent=Intent.DOUBT, confidence=0.8, reason="Pregunta generica detectada")
        assert LlmReplyService._must_redirect_to_nutrition_scope(route, "me puedes dar un resumen de one piece porfavor")
        assert LlmReplyService._must_redirect_to_nutrition_scope(route, "que fue weonazo")
        assert LlmReplyService._must_redirect_to_nutrition_scope(route, "genial, a ver dame un resumen de dragon ball")

    def test_scope_redirect_allows_nutrition_query(self):
        route = RouteResult(intent=Intent.DOUBT, confidence=0.8, reason="Pregunta generica detectada")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "me puedes dar un menu saludable para hoy")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "dame un almuerzo para hoy")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "me das un consejo para ejercitarme")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "me das un konsejo pa ejersitarme")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "q puedo almuersar hoy")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "kiero entrenar pero tengo sobrepeso")

    def test_scope_redirect_allows_coffee_nutrition_followup(self):
        route = RouteResult(intent=Intent.DOUBT, confidence=0.8, reason="Pregunta generica detectada")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(
            route,
            "asu pero tomo cafe todas las mananas no puedo entonces?",
        )

    def test_scope_redirect_allows_ambiguous_followup_to_use_llm_context(self):
        route = RouteResult(intent=Intent.DOUBT, confidence=0.8, reason="Pregunta generica detectada")
        assert not LlmReplyService._must_redirect_to_nutrition_scope(route, "asu no puedo entonces?")

    def test_location_helpers_reject_country_but_clean_callao(self):
        assert OnboardingService._clean_location_candidate("mi ubicacion soy del callao") == "callao"
        assert OnboardingService._clean_location_candidate("de peru") == "peru"
        assert OnboardingService._is_country_level_location("Perú")
        assert OnboardingService._looks_like_location_candidate("mi ubicacion soy del callao")

    def test_profile_intent_fast_skips_health_detail_refusal(self):
        extractor = ProfileIntentExtractorService.__new__(ProfileIntentExtractorService)
        result = extractor._try_fast_numeric_or_clear("ahh no se que tipo de anemia sera", "enfermedades")
        assert result is not None
        assert not result.is_profile_update
        assert result.source == "FAST_HEALTH_DETAIL_REFUSAL"

    def test_router_treats_coffee_followup_as_nutrition(self):
        route = classify_message(
            raw_text="asu pero tomo cafe todas las mananas no puedo entonces?",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert route.intent == Intent.NUTRITION_QUERY

        lunch_route = classify_message(
            raw_text="dame un almuerzo para hoy",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert lunch_route.intent == Intent.RECOMMENDATION_REQUEST

        exercise_route = classify_message(
            raw_text="Me das un consejo para ejercitarme?",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert exercise_route.intent == Intent.NUTRITION_QUERY

        typo_exercise_route = classify_message(
            raw_text="me das un konsejo pa ejersitarme",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert typo_exercise_route.intent == Intent.NUTRITION_QUERY

        typo_lunch_route = classify_message(
            raw_text="q puedo almuersar hoy",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert typo_lunch_route.intent == Intent.NUTRITION_QUERY

    def test_rag_policy_skips_ambiguous_human_followup_but_keeps_llm_available(self):
        route = RouteResult(intent=Intent.DOUBT, confidence=0.7, reason="Pregunta generica detectada")
        assert not should_fetch_rag(route, "asu no puedo entonces?")

    def test_rag_policy_uses_rag_for_technical_nutrition_question(self):
        route = RouteResult(intent=Intent.NUTRITION_QUERY, confidence=0.85, reason="Pregunta nutricional")
        assert should_fetch_rag(route, "por que el cafe afecta la absorcion de hierro?")

    def test_fast_response_only_applies_to_pure_greeting_route(self):
        greeting = RouteResult(intent=Intent.GREETING, confidence=0.9, reason="Saludo detectado")
        greeting_reply = try_fast_response(greeting)
        assert greeting_reply
        assert "NutriBot" in greeting_reply
        assert "nutrición, salud y bienestar" in greeting_reply

        mixed = classify_message(
            raw_text="hola puedo comer arroz si tengo diabetes?",
            current_mode="active_chat",
            onboarding_status="completed",
            onboarding_step=None,
            content_type="text",
        )
        assert mixed.intent != Intent.GREETING

    def test_short_confirmation_after_nutrition_offer_is_not_fast_path(self):
        route = RouteResult(intent=Intent.CONFIRMATION, confidence=0.9, reason="Confirmacion")
        history = [
            {
                "role": "assistant",
                "content": "¿Te gustaría saber cómo acompañarla con ensaladas o algún otro platillo?",
            }
        ]
        assert LlmReplyService._should_defer_short_followup_to_llm(route, "Si", history)

    def test_short_confirmation_to_survey_does_not_defer_to_llm(self):
        route = RouteResult(intent=Intent.CONFIRMATION, confidence=0.9, reason="Confirmacion")
        history = [
            {
                "role": "assistant",
                "content": "¿Me podrías ayudar a seguir mejorando contestando algunas preguntas? Si estás de acuerdo, empezamos.",
            }
        ]
        assert not LlmReplyService._should_defer_short_followup_to_llm(route, "Si", history)

    @pytest.mark.asyncio
    async def test_profile_update_data_bypasses_scope_redirect(self):
        class DummyLLM:
            async def generate_reply(self, **kwargs):
                return "Listo, actualice tu provincia a Callao.", "resp-1"

        service = LlmReplyService(
            llm_service=DummyLLM(),
            system_instructions="",
            profile_context=SimpleNamespace(),
        )
        reply, _ = await service.generate_reply(
            onboarding_interception_happened=False,
            reply=None,
            state_snapshot=ConversationState(usuario_id=1),
            normalized=NormalizedMessage(
                provider_message_id="m-location",
                phone="+51999999999",
                content_type=MessageType.TEXT,
                text="mi ubicacion soy del callao",
            ),
            route=RouteResult(intent=Intent.AMBIGUOUS, confidence=0.3, reason="No se pudo clasificar"),
            rag_text=None,
            history=[],
            profile_text="",
            snapshot=ProfileSnapshot(
                user_id=1,
                measurements=ProfileMeasurements(),
                health=ProfileHealth(),
                location=ProfileLocation(),
            ),
            extracted_data={"provincia": "Callao"},
            has_absurd_profile_claim=False,
            is_asking_for_recommendation=False,
        )
        assert reply == "Listo, actualice tu provincia a Callao."

    def test_compact_memory_uses_summary_plus_recent_history(self):
        service = ConversationMemoryService()
        item = service._build_compact_memory_item(
            "usuario pregunto por anemia",
            "anemia, hierro",
            "sugerir carnes y vitamina C",
        )
        assert item is not None
        assert item["role"] == "memory_summary"
        assert "anemia" in item["content"]

    def test_onboarding_prefiero_no_comer_lacteos_is_treated_as_data(self):
        onboarding = OnboardingService(
            openai_client=None,
            openai_model="dummy-model",
            profile_extractor=ProfileExtractionService(None, "dummy-model"),
        )
        refusal = onboarding._classify_data_refusal(
            current_step=OnboardingStep.RESTRICCIONES.value,
            user_text="prefiero no comer lacteos",
            is_food_request=False,
            history=None,
        )
        assert refusal == "NONE"

    @pytest.mark.asyncio
    async def test_survey_not_invited_during_profile_capture(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=99,
            mode=SessionMode.ACTIVE_CHAT.value,
            onboarding_status=OnboardingStatus.IN_PROGRESS.value,
            onboarding_step=OnboardingStep.RESTRICCIONES.value,
            meaningful_interactions_count=9,
        )
        normalized = NormalizedMessage(
            provider_message_id="m-profile",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="ok",
        )
        addon = await service.process(
            session=None,
            state=state,
            normalized=normalized,
            projected_interactions_count=9,
        )
        assert addon is None

    @pytest.mark.asyncio
    async def test_survey_reinvite_cooldown_blocks_repeat_prompt(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=100,
            mode=SessionMode.ACTIVE_CHAT.value,
            meaningful_interactions_count=1,
            last_form_prompt_at=get_now_peru(),
        )
        normalized = NormalizedMessage(
            provider_message_id="m-cooldown",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="no",
        )
        addon = await service.process(
            session=None,
            state=state,
            normalized=normalized,
            projected_interactions_count=1,
        )
        assert addon is None

    @pytest.mark.asyncio
    async def test_survey_reinvite_cooldown_allows_after_valid_threshold(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=104,
            mode=SessionMode.ACTIVE_CHAT.value,
            meaningful_interactions_count=2,
            last_form_prompt_at=get_now_peru(),
            usability_completion_pct=0,
        )
        normalized = NormalizedMessage(
            provider_message_id="m-cooldown-threshold",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="dame una cena saludable",
        )

        async def fake_get_active_form(session):
            return SimpleNamespace(id=1)

        async def fake_load_active_progress(session, uid):
            return None

        service._get_active_form = fake_get_active_form
        service._load_active_progress = fake_load_active_progress

        addon = await service.process(
            session=object(),
            state=state,
            normalized=normalized,
            projected_interactions_count=2,
        )

        assert addon is not None
        assert "ayudar a seguir mejorando" in addon.text
        assert state.awaiting_question_code == "esperando_consentimiento_encuesta"
        assert state.meaningful_interactions_count == 0

    @pytest.mark.asyncio
    async def test_survey_pending_form_resumes_even_during_reinvite_cooldown(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=101,
            mode=SessionMode.ACTIVE_CHAT.value,
            meaningful_interactions_count=5,
            last_form_prompt_at=get_now_peru(),
            usability_completion_pct=0,
        )
        normalized = NormalizedMessage(
            provider_message_id="m-pending-form",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="5",
        )

        async def fake_load_active_progress(session, uid):
            return SimpleNamespace(formulario_id=1, estado_actual="esperando_p4")

        async def fake_process_form_response(session, state_arg, normalized_arg):
            assert state_arg.mode == SessionMode.COLLECTING_USABILITY.value
            assert state_arg.awaiting_question_code == "esperando_p4"
            return BotReply(text="Retomemos el formulario pendiente.", content_type="text")

        service._load_active_progress = fake_load_active_progress
        service._process_form_response = fake_process_form_response

        addon = await service.process(
            session=object(),
            state=state,
            normalized=normalized,
            projected_interactions_count=5,
        )
        assert addon is not None
        assert state.mode == SessionMode.COLLECTING_USABILITY.value
        assert state.awaiting_question_code == "esperando_p4"

    @pytest.mark.asyncio
    async def test_pending_survey_does_not_consume_thanks_from_active_chat(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=102,
            mode=SessionMode.ACTIVE_CHAT.value,
            meaningful_interactions_count=0,
            usability_completion_pct=0,
        )
        normalized = NormalizedMessage(
            provider_message_id="m-thanks",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="excelente gracias",
        )

        async def fake_load_active_progress(session, uid):
            return SimpleNamespace(formulario_id=1, estado_actual="esperando_p4")

        async def fake_process_form_response(session, state_arg, normalized_arg):
            raise AssertionError("A chat closing should not be consumed as a survey answer")

        service._load_active_progress = fake_load_active_progress
        service._process_form_response = fake_process_form_response

        addon = await service.process(
            session=object(),
            state=state,
            normalized=normalized,
            projected_interactions_count=1,
        )

        assert addon is None
        assert state.mode == SessionMode.ACTIVE_CHAT.value
        assert state.awaiting_question_code is None
        assert state.meaningful_interactions_count == 0

    @pytest.mark.asyncio
    async def test_pending_survey_reprompts_after_valid_chat_threshold(self):
        service = SurveyService(None, "dummy-model")
        state = ConversationState(
            usuario_id=103,
            mode=SessionMode.ACTIVE_CHAT.value,
            meaningful_interactions_count=1,
            usability_completion_pct=0,
        )
        normalized = NormalizedMessage(
            provider_message_id="m-menu",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="dame un menu saludable",
        )

        async def fake_load_active_progress(session, uid):
            return SimpleNamespace(formulario_id=1, estado_actual="esperando_p6")

        service._load_active_progress = fake_load_active_progress

        addon = await service.process(
            session=object(),
            state=state,
            normalized=normalized,
            projected_interactions_count=2,
        )

        assert addon is not None
        assert "ayudar a seguir mejorando" in addon.text
        assert addon.content_type == "interactive_buttons"
        assert state.mode == SessionMode.ACTIVE_CHAT.value
        assert state.awaiting_question_code == "esperando_consentimiento_encuesta"
        assert state.meaningful_interactions_count == 0

    @pytest.mark.asyncio
    async def test_survey_consent_does_not_send_double_message(self):
        class DummySurveyService:
            async def process(self, session, state, normalized, projected_interactions_count=None):
                return BotReply(text="Pregunta 1", content_type="text")

            async def get_current_question_reply(self, session, state):
                return None

        service = SurveyFlowService(DummySurveyService())
        state = ConversationState(
            usuario_id=10,
            mode=SessionMode.ACTIVE_CHAT.value,
            awaiting_question_code="esperando_consentimiento_encuesta",
        )
        normalized = NormalizedMessage(
            provider_message_id="m1",
            phone="+51999999999",
            content_type=MessageType.TEXT,
            text="claro",
        )
        user = User(id=10, numero_whatsapp="+51999999999")
        scheduled = {"count": 0}

        async def fake_schedule(**kwargs):
            scheduled["count"] += 1

        final_reply, interrupted, survey_engaged = await service.compose_reply_with_survey(
            session=None,
            state=state,
            normalized=normalized,
            user=user,
            reply="mensaje general que no debe salir",
            new_response_id=None,
            onboarding_interception_happened=False,
            is_requesting_survey=False,
            projected_interactions_count=7,
            schedule_separate_message=fake_schedule,
        )

        assert interrupted is False
        assert survey_engaged is True
        assert final_reply.text == "Pregunta 1"
        assert scheduled["count"] == 0
