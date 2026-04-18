"""
Nutribot Backend â€” Dependency Injection Container
"""
from openai import AsyncOpenAI
from config import get_settings
from infrastructure.db.connection import get_session_factory
from infrastructure.db.conversation_repo import SqlAlchemyConversationRepository
from infrastructure.db.user_repo import SqlAlchemyUserRepository
from infrastructure.db.rag_repo import RagRepository
from infrastructure.evolution.client import EvolutionApiClient
from infrastructure.openai.embeddings_adapter import OpenAIEmbeddingsAdapter
from infrastructure.openai.media_service import DefaultMediaService
from infrastructure.openai.responses_adapter import OpenAIResponsesAdapter
from infrastructure.openai.stt_adapter import OpenAISpeechToTextAdapter
from infrastructure.openai.tts_adapter import OpenAITextToSpeechAdapter
from application.services.localization_service import LocalizationService
from application.services.nutrition_assessment_service import NutritionAssessmentService
from application.services.profile_extraction_service import ProfileExtractionService
from application.services.profile_context_service import ProfileContextService
from application.services.profile_interception_service import ProfileInterceptionService
from application.services.profile_read_service import ProfileReadService
from application.services.llm_reply_service import LlmReplyService
from application.services.survey_flow_service import SurveyFlowService
from application.services.survey_service import SurveyService
from application.services.onboarding_service import OnboardingService
from application.services.conversation_memory_service import ConversationMemoryService
from application.services.conversation_state_service import ConversationStateService
from application.services.turn_context_service import TurnContextService

from application.services.handlers.reset_handler import ResetHandler
from application.services.handlers.onboarding_handler import OnboardingHandler
from application.services.handlers.profile_update_handler import ProfileUpdateHandler
from application.services.handlers.generic_chat_handler import GenericChatHandler
from application.services.handlers.handler_registry import HandlerRegistry

from application.services.message_orchestrator import MessageOrchestratorService
from application.workers.inbox_worker import InboxWorker
from application.workers.outbox_worker import OutboxWorker
from application.workers.sweeper_worker import SweeperWorker

SYSTEM_INSTRUCTIONS = """Eres NutriBot 🍏, un asistente de orientacion nutricional REFERENCIAL de EsSalud.

IDENTIDAD:
- Presentate ("Hola, soy NutriBot 🍏...") SOLO la primera vez de la sesion o si te saludan directamente.
- En el resto de la conversacion, se directo, amable y no repitas tu presentacion.
- Habla en espanol peruano coloquial pero profesional.

POSICIONAMIENTO REFERENCIAL (OBLIGATORIO):
- Eres una herramienta de ORIENTACION NUTRICIONAL REFERENCIAL de EsSalud.
- NO reemplazas la consulta con un nutricionista profesional.
- NO haces manejo clinico ni planes alimenticios cerrados.
- Las recomendaciones personalizadas son siempre referencia.

QUE SI PUEDES HACER:
- Responder consultas cotidianas sobre alimentacion saludable.
- Dar tips generales (hidratacion, porciones, combinaciones de alimentos).
- Comentar fotos de comida con enfoque nutricional referencial.
- Calcular e interpretar IMC de forma referencial.

QUE NO PUEDES HACER (REGLAS ABSOLUTAS):
1. NUNCA respondas temas ajenos a nutricion/alimentacion.
2. NUNCA diagnostiques enfermedades ni condiciones medicas.
3. NUNCA recetes medicamentos ni dosis.
4. NUNCA des planes alimenticios clinicos cerrados.
5. Si piden manejo medico serio, deriva con calidez a EsSalud.

DATOS DEL USUARIO Y FORMATO:
- NUNCA uses formato LaTeX. WhatsApp no lo renderiza.
- Si el sistema envia [INSTRUCCION CRITICA DE FORMATO], debes respetarla literal al inicio de la respuesta.
- SOLO menciona alergias, enfermedades o restricciones cuando tengan valor real (no "Pendiente" ni "Ninguna").
- Si faltan peso o talla y piden menu completo, explica amablemente que primero necesitas esos datos.
- Regla de privacidad: no inventes ni expongas datos que el usuario no dio.

FORMATO WHATSAPP (OBLIGATORIO):
- Usa solo *texto* para enfasis.
- No uses **texto**, _texto_ ni ***texto***.

COHERENCIA MEDICA Y BIOLOGICA (MANDATORIO):
- No aceptes ni confirmes datos absurdos o inverosimiles.
- Si detectas incoherencia, pide aclaracion con calidez antes de recomendar.

LENGUAJE PERUANO (OBLIGATORIO):
- Prefiere terminos locales: camote, palta, choclo, papa, refrigerio, quinua, jugo.
- Evita sugerir alimentos poco accesibles en el contexto peruano.

TONO CALIDO (OBLIGATORIO):
- Estructura recomendada: apertura calida corta -> respuesta util -> cierre amable breve.
- Usa 1 o 2 emojis en respuestas normales.
- Usa 2 o 3 emojis solo en saludo, felicitacion o cierre positivo.
- No pongas emoji en cada oracion.
- Puedes usar frases cortas como: "Claro 😊", "Te ayudo con eso 🍏", "Vamos paso a paso 💪", "Que bueno 🎉".
- Manten respuestas breves, practicas y humanas (ideal: 3 a 5 oraciones)."""

class Container:
    def __init__(self):
        self.settings = get_settings()
        self.session_factory = get_session_factory()
        self.user_repo = SqlAlchemyUserRepository()
        self.conv_repo = SqlAlchemyConversationRepository()
        self.rag_repo = RagRepository()
        self.evolution_client = EvolutionApiClient()

        # Shared OpenAI Setup
        self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.openai_model = self.settings.openai_model
        self.embeddings_adapter = OpenAIEmbeddingsAdapter(
            client=self.openai_client,
            model=self.settings.openai_embedding_model,
        )
        self.stt_adapter = OpenAISpeechToTextAdapter(
            client=self.openai_client,
            model=self.settings.openai_stt_model,
        )
        self.media_service = DefaultMediaService(
            stt_service=self.stt_adapter,
            evolution_client=self.evolution_client,
        )
        self.tts_adapter = OpenAITextToSpeechAdapter(
            client=self.openai_client,
            model=self.settings.openai_tts_model,
            voice=self.settings.openai_tts_voice,
        )

        self.localization_service = LocalizationService()
        self.nutrition_assessment = NutritionAssessmentService()
        self.memory_service = ConversationMemoryService()
        self.state_service = ConversationStateService()
        
        # Application Services
        self.profile_extractor = ProfileExtractionService(
            openai_client=self.openai_client, 
            model=self.openai_model
        )
        self.profile_reader = ProfileReadService()
        self.profile_context = ProfileContextService(
            nutrition_assessment=self.nutrition_assessment,
        )
        
        self.survey_service = SurveyService(
            openai_client=self.openai_client, 
            openai_model=self.openai_model
        )
        
        self.onboarding_service = OnboardingService(
            openai_client=self.openai_client, 
            openai_model=self.openai_model,
            profile_extractor=self.profile_extractor,
            profile_reader=self.profile_reader,
            nutrition_assessment=self.nutrition_assessment,
            state_service=self.state_service,
        )

        self.llm_service = OpenAIResponsesAdapter(
            system_instructions=SYSTEM_INSTRUCTIONS,
            client=self.openai_client,
            model=self.openai_model,
        )
        self.llm_reply = LlmReplyService(
            llm_service=self.llm_service,
            system_instructions=SYSTEM_INSTRUCTIONS,
            profile_context=self.profile_context,
            localization_service=self.localization_service,
        )
        self.profile_interception = ProfileInterceptionService(
            onboarding_service=self.onboarding_service,
            profile_context=self.profile_context,
            state_service=self.state_service,
        )
        self.survey_flow = SurveyFlowService(
            survey_service=self.survey_service,
        )

        self.turn_context_service = TurnContextService(
            profile_reader=self.profile_reader,
            profile_context=self.profile_context,
            memory_service=self.memory_service,
        )
        
        self.generic_chat_handler = GenericChatHandler(
            llm_reply=self.llm_reply,
            profile_interception=self.profile_interception,
            survey_flow=self.survey_flow,
            state_service=self.state_service,
        )
        self.profile_update_handler = ProfileUpdateHandler(
            profile_extractor=self.profile_extractor,
            profile_context=self.profile_context,
            fallback_handler=self.generic_chat_handler,
        )
        self.onboarding_handler = OnboardingHandler(
            onboarding_service=self.onboarding_service,
            fallback_handler=self.generic_chat_handler,
        )
        self.reset_handler = ResetHandler(
            onboarding_service=self.onboarding_service,
            state_service=self.state_service,
        )
        
        self.handler_registry = HandlerRegistry(
            reset_handler=self.reset_handler,
            onboarding_handler=self.onboarding_handler,
            profile_update_handler=self.profile_update_handler,
            generic_chat_handler=self.generic_chat_handler,
        )

        self.message_orchestrator = MessageOrchestratorService(
            turn_context_service=self.turn_context_service,
            handler_registry=self.handler_registry,
            memory_service=self.memory_service,
            state_service=self.state_service,
        )

        # Workers
        self.inbox_worker = InboxWorker(
            session_factory=self.session_factory,
            user_repo=self.user_repo,
            conv_repo=self.conv_repo,
            media_service=self.media_service,
            embeddings=self.embeddings_adapter,
            rag_repo=self.rag_repo,
            evolution_client=self.evolution_client,
            orchestrator=self.message_orchestrator
        )

        self.outbox_worker = OutboxWorker(
            session_factory=self.session_factory,
            evolution_client=self.evolution_client,
            tts_adapter=self.tts_adapter
        )


        self.sweeper_worker = SweeperWorker(
            session_factory=self.session_factory
        )

_container_instance: Container | None = None


def get_container() -> Container:
    """Construye el contenedor una sola vez cuando realmente se necesita."""
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
    return _container_instance


class _LazyContainerProxy:
    """Proxy para mantener compatibilidad con `from di import container` sin eager init."""

    def __getattr__(self, item):
        return getattr(get_container(), item)


container = _LazyContainerProxy()

