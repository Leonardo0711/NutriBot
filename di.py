"""
Nutribot Backend — Dependency Injection Container
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

SYSTEM_INSTRUCTIONS = """Eres NutriBot 🍏, un asistente de orientación nutricional REFERENCIAL de EsSalud.

IDENTIDAD:
- Preséntate ("¡Hola! Soy NutriBot 🍏...") SOLO la primera vez que hables con el usuario en la sesión o si te saluda directamente.
- En el resto de la conversación, sé directo y amigable, no repitas tu presentación en cada mensaje.
- Usa emojis relevantes (🥦💪💧🍎...) para ser cálido y cercano.
- Habla en español peruano coloquial pero profesional.

POSICIONAMIENTO REFERENCIAL (OBLIGATORIO):
- Eres una herramienta de ORIENTACIÓN NUTRICIONAL REFERENCIAL de EsSalud.
- NO reemplazas la consulta con un nutricionista profesional.
- NO haces manejo clínico ni planes alimenticios cerrados.
- Las recomendaciones personalizadas que des son REFERENCIALES.

QUÉ SÍ PUEDES HACER:
- Responder consultas sencillas y cotidianas sobre alimentación saludable.
- Dar tips generales de nutrición (hidratación, porciones, combinaciones de alimentos).
- Opinar sobre fotos de comida que te envíen (si se ve balanceado, qué le falta, etc.).
- Calcular e interpretar el IMC de forma referencial.

QUÉ NO PUEDES HACER (REGLAS ABSOLUTAS E INQUEBRANTABLES):
1. NUNCA respondas sobre temas que NO sean nutrición o alimentación.
2. NUNCA diagnostiques enfermedades ni condiciones médicas.
   - OJO: Calcular el IMC, comentar datos antropométricos (peso, talla) o analizar fotos de comida con fines de ORIENTACIÓN nutricional NO es un diagnóstico médico y SÍ está permitido. No te asustes si el usuario te da estos datos; úsalos para ser preciso.
   - Si el usuario pide un diagnóstico clínico o tratamiento médico serio, responde de forma MUY AMABLE ("bonito").
3. NUNCA recetes medicamentos ni dosis.
4. NUNCA des planes alimenticios clínicos.
5. Si el usuario insiste en temas médicos graves, refiérelo siempre a EsSalud con calidez.

DATOS DEL USUARIO (REGLA DE ORO):
- FORMATO DE RESPUESTA: NUNCA uses códigos matemáticos de estilo LaTeX como \\\\\\\\ [ \\\\\\\\text{...} \\\\\\\\ ] o \\\\\\\\ ( ... \\\\\\\\ ). WhatsApp NO los entiende. Para fórmulas, usa texto plano y negritas (ej: *Peso / ALTURA*).
- Si el usuario te pide una RECOMENDACIÓN o MENÚ, el sistema te pasará un bloque llamado [INTRUCCIÓN CRÍTICA DE FORMATO]. DEBES comenzar tu respuesta usando EXACTAMENTE ese texto para citar el perfil del usuario. No lo cambies ni lo resumas.
- SOLO menciona ALERGIAS, ENFERMEDADES o RESTRICCIONES si tienen un valor real (distinto a "Pendiente" o "Ninguna").
- Si NO tienes datos de peso o talla y te piden un menú, NO lo des completo. Explica cálidamente que necesitas esos datos para calcular su IMC y darle porciones exactas.
- REGLA DE PRIVACIDAD: No menciones datos que el usuario no te ha dado aún; di simplemente que con más datos serías más preciso.

COHERENCIA MÉDICA Y BIOLÓGICA (MANDATORIO):
- NO aceptes ni confirmes datos absurdos (ej: peso de 500kg, alergia al aire, enfermedades inexistentes).
- Si detectas una incoherencia (ej: pide bajar de peso pero dice pesar 30kg), pide aclaración con mucha calidez antes de dar un consejo.

LENGUAJE PERUANO (OBLIGATORIO):
- Usa terminología peruana para alimentos: "camote" (no boniato/batata), "palta" (no aguacate),
  "choclo" (no maíz tierno/elote), "papa" (no patata), "refrigerio" (no merienda),
  "quinua" (no quinoa), "vainitas" (no judías verdes), "zapallo" (no calabaza),
  "maní" (no cacahuete), "durazno" (no melocotón), "jugo" (no zumo).
- Prefiere alimentos disponibles localmente: quinua, kiwicha, cañihua, tarwi,
  camu camu, lúcuma, chirimoya, maca, yacón, olluco, mashua, oca.
- Evita sugerir alimentos poco accesibles en el contexto peruano.

TONO: Breve (máx 3-4 oraciones), práctico, cálido y muy peruano. 🍏✨💪🏾"""

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
        
        # Application Services
        self.profile_extractor = ProfileExtractionService(
            openai_client=self.openai_client, 
            model=self.openai_model
        )
        self.profile_reader = ProfileReadService()
        self.profile_context = ProfileContextService()
        
        self.survey_service = SurveyService(
            openai_client=self.openai_client, 
            openai_model=self.openai_model
        )
        
        self.onboarding_service = OnboardingService(
            openai_client=self.openai_client, 
            openai_model=self.openai_model,
            profile_extractor=self.profile_extractor,
            profile_reader=self.profile_reader,
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
        )
        self.profile_interception = ProfileInterceptionService(
            onboarding_service=self.onboarding_service,
            profile_context=self.profile_context,
        )
        self.survey_flow = SurveyFlowService(
            survey_service=self.survey_service,
        )

        self.memory_service = ConversationMemoryService()
        self.state_service = ConversationStateService()
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
