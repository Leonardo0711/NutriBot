"""
Nutribot Backend — Domain Value Objects
Enums y tipos inmutables que representan conceptos del dominio.
"""
from enum import Enum


class MessageType(str, Enum):
    """Tipos de mensaje que llegan desde WhatsApp vía Evolution API."""
    TEXT = "textMessage"
    AUDIO = "audioMessage"
    PTT = "pttMessage"
    IMAGE = "imageMessage"

    @property
    def is_voice(self) -> bool:
        """True si el mensaje es audio o nota de voz (PTT)."""
        return self in (MessageType.AUDIO, MessageType.PTT)

    @property
    def is_media(self) -> bool:
        """True si el mensaje requiere descarga de archivo."""
        return self in (MessageType.AUDIO, MessageType.PTT, MessageType.IMAGE)


class SessionMode(str, Enum):
    """Modos de la máquina de estado conversacional."""
    ACTIVE_CHAT = "active_chat"
    COLLECTING_USABILITY = "collecting_usability"



class QuestionCode(str, Enum):
    """Preguntas del cuestionario de usabilidad."""
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    P5 = "p5"
    P6 = "p6"
    P7 = "p7"
    P8 = "p8"
    P9 = "p9"
    P10 = "p10"
    NPS = "nps"
    COMENTARIO = "comentario"


class OnboardingStatus(str, Enum):
    """Estado del proceso de onboarding nutricional."""
    NOT_STARTED = "not_started"
    INVITED = "invited"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    PAUSED = "paused"


class OnboardingStep(str, Enum):
    """Paso actual del flujo de onboarding."""
    INVITACION = "invitacion_inicial"
    EDAD = "edad"
    ALERGIAS = "alergias"
    ENFERMEDADES = "enfermedades"
    TIPO_DIETA = "tipo_dieta"
    OBJETIVO = "objetivo_nutricional"
    PESO = "peso_kg"
    ALTURA = "altura_cm"
    REGION = "region"
    PROVINCIA = "provincia"
    DISTRITO = "distrito"
    RESTRICCIONES = "restricciones_alimentarias"


ONBOARDING_PHASE_1 = [
    OnboardingStep.INVITACION,
    OnboardingStep.EDAD,
    OnboardingStep.PESO,
    OnboardingStep.ALTURA,
    OnboardingStep.ALERGIAS,
    OnboardingStep.OBJETIVO,
]

ONBOARDING_PHASE_2 = [
    OnboardingStep.TIPO_DIETA,
    OnboardingStep.ENFERMEDADES,
    OnboardingStep.RESTRICCIONES,
    OnboardingStep.PROVINCIA,
    OnboardingStep.DISTRITO,
]

# Backward-compat: orden completo para búsquedas exhaustivas
ONBOARDING_STEPS_ORDER = ONBOARDING_PHASE_1 + ONBOARDING_PHASE_2

class ExtractionStatus(str, Enum):
    """Estado de confianza de una extracción de perfil."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class JobStatus(str, Enum):
    """Estado genérico para colas de trabajo (inbox, outbox, extraction)."""
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class OutgoingContentType(str, Enum):
    """Tipo de contenido de un mensaje saliente."""
    TEXT = "text"
    AUDIO = "audio"
    AUDIO_TTS = "audio_tts"
    INTERACTIVE_BUTTONS = "interactive_buttons"
    INTERACTIVE_LIST = "interactive_list"


class OutgoingStatus(str, Enum):
    """Estado de entrega de un mensaje saliente."""
    PENDING = "pending"
    PROCESSING = "processing"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
