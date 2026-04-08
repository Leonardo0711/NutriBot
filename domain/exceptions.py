"""
Nutribot Backend — Exceptions
Excepciones de dominio personalizadas.
"""


class ConcurrentStateUpdateError(Exception):
    """El estado de la conversación cambió mientras el LLM operaba en red.
    El mensaje debe ser reintentado con un snapshot fresco."""
    pass


class MessageNotProcessableError(Exception):
    """El payload del webhook no contiene un mensaje procesable."""
    pass
