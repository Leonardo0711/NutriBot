# application/services/profile_intent_extractor_service.py
"""
Nutribot Backend — ProfileIntentExtractorService
=================================================
Cerebro semántico que entiende la INTENCIÓN real del mensaje.
No usa listas cerradas — delega la comprensión al LLM para mensajes complejos.

Fast paths inteligentes SOLO para lo trivialmente obvio (números directos,
"ninguna" literal). Todo lo demás pasa por comprensión real.

Flujo:
1. Fast-path numérico / negación literal (sin LLM)
2. LLM estructurado con JSON mode (chat.completions.create)
3. Resolución semántica de entidades (SemanticEntityResolver)
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from domain.profile_intent import ProfileIntentResult, ProfileIntentValue
from domain.parsers import parse_age, parse_weight, parse_height
from application.services.semantic_entity_resolver import SemanticEntityResolver

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Eres un clasificador de intención para un chatbot nutricional peruano.
Tu ÚNICA tarea es analizar si el mensaje del usuario contiene una actualización
de perfil nutricional y extraer la información estructurada.

Campos válidos de perfil: edad, peso_kg, altura_cm, alergias, enfermedades,
restricciones_alimentarias, tipo_dieta, objetivo_nutricional, provincia, distrito.

Operaciones válidas:
- ADD: agregar un elemento a una lista (alergias, enfermedades, restricciones)
- REMOVE: quitar un elemento de una lista ("ya no soy intolerante", "quítame", "ya no")
- REPLACE: reemplazar el valor actual ("ahora mi objetivo es otro")
- CLEAR: vaciar un campo ("no tengo alergias", "ninguna enfermedad")
- CORRECTION: corregir un dato anterior ("me equivoqué, peso 68", "perdón, era 70")
- HISTORICAL_UPDATE: actualización temporal natural ("ahora peso 70", "he bajado a 65")
- NOOP: no es una actualización de perfil

Reglas importantes:
- "prefiero no comer lácteos" → ADD restricciones_alimentarias, value="lacteos"
- "ya puedo comer lácteos" → REMOVE restricciones_alimentarias, value="lacteos"
- "no tengo alergias" → CLEAR alergias (NO es ADD de "no tengo alergias")
- "soy diabético" → ADD enfermedades, value="diabetes"
- "ya no tengo diabetes" → REMOVE enfermedades, value="diabetes"
- "quiero bajar de peso" → REPLACE objetivo_nutricional, value="bajar de peso"
- "soy vegetariano" → REPLACE tipo_dieta, value="vegetariana"
- "me equivoqué, peso 68" → CORRECTION peso_kg, value=68
- "ahora peso 70" → HISTORICAL_UPDATE peso_kg, value=70
- "¿qué recetas me recomiendas?" → NOOP (pregunta nutricional, NO perfil)
- "dame un menú saludable" → NOOP (solicitud de contenido, NO perfil)
- "hola, buenos días" → NOOP

Si el usuario menciona una consulta nutricional (receta, menú, qué comer, etc.)
SIN intención de modificar su perfil, responde NOOP.

Responde SOLO un JSON válido con esta estructura:
{
  "is_profile_update": true/false,
  "field_code": "campo o null",
  "operation": "ADD/REMOVE/REPLACE/CLEAR/CORRECTION/HISTORICAL_UPDATE/NOOP",
  "values": ["valor1", "valor2"],
  "confidence": 0.0-1.0,
  "needs_clarification": true/false,
  "clarification_question": "pregunta si aplica o null"
}
"""


class ProfileIntentExtractorService:
    """
    Extrae la intención de actualización de perfil de un mensaje.
    No depende de listas cerradas — comprensión real vía LLM.
    """

    NUMERIC_FIELDS = {"edad", "peso_kg", "altura_cm", "peso", "altura"}
    LIST_FIELDS = {"alergias", "enfermedades", "restricciones_alimentarias"}
    LITERAL_CLEAR_NORMALIZED = frozenset({
        "ninguna", "ninguno", "nada", "no tengo",
        "no padezco", "no sufro", "sin alergias",
        "sin enfermedades", "sin restricciones",
    })

    def __init__(
        self,
        semantic_resolver: SemanticEntityResolver,
        openai_client: AsyncOpenAI | None = None,
        model: str | None = None,
    ):
        self._semantic_resolver = semantic_resolver
        settings = get_settings()
        self._client = openai_client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or "gpt-4o-mini"

    async def extract(
        self,
        *,
        user_text: str,
        session: AsyncSession,
        expected_field: str | None = None,
        usuario_id: int | None = None,
        router_intent: str | None = None,
        router_confidence: float = 0.0,
    ) -> ProfileIntentResult:
        """
        Punto de entrada principal.
        expected_field es el paso de onboarding actual (si aplica).
        """
        text_clean = (user_text or "").strip()
        if not text_clean:
            return ProfileIntentResult()

        # ── Fast paths inteligentes (sin LLM) ──
        # Solo para lo TRIVIALMENTE obvio: número directo a campo esperado

        if expected_field:
            fast = self._try_fast_numeric_or_clear(text_clean, expected_field)
            if fast:
                if fast.has_values:
                    await self._resolve_values(session, fast, usuario_id)
                return fast

            # Fast-path para sexo: respuestas obvias sin LLM
            fast_sexo = self._try_fast_sexo(text_clean, expected_field)
            if fast_sexo:
                return fast_sexo

        # ── Guard: si el router ya clasificó con alta confianza algo que
        #    NO es perfil (saludo, reset, encuesta, confirmación), no gastar tokens ──
        skip_intents = {
            "GREETING", "RESET", "SURVEY_CONTINUE", "CONFIRMATION",
            "DENIAL", "SKIP", "IMAGE", "AUDIO",
        }
        if router_intent in skip_intents and router_confidence >= 0.85:
            return ProfileIntentResult()

        # ── LLM: comprensión real de la intención ──
        result = await self._extract_with_llm(text_clean, expected_field)

        # ── Resolver entidades semánticamente si es actualización de perfil ──
        if result.is_profile_update and result.has_values:
            await self._resolve_values(session, result, usuario_id)

        return result

    def _try_fast_numeric_or_clear(
        self, text: str, expected_field: str
    ) -> ProfileIntentResult | None:
        """
        Fast path SIN LLM para respuestas triviales durante onboarding:
        - Un número puro cuando se espera edad/peso/altura
        - "ninguna" literal cuando se espera lista
        """
        canonical = self._canonical_field(expected_field)

        # Número directo a campo numérico esperado
        if canonical in self.NUMERIC_FIELDS:
            value = None
            if canonical in ("edad",):
                value = parse_age(text)
            elif canonical in ("peso_kg", "peso"):
                value = parse_weight(text)
            elif canonical in ("altura_cm", "altura"):
                value = parse_height(text)

            if value is not None:
                real_field = self._canonical_field(canonical)
                return ProfileIntentResult(
                    is_profile_update=True,
                    field_code=real_field,
                    operation="REPLACE",
                    values=[ProfileIntentValue(
                        raw_value=str(value),
                        normalized_value=str(value),
                        confidence=1.0,
                    )],
                    confidence=1.0,
                    evidence_text=text,
                    source="FAST_NUMERIC",
                )

        # "Ninguna" literal para campos de lista
        if canonical in self.LIST_FIELDS or canonical in (
            "tipo_dieta", "objetivo_nutricional"
        ):
            from application.services.semantic_entity_resolver import SemanticEntityResolver
            norm = SemanticEntityResolver.normalize(text)
            if norm in self.LITERAL_CLEAR_NORMALIZED:
                return ProfileIntentResult(
                    is_profile_update=True,
                    field_code=canonical,
                    operation="CLEAR",
                    values=[ProfileIntentValue(
                        raw_value="NINGUNA",
                        normalized_value="NINGUNA",
                        confidence=1.0,
                    )],
                    confidence=1.0,
                    evidence_text=text,
                    source="FAST_NUMERIC",
                )

        return None

    def _try_fast_sexo(self, text: str, expected_field: str) -> ProfileIntentResult | None:
        """Fast path para campo sexo: detecta hombre/mujer sin LLM."""
        canonical = self._canonical_field(expected_field)
        if canonical != "sexo":
            return None

        norm = text.lower().strip()
        male_kw = {"hombre", "masculino", "varon", "varón", "macho", "h", "m"}
        female_kw = {"mujer", "femenino", "femenina", "dama", "f"}

        value = None
        if norm in male_kw or any(norm.startswith(k) for k in ("hombre", "masculin", "varon", "varón")):
            value = "masculino"
        elif norm in female_kw or any(norm.startswith(k) for k in ("mujer", "femenin", "dama")):
            value = "femenino"

        if value:
            return ProfileIntentResult(
                is_profile_update=True,
                field_code="sexo",
                operation="REPLACE",
                values=[ProfileIntentValue(
                    raw_value=value,
                    normalized_value=value,
                    confidence=1.0,
                )],
                confidence=1.0,
                evidence_text=text,
                source="FAST_SEXO",
            )
        return None

    async def _extract_with_llm(
        self, user_text: str, expected_field: str | None
    ) -> ProfileIntentResult:
        """
        Comprensión real de la intención usando LLM.
        No depende de listas de frases — entiende lenguaje natural.
        """
        try:
            context_hint = ""
            if expected_field:
                context_hint = f"\n\n[CONTEXTO] El sistema le está preguntando por: {expected_field}"

            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"{user_text}{context_hint}"},
                ],
            )

            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)

            is_update = bool(data.get("is_profile_update", False))
            if not is_update:
                return ProfileIntentResult(
                    is_profile_update=False,
                    confidence=float(data.get("confidence", 0.0)),
                    source="LLM_STRUCTURED",
                )

            raw_values = data.get("values") or []
            values = [
                ProfileIntentValue(
                    raw_value=str(v),
                    normalized_value=SemanticEntityResolver.normalize(str(v)),
                )
                for v in raw_values
                if v
            ]

            return ProfileIntentResult(
                is_profile_update=True,
                field_code=data.get("field_code"),
                operation=data.get("operation", "REPLACE"),
                values=values,
                confidence=float(data.get("confidence", 0.8)),
                evidence_text=user_text,
                needs_clarification=bool(data.get("needs_clarification", False)),
                clarification_question=data.get("clarification_question"),
                source="LLM_STRUCTURED",
            )

        except json.JSONDecodeError:
            logger.warning("ProfileIntentExtractor: LLM returned invalid JSON")
            return ProfileIntentResult()
        except Exception:
            logger.exception("ProfileIntentExtractor: LLM call failed")
            return ProfileIntentResult()

    async def _resolve_values(
        self,
        session: AsyncSession,
        result: ProfileIntentResult,
        usuario_id: int | None,
    ) -> None:
        """
        Resuelve cada valor extraído contra los maestros semánticos.
        Aplica solo a campos que tienen maestros (listas, dieta, objetivo, ubicación).
        """
        field = result.field_code
        if not field or field in self.NUMERIC_FIELDS:
            return  # Los numéricos no necesitan resolución semántica

        if result.operation == "CLEAR":
            return  # CLEAR no necesita resolver valores

        for val in result.values:
            if not val.raw_value or val.raw_value == "NINGUNA":
                continue

            resolution = await self._semantic_resolver.resolve(
                session,
                field_code=field,
                raw_value=val.raw_value,
                usuario_id=usuario_id,
            )

            val.entity_type = resolution.entity_type
            val.entity_code = resolution.entity_code
            val.entity_label = resolution.entity_label
            val.resolution_strategy = resolution.strategy
            val.confidence = resolution.confidence
            val.ambiguous = resolution.ambiguous
            val.candidates = resolution.candidates

            # Si ambiguo y hay candidatos → sugerir aclaración
            if resolution.ambiguous and resolution.candidates:
                result.needs_clarification = True
                top_names = [c.get("label", c.get("code", "?")) for c in resolution.candidates[:3]]
                result.clarification_question = (
                    f"¿Te refieres a alguno de estos?: {', '.join(top_names)}"
                )

    @staticmethod
    def _canonical_field(step: str | None) -> str:
        if not step:
            return ""
        mapping = {
            "peso": "peso_kg",
            "altura": "altura_cm",
            "talla": "altura_cm",
            "restricciones": "restricciones_alimentarias",
            "objetivo": "objetivo_nutricional",
            "meta": "objetivo_nutricional",
        }
        return mapping.get(step.lower(), step.lower())
