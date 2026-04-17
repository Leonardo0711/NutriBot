"""
Nutribot Backend - LocalizationService
Postproceso léxico para peruanizar la salida del LLM.
Reemplaza términos genéricos/internacionales por sus equivalentes peruanos.
"""
from __future__ import annotations

import re
from typing import Optional


# Glosario de reemplazo: término internacional -> término peruano
PERU_GLOSSARY: dict[str, str] = {
    "merienda": "refrigerio",
    "meriendas": "refrigerios",
    "boniato": "camote",
    "boniatos": "camotes",
    "batata": "camote",
    "batatas": "camotes",
    "porridge": "avena cocida",
    "quinoa": "quinua",
    "aguacate": "palta",
    "aguacates": "paltas",
    "patata": "papa",
    "patatas": "papas",
    "zumo": "jugo",
    "zumos": "jugos",
    "melocotón": "durazno",
    "melocotones": "duraznos",
    "cacahuete": "maní",
    "cacahuetes": "maníes",
    "judías verdes": "vainitas",
    "calabaza": "zapallo",
    "calabazas": "zapallos",
    "elote": "choclo",
    "elotes": "choclos",
    "maíz tierno": "choclo",
    "plátano macho": "plátano de freír",
    "frijoles negros": "frejoles negros",
    "frijoles": "frejoles",
    "frijol": "frejol",
    "ejotes": "vainitas",
    "pimiento": "ají",
    "pimientos": "ajíes",
    "chile": "ají",
    "chiles": "ajíes",
    "banana": "plátano",
    "bananas": "plátanos",
    "habichuela": "frejol",
    "habichuelas": "frejoles",
    "arándano": "arándano",  # se mantiene, existe en Perú
    "snack": "refrigerio",
    "snacks": "refrigerios",
}


class LocalizationService:
    """Servicio de localización léxica peruana para postproceso de texto."""

    def __init__(self, glossary: Optional[dict[str, str]] = None):
        self._glossary = glossary or PERU_GLOSSARY
        # Compilar patrones de reemplazo ordenados de mayor a menor longitud
        # para que los términos multi-palabra se procesen primero
        self._patterns: list[tuple[re.Pattern, str]] = []
        sorted_terms = sorted(self._glossary.keys(), key=len, reverse=True)
        for term in sorted_terms:
            replacement = self._glossary[term]
            if term == replacement:
                continue  # skip identity mappings
            pattern = re.compile(
                r'\b' + re.escape(term) + r'\b',
                re.IGNORECASE
            )
            self._patterns.append((pattern, replacement))

    def peruanize(self, text: str) -> str:
        """Aplica reemplazos léxicos peruanos al texto."""
        if not text:
            return text
        result = text
        for pattern, replacement in self._patterns:
            # Preservar capitalización del original
            def _replace_match(match: re.Match) -> str:
                original = match.group(0)
                if original[0].isupper():
                    return replacement.capitalize()
                return replacement
            result = pattern.sub(_replace_match, result)
        return result
