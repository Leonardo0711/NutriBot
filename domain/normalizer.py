"""
Nutribot Backend — Text Normalizer
===================================
Normalización de texto para el router barato.
No intenta corregir ortografía, solo reduce ruido
suficiente para que el clasificador pueda decidir.
"""
from __future__ import annotations

import re
import unicodedata


# ──────────────────────────────────────────────
# Abreviaciones comunes del español peruano coloquial
# ──────────────────────────────────────────────
ABBREVIATIONS = {
    "p": "para",
    "pa": "para",
    "pq": "porque",
    "pqe": "porque",
    "xq": "porque",
    "xk": "porque",
    "xke": "porque",
    "q": "que",
    "k": "que",
    "d": "de",
    "dl": "del",
    "tb": "también",
    "tmb": "también",
    "tbien": "también",
    "tmbn": "también",
    "x": "por",
    "bn": "bien",
    "grax": "gracias",
    "grasias": "gracias",
    "grcias": "gracias",
    "dsp": "después",
    "desp": "después",
    "noc": "no se",
    "ns": "no se",
    "nose": "no se",
    "porfa": "por favor",
    "porfavor": "por favor",
    "porfi": "por favor",
    "plz": "por favor",
    "pls": "por favor",
    "osea": "o sea",
    "oe": "oye",
    "oie": "oye",
    "sip": "si",
    "sep": "si",
    "nel": "no",
    "nah": "no",
    "nop": "no",
    "kiero": "quiero",
    "qiero": "quiero",
    "kien": "quien",
    "aki": "aqui",
    "ahi": "ahi",
    "vdd": "verdad",
    "vrdd": "verdad",
    "jaja": "",
    "jajaja": "",
    "jajajaja": "",
    "xd": "",
    "xDD": "",
    "lol": "",
    "v:": "",
}


def strip_accents(text: str) -> str:
    """Remueve diacríticos (tildes, ñ → n, etc.)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_text(raw: str) -> str:
    """
    Pipeline de normalización:
    1. Minúsculas
    2. Strip acentos
    3. Limpiar signos repetidos
    4. Expandir abreviaciones
    5. Colapsar espacios
    6. Strip bordes
    """
    if not raw:
        return ""

    text = raw.lower()
    text = strip_accents(text)

    # Quitar emojis y caracteres no-ascii excepto ñ y signos básicos
    text = re.sub(r'[^\w\s.,;:!?¿¡\-/()@#]', ' ', text)

    # Colapsar signos repetidos: "!!!" → "!", "???" → "?"
    text = re.sub(r'([!?.])\1+', r'\1', text)

    # Colapsar letras repetidas: "holaaaa" → "holaa" (max 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Expandir abreviaciones (solo palabras completas)
    words = text.split()
    expanded = []
    for w in words:
        clean_w = re.sub(r'[.,;:!?]', '', w)
        replacement = ABBREVIATIONS.get(clean_w)
        if replacement is not None:
            if replacement:  # No agregar si es string vacío (risas, xd, etc.)
                expanded.append(replacement)
        else:
            expanded.append(w)
    text = " ".join(expanded)

    # Colapsar espacios
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ──────────────────────────────────────────────
# Detección de patrones numéricos
# ──────────────────────────────────────────────

def extract_numbers(text: str) -> list[float]:
    """Extrae todos los números del texto normalizado."""
    # Captura enteros y decimales (ej: 80, 1.68, 80.5)
    matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    return [float(m) for m in matches]


def extract_number_with_unit(text: str) -> list[tuple[float, str]]:
    """Extrae pares (número, unidad) del texto."""
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*(?:kg|kilos?|kilogramos?)', "kg"),
        (r'(\d+(?:\.\d+)?)\s*(?:cm|centimetros?)', "cm"),
        (r'(\d+(?:\.\d+)?)\s*(?:metros?|mts?|m)\b', "m"),
        (r'(\d+(?:\.\d+)?)\s*(?:anos?|years?)', "years"),
        (r'(\d+(?:\.\d+)?)\s*(?:lbs?|libras?)', "lbs"),
    ]
    results = []
    for pattern, unit in patterns:
        for m in re.finditer(pattern, text):
            results.append((float(m.group(1)), unit))
    return results


# ──────────────────────────────────────────────
# Similitud aproximada (fuzzy básico sin deps)
# ──────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Distancia de Levenshtein simple."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def fuzzy_match(text: str, target: str, threshold: float = 0.7) -> bool:
    """Retorna True si el texto es suficientemente parecido al target."""
    if not text or not target:
        return False
    text_n = strip_accents(text.lower())
    target_n = strip_accents(target.lower())
    if target_n in text_n:
        return True
    # Por palabra
    for word in text_n.split():
        if len(word) < 3:
            continue
        dist = _levenshtein(word, target_n)
        max_len = max(len(word), len(target_n))
        similarity = 1 - (dist / max_len)
        if similarity >= threshold:
            return True
    return False


def fuzzy_match_any(text: str, targets: list[str], threshold: float = 0.7) -> str | None:
    """Retorna el primer target que haga match fuzzy, o None."""
    for target in targets:
        if fuzzy_match(text, target, threshold):
            return target
    return None
