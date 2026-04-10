"""
Nutribot Backend — Parsers
Lógica determinística para limpiar y estandarizar datos de perfil.
"""
import re
from typing import Optional

def parse_weight(val: str) -> Optional[float]:
    """Limpia peso y convierte unidades (lbs -> kg)."""
    if not val: return None
    v = val.lower().strip()
    # Soporta formatos como "72.5 kg", "70", "80quilos"
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(kg|kilo|quilo|lb|libra)?", v)
    if not match: return None
    
    try:
        num = float(match.group(1).replace(",", "."))
        unit = match.group(2)
        if unit and ("lb" in unit or "lib" in unit):
            return round(num * 0.453592, 2)
        # Validación de rango biológico (3kg a 450kg)
        if num < 3.0 or num > 450.0: return None
        return round(num, 2)
    except (ValueError, TypeError):
        return None

def parse_height(val: str) -> Optional[float]:
    """Limpia talla y convierte unidades (mts -> cm)."""
    if not val: return None
    v = val.lower().strip()
    
    # Soporte para pies/pulgadas (5'4")
    ft_in_match = re.search(r"(\d+)\s*'\s*(\d+)\s*\"", v)
    if ft_in_match:
        try:
            ft = int(ft_in_match.group(1))
            inches = int(ft_in_match.group(2))
            return round((ft * 30.48) + (inches * 2.54), 2)
        except (ValueError, TypeError): pass
    
    # Soporte para "1 con 70", "1 con 7" (Muy común en Perú)
    con_match = re.search(r"(\d+)\s+con\s+(\d+)", v)
    if con_match:
        try:
            m = int(con_match.group(1))
            cm = int(con_match.group(2))
            if cm < 10: cm = cm * 10 
            return float(m * 100 + cm)
        except: pass

    # Formato estándar: "1.70 m", "170 cm", "170"
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(cm|m|metro)?", v)
    if not match: return None
    
    try:
        num_str = match.group(1).replace(",", ".")
        num = float(num_str)
        unit = match.group(2)
        
        # Heurística: Si es < 3 y no tiene unidad 'cm', asumimos metros
        if (num < 3.0 and unit != "cm") or (unit and ("m" in unit and "cm" not in unit)):
            num = num * 100.0
            
        # Validación de rango biológico (50cm a 250cm)
        if num < 50.0 or num > 250.0: return None
        return round(num, 2)
    except (ValueError, TypeError):
        return None

def parse_age(val: str) -> Optional[int]:
    """Extrae edad de forma robusta."""
    if not val: return None
    match = re.search(r"(\d+)", str(val))
    if not match: return None
    try:
        num = int(match.group(1))
        if num < 1 or num > 120: return None
        return num
    except (ValueError, TypeError):
        return None

def standardize_text_list(val: str) -> str:
    """Estandariza listas de texto (Alergias, Enfermedades) a MAYÚSCULAS o 'NINGUNA'."""
    if not val: return "NINGUNA"
    v = val.strip().upper()
    
    # Palabras clave de negación
    negations = ["NINGUNA", "NO TENGO", "NADA", "NO", "SIN", "NINGUN", "NULL"]
    if v in negations or any(n in v for n in ["NINGUNA EXCEPCION", "NO TENGO NINGUNA"]):
        return "NINGUNA"
    
    # Limpiar conectores comunes y normalizar separadores
    v = v.replace(" Y ", ", ").replace(" - ", ", ").replace(" / ", ", ")
    
    # Limpiar prefijos conversacionales comunes
    prefixes = [
        "NO COMO ", "NO PUEDO COMER ", "SOY ALERGICO AL ", "SOY ALERGICO A LA ", 
        "SOY ALERGICO A ", "TENGO ", "PADESCO DE ", "SUFRO DE ", "MI OBJETIVO ES ",
        "QUIERO ", "DEBO "
    ]
    for p in prefixes:
        if v.startswith(p):
            v = v[len(p):]

    parts = [p.strip() for p in v.split(",") if p.strip()]
    
    if not parts: return "NINGUNA"
    return ", ".join(parts)
