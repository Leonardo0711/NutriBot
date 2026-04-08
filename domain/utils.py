"""
Nutribot Backend — Utilities
Utilidades transversales al dominio.
"""
from datetime import datetime, timezone, timedelta

# Definir la zona horaria de Perú (UTC-5)
PERU_TZ = timezone(timedelta(hours=-5))

def get_now_peru() -> datetime:
    """Retorna la fecha y hora actual en la zona horaria de Perú.
    Retorna un datetime NAIVE (sin tzinfo) porque las columnas de la DB
    son TIMESTAMP WITHOUT TIME ZONE y la conexión ya tiene timezone='America/Lima'.
    """
    return datetime.now(PERU_TZ).replace(tzinfo=None)

