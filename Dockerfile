# ──────────────────────────────────────────────
# Nutribot Backend — Dockerfile
# Imagen ligera con ffmpeg para conversión de audio
# ──────────────────────────────────────────────
FROM python:3.12-slim

# ffmpeg es necesario para convertir ogg/opus → mp3 antes de enviar a STT
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero (cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

EXPOSE 8000

# Uvicorn con reload desactivado en producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
