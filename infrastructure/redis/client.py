"""
Nutribot Backend — Redis Client
================================
Capa de conexion a Redis para colas, locks y cache.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import redis.asyncio as redis

from config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Retorna una conexion compartida a Redis."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
    return _pool


async def close_redis() -> None:
    """Cierra el pool de Redis. Llamar en app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("Redis pool cerrado.")


# ──────────────────────────────────────────────
# Colas (Listas FIFO)
# ──────────────────────────────────────────────

INBOX_QUEUE = "nb:inbox"
OUTBOX_QUEUE = "nb:outbox"
# Compatibilidad legacy: algunos workers/sweeper aún referencian esta cola.
EXTRACT_QUEUE = "nb:extract"


async def enqueue(queue: str, message_id: int | str) -> None:
    """Publica un ID en la cola Redis."""
    r = get_redis()
    await r.rpush(queue, str(message_id))
    logger.debug("Enqueued %s -> %s", message_id, queue)


async def dequeue(queue: str, timeout: float = 1.0) -> Optional[str]:
    """Consume un ID de la cola Redis (blocking pop)."""
    r = get_redis()
    result = await r.blpop(queue, timeout=timeout)
    if result:
        _, value = result
        return value
    return None


async def queue_depth(queue: str) -> int:
    """Retorna la profundidad actual de una cola."""
    r = get_redis()
    return await r.llen(queue)


# ──────────────────────────────────────────────
# Locks efimeros
# ──────────────────────────────────────────────

LOCK_PREFIX = "nb:lock:"
_RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""
_RATE_LIMIT_INCR_LUA = """
local count = redis.call("INCR", KEYS[1])
if count == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return count
"""


async def acquire_lock(key: str, ttl_seconds: int = 30) -> Optional[str]:
    """Intenta adquirir un lock efimero. Retorna token si lo consiguio."""
    r = get_redis()
    lock_key = f"{LOCK_PREFIX}{key}"
    token = str(uuid.uuid4())
    acquired = await r.set(lock_key, token, nx=True, ex=ttl_seconds)
    if not acquired:
        return None
    return token


async def release_lock(key: str, token: str) -> bool:
    """Libera lock solo si el token coincide (compare-and-delete)."""
    r = get_redis()
    lock_key = f"{LOCK_PREFIX}{key}"
    released = await r.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
    return bool(released)


# ──────────────────────────────────────────────
# Deduplicacion temporal
# ──────────────────────────────────────────────

DEDUP_PREFIX = "nb:dedup:"


async def is_duplicate(provider_message_id: str, ttl_seconds: int = 300) -> bool:
    """
    Verifica si un provider_message_id ya fue visto recientemente.
    Si no, lo marca como visto con TTL.
    Retorna True si es duplicado.
    """
    r = get_redis()
    dedup_key = f"{DEDUP_PREFIX}{provider_message_id}"
    was_set = await r.set(dedup_key, "1", nx=True, ex=ttl_seconds)
    return not bool(was_set)  # Si no se pudo setear, ya existia = duplicado


async def mark_seen(provider_message_id: str, ttl_seconds: int = 300) -> None:
    """
    Marca un provider_message_id como visto (best effort, no bloqueante).
    Redis aqui es cache/acelerador, NO fuente de verdad.
    """
    r = get_redis()
    dedup_key = f"{DEDUP_PREFIX}{provider_message_id}"
    await r.set(dedup_key, "1", ex=ttl_seconds)


# ──────────────────────────────────────────────
# Rate limiting
# ──────────────────────────────────────────────

RATE_LIMIT_PREFIX = "nb:rl:"


async def check_rate_limit(
    subject: str,
    category: str = "msg",
    max_count: int = 10,
    window_seconds: int = 60,
) -> bool:
    """
    Verifica rate limit por subject/categoria.
    Retorna True si esta DENTRO del limite (puede proceder).
    Retorna False si EXCEDIO el limite.
    """
    r = get_redis()
    rl_key = f"{RATE_LIMIT_PREFIX}{subject}:{category}"
    current_count = int(await r.eval(_RATE_LIMIT_INCR_LUA, 1, rl_key, int(window_seconds)))
    return current_count <= max_count


async def is_rate_limited(
    subject: str,
    category: str = "msg",
    max_count: int = 10,
    window_seconds: int = 60,
) -> bool:
    """
    Alias conveniente: retorna True si el rate limit fue excedido.
    """
    allowed = await check_rate_limit(subject, category, max_count, window_seconds)
    return not allowed
