"""
Nutribot Backend — Load Test Script
=====================================
Simula múltiples usuarios enviando mensajes al webhook para
validar la arquitectura desacoplada bajo carga.

Uso:
    python load_test.py --url http://localhost:8000 --users 20 --messages 10
    python load_test.py --url http://187.77.19.172:8000 --users 50 --messages 20

Resultado:
    Métricas de latencia (p50, p95, p99), throughput, errores y rate limited.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
except ImportError:
    print("Instala httpx: pip install httpx")
    exit(1)

try:
    import asyncpg
except ImportError:
    asyncpg = None


@dataclass
class LoadTestResult:
    total_requests: int = 0
    successful: int = 0
    duplicates: int = 0
    rate_limited: int = 0
    errors: int = 0
    latencies: list = field(default_factory=list)
    expected_reply_ids: list[str] = field(default_factory=list)
    e2e_checked: bool = False
    e2e_incoming_done: int = 0
    e2e_outgoing_sent: int = 0
    e2e_outgoing_failed: int = 0
    e2e_timeout: bool = False
    e2e_elapsed_s: float = 0.0
    start_time: float = 0
    end_time: float = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def rps(self) -> float:
        return self.total_requests / self.duration if self.duration > 0 else 0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, pct: int) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * pct / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]


def generate_fake_webhook(phone: str, message_text: str, msg_id: str) -> dict:
    """Genera un payload de webhook compatible con Evolution API."""
    return {
        "event": "messages.upsert",
        "instance": "nutribot",
        "data": {
            "key": {
                "remoteJid": f"{phone}@s.whatsapp.net",
                "fromMe": False,
                "id": msg_id,
            },
            "messageType": "conversation",
            "message": {
                "conversation": message_text,
            },
            "messageTimestamp": int(time.time()),
            "pushName": f"LoadTest User {phone[-4:]}",
        },
    }


def normalize_asyncpg_url(db_url: str) -> str:
    if db_url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + db_url.split("://", 1)[1]
    return db_url


async def verify_end_to_end_via_db(
    db_url: str,
    provider_ids: list[str],
    timeout_s: float,
    poll_interval_s: float,
) -> dict:
    """
    Verifica flujo completo: incoming done + outgoing sent para cada mensaje aceptado.
    Usa idempotency_key actual reply:{provider_message_id}
    y soporta legado reply:{provider_message_id}:text.
    """
    started = time.monotonic()
    target_count = len(provider_ids)
    if target_count == 0:
        return {
            "checked": False,
            "incoming_done": 0,
            "outgoing_sent": 0,
            "outgoing_failed": 0,
            "timeout": False,
            "elapsed_s": 0.0,
        }

    if asyncpg is None:
        raise RuntimeError("Falta dependencia asyncpg para validación end-to-end.")

    outgoing_keys_current = [f"reply:{mid}" for mid in provider_ids]
    outgoing_keys_legacy = [f"reply:{mid}:text" for mid in provider_ids]
    outgoing_keys = outgoing_keys_current + outgoing_keys_legacy
    conn = await asyncpg.connect(normalize_asyncpg_url(db_url))
    try:
        while True:
            incoming_done = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM incoming_messages
                WHERE provider_message_id = ANY($1::text[])
                  AND status = 'done'
                """,
                provider_ids,
            )
            outgoing_sent = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM outgoing_messages
                WHERE idempotency_key = ANY($1::text[])
                  AND status = 'sent'
                """,
                outgoing_keys,
            )
            outgoing_failed = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM outgoing_messages
                WHERE idempotency_key = ANY($1::text[])
                  AND status = 'failed'
                """,
                outgoing_keys,
            )

            elapsed = time.monotonic() - started
            completed = incoming_done >= target_count and outgoing_sent >= target_count
            timed_out = elapsed >= timeout_s
            if completed or timed_out:
                return {
                    "checked": True,
                    "incoming_done": int(incoming_done or 0),
                    "outgoing_sent": int(outgoing_sent or 0),
                    "outgoing_failed": int(outgoing_failed or 0),
                    "timeout": timed_out and not completed,
                    "elapsed_s": round(elapsed, 2),
                }

            await asyncio.sleep(poll_interval_s)
    finally:
        await conn.close()


SAMPLE_MESSAGES = [
    "Hola",
    "Buenos días",
    "Tengo 35 años",
    "Mi peso es 75 kg",
    "Mido 1.70",
    "Dame un menú para diabetes tipo 2",
    "¿Qué puedo desayunar mañana?",
    "Soy alérgico al maní",
    "No tengo restricciones",
    "Quiero bajar de peso",
    "personalizar mi perfil",
    "omitir",
    "si",
    "no",
    "80",
    "¿Qué es el IMC?",
    "Recomiéndame algo saludable para almorzar",
    "Gracias",
    "Me equivoqué, mi edad es 40",
    "Encuesta",
]


async def simulate_user(
    client: httpx.AsyncClient,
    url: str,
    webhook_secret: Optional[str],
    phone: str,
    num_messages: int,
    delay_min: float,
    delay_max: float,
    result: LoadTestResult,
):
    """Simula un usuario enviando mensajes con delays aleatorios."""
    for i in range(num_messages):
        msg = random.choice(SAMPLE_MESSAGES)
        msg_id = "LOAD_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        payload = generate_fake_webhook(phone, msg, msg_id)

        t0 = time.monotonic()
        try:
            headers = {"X-Webhook-Secret": webhook_secret} if webhook_secret else None
            resp = await client.post(f"{url}/webhook", json=payload, headers=headers, timeout=10)
            latency = time.monotonic() - t0
            result.latencies.append(latency)
            result.total_requests += 1

            body = resp.json()
            status = body.get("status", "")
            if status == "ok":
                result.successful += 1
                result.expected_reply_ids.append(msg_id)
            elif status == "duplicate":
                result.duplicates += 1
            elif status == "rate_limited":
                result.rate_limited += 1
            else:
                result.errors += 1

        except Exception as e:
            result.total_requests += 1
            result.errors += 1
            result.latencies.append(time.monotonic() - t0)

        # Delay entre mensajes (simula comportamiento humano)
        delay = random.uniform(delay_min, delay_max)
        await asyncio.sleep(delay)


async def run_load_test(
    url: str,
    num_users: int,
    messages_per_user: int,
    delay_min: float,
    delay_max: float,
    webhook_secret: Optional[str] = None,
    db_url: Optional[str] = None,
    e2e_timeout_s: float = 60.0,
    e2e_poll_interval_s: float = 1.0,
):
    """Ejecuta la prueba de carga con N usuarios concurrentes."""
    result = LoadTestResult()

    # Verificar que el servidor esté vivo
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}/health", timeout=5)
            health = resp.json()
            print(f"✅ Servidor OK: {health}")
        except Exception as e:
            print(f"❌ No se pudo conectar a {url}/health: {e}")
            return

    # Generar teléfonos ficticios
    phones = [f"5199{random.randint(1000000, 9999999)}" for _ in range(num_users)]

    print(f"\n🚀 Iniciando prueba de carga:")
    print(f"   URL: {url}")
    print(f"   Usuarios: {num_users}")
    print(f"   Mensajes por usuario: {messages_per_user}")
    print(f"   Total mensajes: {num_users * messages_per_user}")
    print(f"   Delay entre mensajes: {delay_min:.1f}-{delay_max:.1f}s")
    print()

    result.start_time = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [
            simulate_user(client, url, webhook_secret, phone, messages_per_user, delay_min, delay_max, result)
            for phone in phones
        ]
        await asyncio.gather(*tasks)

    result.end_time = time.monotonic()

    # Consultar estado de colas
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{url}/health/queues", timeout=5)
            queues = resp.json()
        except Exception:
            queues = {"status": "unavailable"}

    # Verificación E2E opcional (DB) para validar inbox + outbox.
    if db_url:
        try:
            e2e = await verify_end_to_end_via_db(
                db_url=db_url,
                provider_ids=result.expected_reply_ids,
                timeout_s=e2e_timeout_s,
                poll_interval_s=e2e_poll_interval_s,
            )
            result.e2e_checked = e2e["checked"]
            result.e2e_incoming_done = e2e["incoming_done"]
            result.e2e_outgoing_sent = e2e["outgoing_sent"]
            result.e2e_outgoing_failed = e2e["outgoing_failed"]
            result.e2e_timeout = e2e["timeout"]
            result.e2e_elapsed_s = e2e["elapsed_s"]
        except Exception as e:
            print(f"⚠️  No se pudo validar end-to-end vía DB: {e}")

    # Reporte
    print("=" * 60)
    print("📊 RESULTADOS DE PRUEBA DE CARGA")
    print("=" * 60)
    print(f"  Duración total:     {result.duration:.2f}s")
    print(f"  Total requests:     {result.total_requests}")
    print(f"  Throughput:         {result.rps:.1f} req/s")
    print()
    print(f"  ✅ Exitosos:        {result.successful}")
    print(f"  🔄 Duplicados:      {result.duplicates}")
    print(f"  ⏳ Rate limited:    {result.rate_limited}")
    print(f"  ❌ Errores:         {result.errors}")
    print()
    print(f"  Latencia p50:       {result.p50*1000:.1f}ms")
    print(f"  Latencia p95:       {result.p95*1000:.1f}ms")
    print(f"  Latencia p99:       {result.p99*1000:.1f}ms")
    print()
    print(f"  Colas Redis:        {queues}")
    if db_url:
        print()
        print("  Verificación E2E (DB):")
        print(f"    Esperados (status=ok): {len(result.expected_reply_ids)}")
        print(f"    incoming done:         {result.e2e_incoming_done}")
        print(f"    outgoing sent:         {result.e2e_outgoing_sent}")
        print(f"    outgoing failed:       {result.e2e_outgoing_failed}")
        print(f"    timeout:               {result.e2e_timeout}")
        print(f"    elapsed:               {result.e2e_elapsed_s:.2f}s")
    print("=" * 60)

    # Guardar resultados en JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "url": url,
            "users": num_users,
            "messages_per_user": messages_per_user,
            "delay_min": delay_min,
            "delay_max": delay_max,
            "webhook_secret_provided": bool(webhook_secret),
            "db_url_provided": bool(db_url),
            "e2e_timeout_s": e2e_timeout_s,
            "e2e_poll_interval_s": e2e_poll_interval_s,
        },
        "results": {
            "duration_s": round(result.duration, 2),
            "total_requests": result.total_requests,
            "throughput_rps": round(result.rps, 1),
            "successful": result.successful,
            "duplicates": result.duplicates,
            "rate_limited": result.rate_limited,
            "errors": result.errors,
            "latency_p50_ms": round(result.p50 * 1000, 1),
            "latency_p95_ms": round(result.p95 * 1000, 1),
            "latency_p99_ms": round(result.p99 * 1000, 1),
            "expected_reply_ids": len(result.expected_reply_ids),
            "e2e_checked": result.e2e_checked,
            "e2e_incoming_done": result.e2e_incoming_done,
            "e2e_outgoing_sent": result.e2e_outgoing_sent,
            "e2e_outgoing_failed": result.e2e_outgoing_failed,
            "e2e_timeout": result.e2e_timeout,
            "e2e_elapsed_s": result.e2e_elapsed_s,
        },
        "queues": queues,
    }

    report_file = "load_test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📝 Reporte guardado en: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="NutriBot Load Test")
    parser.add_argument("--url", default="http://localhost:8000", help="URL del backend")
    parser.add_argument("--users", type=int, default=20, help="Usuarios concurrentes")
    parser.add_argument("--messages", type=int, default=10, help="Mensajes por usuario")
    parser.add_argument("--delay-min", type=float, default=0.1, help="Delay mínimo entre mensajes (s)")
    parser.add_argument("--delay-max", type=float, default=0.5, help="Delay máximo entre mensajes (s)")
    parser.add_argument(
        "--webhook-secret",
        default=os.getenv("WEBHOOK_SECRET", ""),
        help="Header X-Webhook-Secret. Default: WEBHOOK_SECRET env var.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="URL de PostgreSQL para validar end-to-end (ej: postgresql://user:pass@host:5432/db)",
    )
    parser.add_argument("--e2e-timeout", type=float, default=60.0, help="Timeout de validación E2E en segundos")
    parser.add_argument("--e2e-poll-interval", type=float, default=1.0, help="Intervalo de sondeo E2E en segundos")
    args = parser.parse_args()

    asyncio.run(run_load_test(
        url=args.url,
        num_users=args.users,
        messages_per_user=args.messages,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        webhook_secret=(args.webhook_secret or None),
        db_url=args.db_url,
        e2e_timeout_s=args.e2e_timeout,
        e2e_poll_interval_s=args.e2e_poll_interval,
    ))


if __name__ == "__main__":
    main()
