"""
Smoke test for interactive survey flows.

Covers:
1) survey without audio/image trials
2) survey with audio trial
3) survey with image trial
4) duplicate webhook idempotency check (same provider_message_id)

Usage:
  python scripts/smoke_interactive_survey.py --url http://localhost:8000 --webhook-secret YOUR_SECRET
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import string
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from sqlalchemy import text

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from infrastructure.db.connection import get_session_factory  # noqa: E402


CONSENT_STATE = "esperando_consentimiento_encuesta"


@dataclass
class CaseExpectation:
    name: str
    audio_evaluado: bool
    audio_no_aplica: bool
    imagen_evaluada: bool
    imagen_no_aplica: bool


def _rand_token(size: int = 8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=size))


def _build_base(phone: str, provider_id: str, message_type: str, message_obj: dict) -> dict:
    return {
        "event": "messages.upsert",
        "instance": "nutribot",
        "data": {
            "key": {
                "remoteJid": f"{phone}@s.whatsapp.net",
                "fromMe": False,
                "id": provider_id,
            },
            "messageType": message_type,
            "message": message_obj,
            "messageTimestamp": int(time.time()),
            "pushName": f"Smoke {phone[-4:]}",
        },
    }


def build_text(phone: str, provider_id: str, body: str) -> dict:
    return _build_base(phone, provider_id, "conversation", {"conversation": body})


def build_button(phone: str, provider_id: str, selected_id: str, selected_text: str) -> dict:
    return _build_base(
        phone,
        provider_id,
        "buttonsResponseMessage",
        {
            "buttonsResponseMessage": {
                "selectedButtonId": selected_id,
                "selectedDisplayText": selected_text,
            }
        },
    )


def build_list(phone: str, provider_id: str, selected_row_id: str, title: str) -> dict:
    return _build_base(
        phone,
        provider_id,
        "listResponseMessage",
        {
            "listResponseMessage": {
                "singleSelectReply": {
                    "selectedRowId": selected_row_id,
                    "title": title,
                }
            }
        },
    )


def build_audio(phone: str, provider_id: str) -> dict:
    return _build_base(
        phone,
        provider_id,
        "audioMessage",
        {"audioMessage": {"mimetype": "audio/ogg; codecs=opus"}},
    )


def build_image(phone: str, provider_id: str) -> dict:
    return _build_base(
        phone,
        provider_id,
        "imageMessage",
        {"imageMessage": {"mimetype": "image/jpeg", "caption": "smoke image"}},
    )


class SmokeInteractiveSurvey:
    def __init__(self, base_url: str, webhook_secret: str, timeout_s: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.webhook_secret = webhook_secret
        self.timeout_s = timeout_s
        self.session_factory = get_session_factory()
        self._counter = 0

    def _next_mid(self, prefix: str) -> str:
        self._counter += 1
        return f"SMK_{prefix}_{self._counter}_{_rand_token(6)}"

    async def _post_webhook(self, payload: dict) -> dict:
        headers = {"X-Webhook-Secret": self.webhook_secret}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{self.base_url}/webhook", json=payload, headers=headers)
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if resp.status_code >= 400:
                raise RuntimeError(f"Webhook HTTP {resp.status_code}: {body}")
            return body

    async def _wait_incoming_done(self, provider_message_id: str) -> None:
        started = time.monotonic()
        while time.monotonic() - started < self.timeout_s:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(
                        """
                        SELECT status, error_detail
                        FROM incoming_messages
                        WHERE provider_message_id = :mid
                        ORDER BY id DESC
                        LIMIT 1
                        """
                    ),
                    {"mid": provider_message_id},
                )
                row = result.fetchone()
                if row:
                    if row.status == "done":
                        return
                    if row.status == "failed":
                        raise RuntimeError(
                            f"incoming failed for {provider_message_id}: {row.error_detail}"
                        )
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Timeout waiting incoming done for {provider_message_id}")

    async def _get_user_id(self, phone: str) -> int:
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT id FROM usuarios WHERE numero_whatsapp = :phone LIMIT 1"),
                {"phone": phone},
            )
            row = result.fetchone()
            if not row:
                raise RuntimeError(f"user not found for phone {phone}")
            return int(row.id)

    async def _get_state(self, user_id: int) -> tuple[str, Optional[str]]:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT mode, awaiting_question_code
                    FROM conversation_state
                    WHERE usuario_id = :uid
                    """
                ),
                {"uid": user_id},
            )
            row = result.fetchone()
            if not row:
                raise RuntimeError(f"conversation_state not found for user_id={user_id}")
            return str(row.mode), row.awaiting_question_code

    async def _assert_state(self, user_id: int, expected_mode: str, expected_awaiting: Optional[str]) -> None:
        mode, awaiting = await self._get_state(user_id)
        if mode != expected_mode or awaiting != expected_awaiting:
            raise AssertionError(
                f"state mismatch user={user_id}: got mode={mode}, awaiting={awaiting}; "
                f"expected mode={expected_mode}, awaiting={expected_awaiting}"
            )

    async def _send_and_wait(self, payload: dict, expect_status: set[str]) -> dict:
        provider_message_id = payload["data"]["key"]["id"]
        body = await self._post_webhook(payload)
        status = body.get("status", "")
        if status not in expect_status:
            raise AssertionError(
                f"unexpected webhook status for {provider_message_id}: {status}. expected={expect_status}"
            )
        if status == "ok":
            await self._wait_incoming_done(provider_message_id)
        return body

    async def _preflight_schema(self) -> None:
        required = [
            ("outgoing_messages", "payload_json"),
            ("formulario_en_progreso", "audio_test_requested"),
            ("formulario_en_progreso", "image_test_requested"),
            ("respuestas_formulario", "audio_evaluado"),
            ("respuestas_formulario", "imagen_evaluada"),
        ]
        async with self.session_factory() as session:
            for table_name, column_name in required:
                result = await session.execute(
                    text(
                        """
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = :table_name
                          AND column_name = :column_name
                        LIMIT 1
                        """
                    ),
                    {"table_name": table_name, "column_name": column_name},
                )
                if result.scalar() is None:
                    raise RuntimeError(
                        "Missing schema column "
                        f"{table_name}.{column_name}. Apply migration 009_interactive_survey_and_media_trials.sql"
                    )

    async def _verify_outbox_interactive(self, user_id: int) -> None:
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM outgoing_messages
                    WHERE usuario_id = :uid
                      AND content_type IN ('interactive_buttons', 'interactive_list')
                    """
                ),
                {"uid": user_id},
            )
            count = int(result.scalar() or 0)
            if count <= 0:
                raise AssertionError(f"No interactive outbox messages found for user {user_id}")

    async def _verify_final_case(self, user_id: int, expected: CaseExpectation) -> None:
        await self._assert_state(user_id, "active_chat", None)

        async with self.session_factory() as session:
            progress_result = await session.execute(
                text(
                    """
                    SELECT estado_actual, completado_en
                    FROM formulario_en_progreso
                    WHERE usuario_id = :uid
                    ORDER BY actualizado_en DESC
                    LIMIT 1
                    """
                ),
                {"uid": user_id},
            )
            progress = progress_result.fetchone()
            if not progress:
                raise AssertionError("formulario_en_progreso row not found")
            if progress.estado_actual != "completado":
                raise AssertionError(f"Expected estado_actual=completado, got {progress.estado_actual}")
            if progress.completado_en is None:
                raise AssertionError("Expected completado_en to be set")

            resp_result = await session.execute(
                text(
                    """
                    SELECT audio_evaluado, audio_no_aplica, imagen_evaluada, imagen_no_aplica
                    FROM respuestas_formulario
                    WHERE usuario_id = :uid
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {"uid": user_id},
            )
            resp = resp_result.fetchone()
            if not resp:
                raise AssertionError("respuestas_formulario row not found")

            got = (
                bool(resp.audio_evaluado),
                bool(resp.audio_no_aplica),
                bool(resp.imagen_evaluada),
                bool(resp.imagen_no_aplica),
            )
            exp = (
                expected.audio_evaluado,
                expected.audio_no_aplica,
                expected.imagen_evaluada,
                expected.imagen_no_aplica,
            )
            if got != exp:
                raise AssertionError(f"Analytics flags mismatch for {expected.name}: got={got} expected={exp}")

        await self._verify_outbox_interactive(user_id)

    async def run_case(self, case_name: str, expectation: CaseExpectation) -> None:
        phone = f"5199{random.randint(1000000, 9999999)}"
        prefix = case_name.upper()
        print(f"\n[CASE {case_name}] phone={phone}")

        # Start survey
        mid = self._next_mid(prefix)
        await self._send_and_wait(build_text(phone, mid, "encuesta"), {"ok"})
        uid = await self._get_user_id(phone)
        await self._assert_state(uid, "active_chat", CONSENT_STATE)

        # Consent + duplicate check
        mid = self._next_mid(prefix)
        consent_payload = build_button(phone, mid, "survey:consent:yes", "Si")
        await self._send_and_wait(consent_payload, {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_correo")
        # Duplicate with same provider_message_id must be dropped by DB uniqueness.
        duplicate_body = await self._send_and_wait(consent_payload, {"duplicate"})
        if duplicate_body.get("status") != "duplicate":
            raise AssertionError("Duplicate check failed")

        # Email
        mid = self._next_mid(prefix)
        await self._send_and_wait(build_text(phone, mid, f"qa_{case_name}@example.com"), {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_p1")

        # P1..P7
        for i in range(1, 8):
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_list(phone, mid, f"survey:p{i}:4", "4"), {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_audio_optin")

        # Audio/Image branch
        if case_name == "no_media":
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:audio_optin:no", "Omitir"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_imagen_optin")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:image_optin:no", "Omitir"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_p10")
        elif case_name == "audio":
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:audio_optin:yes", "Si, probar"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_audio_prueba")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_audio(phone, mid), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_p8")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_list(phone, mid, "survey:p8:4", "4"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_imagen_optin")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:image_optin:no", "Omitir"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_p10")
        elif case_name == "image":
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:audio_optin:no", "Omitir"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_imagen_optin")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_button(phone, mid, "survey:image_optin:yes", "Si, probar"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_imagen_prueba")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_image(phone, mid), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_p9")
            mid = self._next_mid(prefix)
            await self._send_and_wait(build_list(phone, mid, "survey:p9:5", "5"), {"ok"})
            await self._assert_state(uid, "collecting_usability", "esperando_p10")
        else:
            raise ValueError(f"unsupported case: {case_name}")

        # P10, NPS, comment, auth
        mid = self._next_mid(prefix)
        await self._send_and_wait(build_list(phone, mid, "survey:p10:4", "4"), {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_nps")

        mid = self._next_mid(prefix)
        await self._send_and_wait(build_list(phone, mid, "survey:nps:9", "9"), {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_comentario")

        mid = self._next_mid(prefix)
        await self._send_and_wait(build_text(phone, mid, "todo bien"), {"ok"})
        await self._assert_state(uid, "collecting_usability", "esperando_autorizacion")

        mid = self._next_mid(prefix)
        await self._send_and_wait(build_button(phone, mid, "survey:auth:yes", "Si autorizo"), {"ok"})

        await self._verify_final_case(uid, expectation)
        print(f"[CASE {case_name}] OK")

    async def run(self, cases: list[str]) -> None:
        await self._preflight_schema()
        await self._healthcheck()

        mapping = {
            "no_media": CaseExpectation(
                name="no_media",
                audio_evaluado=False,
                audio_no_aplica=True,
                imagen_evaluada=False,
                imagen_no_aplica=True,
            ),
            "audio": CaseExpectation(
                name="audio",
                audio_evaluado=True,
                audio_no_aplica=False,
                imagen_evaluada=False,
                imagen_no_aplica=True,
            ),
            "image": CaseExpectation(
                name="image",
                audio_evaluado=False,
                audio_no_aplica=True,
                imagen_evaluada=True,
                imagen_no_aplica=False,
            ),
        }
        for case_name in cases:
            await self.run_case(case_name, mapping[case_name])
        print("\nALL CASES PASSED")

    async def _healthcheck(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            if resp.status_code >= 400:
                raise RuntimeError(f"Healthcheck failed: HTTP {resp.status_code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for interactive survey.")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL.")
    parser.add_argument(
        "--webhook-secret",
        default=os.getenv("WEBHOOK_SECRET", ""),
        help="X-Webhook-Secret header value. Default from WEBHOOK_SECRET env var.",
    )
    parser.add_argument(
        "--cases",
        default="no_media,audio,image",
        help="Comma-separated cases: no_media,audio,image",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="Timeout per inbound message.")
    args = parser.parse_args()
    if not args.webhook_secret:
        raise ValueError("Missing --webhook-secret (or WEBHOOK_SECRET env var).")
    return args


async def _amain() -> None:
    args = parse_args()
    raw_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    valid = {"no_media", "audio", "image"}
    unknown = [c for c in raw_cases if c not in valid]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}. Valid={sorted(valid)}")

    runner = SmokeInteractiveSurvey(
        base_url=args.url,
        webhook_secret=args.webhook_secret,
        timeout_s=args.timeout,
    )
    await runner.run(raw_cases)


if __name__ == "__main__":
    asyncio.run(_amain())

