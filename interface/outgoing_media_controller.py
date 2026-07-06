"""Temporary public media endpoint for provider delivery."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import text

from infrastructure.db.connection import get_session_factory

router = APIRouter()

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


@router.get("/media/outgoing/{token}")
async def get_outgoing_media(token: str):
    if not _TOKEN_RE.match(token):
        raise HTTPException(status_code=404, detail="media_not_found")

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT content_type, data
                FROM outgoing_media_files
                WHERE token = :token
                  AND expires_at > TIMEZONE('America/Lima', NOW())
                LIMIT 1
                """
            ),
            {"token": token},
        )
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="media_not_found")

    content_type, data = row
    return Response(
        content=bytes(data),
        media_type=content_type or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )
