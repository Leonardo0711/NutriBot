"""
Nutribot Backend - Reply objects
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BotReply:
    text: Optional[str] = None
    content_type: str = "text"
    payload_json: Optional[dict] = None
    response_id: Optional[str] = None

