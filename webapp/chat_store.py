from __future__ import annotations

import json
import uuid
from pathlib import Path


CHATS_DIR = Path("webapp_data/chats")
MAX_HISTORY_MESSAGES = 40


def ensure_chat_storage() -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)


def create_chat_id() -> str:
    return str(uuid.uuid4())


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def load_messages(chat_id: str) -> list[dict]:
    ensure_chat_storage()
    path = _chat_path(chat_id)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_messages(chat_id: str, messages: list[dict]) -> None:
    ensure_chat_storage()
    tail = messages[-MAX_HISTORY_MESSAGES:]
    _chat_path(chat_id).write_text(
        json.dumps(tail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_messages(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()
