"""In-memory multi-chat session helpers (Streamlit session_state friendly)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(
    *,
    title: str | None = None,
    system_prompt: str = "",
    messages: list | None = None,
) -> dict:
    session_id = uuid.uuid4().hex[:8]
    default_title = title.strip() if title and title.strip() else f"對話 {session_id}"
    return {
        "id": session_id,
        "title": default_title,
        "system_prompt": system_prompt,
        "messages": list(messages) if messages is not None else [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def list_session_labels(sessions: dict[str, dict]) -> list[tuple[str, str]]:
    """Return (id, label) pairs sorted by updated_at descending."""
    items = list(sessions.values())
    items.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return [(s["id"], s.get("title") or s["id"]) for s in items]


def get_session(sessions: dict[str, dict], session_id: str) -> dict | None:
    return sessions.get(session_id)


def upsert_session(sessions: dict[str, dict], session: dict) -> dict[str, dict]:
    updated = dict(sessions)
    session = dict(session)
    session["updated_at"] = _now_iso()
    updated[session["id"]] = session
    return updated


def rename_session(sessions: dict[str, dict], session_id: str, title: str) -> dict[str, dict]:
    session = sessions.get(session_id)
    if not session:
        return sessions
    cleaned = title.strip() or session.get("title") or session_id
    updated_session = dict(session)
    updated_session["title"] = cleaned
    return upsert_session(sessions, updated_session)


def delete_session(sessions: dict[str, dict], session_id: str) -> dict[str, dict]:
    updated = dict(sessions)
    updated.pop(session_id, None)
    return updated


def save_active_session(
    sessions: dict[str, dict],
    session_id: str,
    *,
    messages: list,
    system_prompt: str,
    title: str | None = None,
) -> dict[str, dict]:
    session = sessions.get(session_id)
    if not session:
        session = create_session(
            title=title,
            system_prompt=system_prompt,
            messages=messages,
        )
        session["id"] = session_id
    else:
        session = dict(session)
        session["messages"] = list(messages)
        session["system_prompt"] = system_prompt
        if title is not None and title.strip():
            session["title"] = title.strip()
    return upsert_session(sessions, session)


def ensure_default_session(
    sessions: dict[str, dict] | None,
    active_session_id: str | None,
    *,
    system_prompt: str,
    messages: list,
) -> tuple[dict[str, dict], str]:
    """Ensure at least one session exists; return (sessions, active_id)."""
    store = dict(sessions or {})
    if not store:
        session = create_session(
            title="對話 1",
            system_prompt=system_prompt,
            messages=messages,
        )
        store[session["id"]] = session
        return store, session["id"]

    if active_session_id and active_session_id in store:
        return store, active_session_id

    # Fall back to most recently updated session.
    labels = list_session_labels(store)
    return store, labels[0][0]
