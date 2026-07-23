# Chat API interaction layer.

# All ollama.chat() calls, streaming logic, and response assembly live here.
# The Streamlit UI layer never touches the Ollama client directly.

# Public API:
# build_payload(history, system_prompt) -> list[dict]
# fetch_response(model, payload, options, thinking_mode) -> tuple[str, str]
# stream_response(model, payload, options, thinking_mode, placeholders) -> tuple[str, str]
# trim_history(messages, limit) -> list[ChatMessage]

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

import ollama
import streamlit as st

from config import MAX_HISTORY_MESSAGES
from chattypes import ChatMessage, ThinkingMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# HISTORY MANAGEMENT

def trim_history(
    messages: list[ChatMessage],
    limit: int = MAX_HISTORY_MESSAGES,
) -> list[ChatMessage]:
    """Return at most *limit* most-recent messages.

    Trimming from the front preserves the most recent context. The caller is
    responsible for prepending the system prompt separately.
    """
    if len(messages) > limit:
        logger.warning(
            "History trimmed from %d to %d messages.", len(messages), limit
        )
        return messages[-limit:]
    return messages

def build_payload(
    history: list[ChatMessage],
    system_prompt: str,
) -> list[dict]:
    """Assemble the messages list sent to ollama.chat().

    The system prompt is always prepended as the first message. Only role and
    content are forwarded — the 'thinking' field is internal UI state.
    """
    trimmed = trim_history(history)
    return [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in trimmed
    ]

# THINKING-TAG PARSER (QWEN3 STYLE)

def _parse_thinking_tags(text: str) -> tuple[str, str]:
    """Split Qwen3 content into (thinking, reply).

    Qwen3 wraps its chain-of-thought in <think>…</think> at the start of the
    generated content. Returns (thinking_text, clean_reply). If no tags are
    found the full text is returned as the reply with an empty thinking string.
    """
    match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()

# NON-STREAMING RESPONSE

def fetch_response(
    *,
    model: str,
    payload: list[dict],
    options: dict,
    thinking_mode: ThinkingMode | None,
) -> tuple[str, str]:
    """Call ollama.chat() without streaming.

    Returns (full_response, full_thinking). Raises on Ollama errors so the
    caller can decide how to surface them in the UI.
    """
    response = ollama.chat(
        model=model,
        messages=payload,
        stream=False,
        options=options,
    )

    if thinking_mode == "field":
        thinking = response.message.thinking or ""
        content = response.message.content or ""
    elif thinking_mode == "tags":
        thinking, content = _parse_thinking_tags(response.message.content or "")
    else:
        thinking = ""
        content = response.message.content or ""

    return content, thinking

# STREAMING RESPONSE

class _StreamPlaceholders:
    """Thin wrapper around the two st.empty() slots used during streaming."""

    def __init__(
        self,
        reply_placeholder: st.delta_generator.DeltaGenerator,
        thinking_placeholder: st.delta_generator.DeltaGenerator | None = None,
    ) -> None:
        self.reply = reply_placeholder
        self.thinking = thinking_placeholder

    def update_thinking(self, text: str, *, cursor: bool = False) -> None:
        if self.thinking is not None:
            self.thinking.markdown(text + ("▌" if cursor else ""))

    def update_reply(self, text: str, *, cursor: bool = False) -> None:
        self.reply.markdown(text + ("▌" if cursor else ""))

def stream_response(
    *,
    model: str,
    payload: list[dict],
    options: dict,
    thinking_mode: ThinkingMode | None,
    placeholders: _StreamPlaceholders,
) -> tuple[str, str]:
    """Stream ollama.chat() output into Streamlit placeholders.

    Returns (full_response, full_thinking) once the stream is exhausted.
    Each thinking mode has its own branch; the plain (no-thinking) path is the
    default.
    """
    full_thinking = ""
    full_response = ""

    for chunk in ollama.chat(
        model=model,
        messages=payload,
        stream=True,
        options=options,
    ):
        if thinking_mode == "field":
            full_thinking, full_response = _handle_field_chunk(
                chunk, full_thinking, full_response, placeholders
            )
        elif thinking_mode == "tags":
            full_thinking, full_response = _handle_tags_chunk(
                chunk, full_thinking, full_response, placeholders
            )
        else:
            content_chunk = chunk.message.content or ""
            full_response += content_chunk
            placeholders.update_reply(full_response, cursor=True)

    # Final render — remove cursors
    if full_thinking:
        placeholders.update_thinking(full_thinking)

    # For tag-mode the raw response still contains <think> markup; clean it now
    if thinking_mode == "tags":
        full_thinking, full_response = _parse_thinking_tags(full_response)

    placeholders.update_reply(full_response)
    return full_response, full_thinking

def _handle_field_chunk(
    chunk,
    full_thinking: str,
    full_response: str,
    placeholders: _StreamPlaceholders,
) -> tuple[str, str]:
    """Process one chunk from a Gemma-style field-thinking model."""
    thinking_chunk = chunk.message.thinking or ""
    content_chunk = chunk.message.content or ""

    if thinking_chunk:
        full_thinking += thinking_chunk
        placeholders.update_thinking(full_thinking, cursor=True)
    if content_chunk:
        full_response += content_chunk
        placeholders.update_reply(full_response, cursor=True)

    return full_thinking, full_response

def _handle_tags_chunk(
    chunk,
    full_thinking: str,
    full_response: str,
    placeholders: _StreamPlaceholders,
) -> tuple[str, str]:
    """Process one chunk from a Qwen3-style tag-thinking model."""
    content_chunk = chunk.message.content or ""
    full_response += content_chunk

    if "</think>" not in full_response:
        # Still inside the <think> block — show raw (minus opening tag)
        raw_thinking = full_response.replace("<think>", "").strip()
        placeholders.update_thinking(raw_thinking, cursor=True)
    else:
        # Think block complete — split and render both sides
        ft, fr = _parse_thinking_tags(full_response)
        full_thinking = ft
        placeholders.update_thinking(full_thinking)
        placeholders.update_reply(fr, cursor=True)

    return full_thinking, full_response