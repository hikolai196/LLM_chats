# Ollama Chatbot — Streamlit entry point.

# This module is responsible only for UI: layout, widgets, and rendering.
# All API interaction is delegated to chat.py; model registry to models.py.

import streamlit as st
import ollama

from chat import (
    _StreamPlaceholders,
    build_payload,
    fetch_response,
    stream_response,
)
from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    HISTORY_WARNING_THRESHOLD,
    MAX_HISTORY_MESSAGES,
    MAX_TOKENS_RANGE,
    TEMPERATURE_RANGE,
    TOP_P_RANGE,
)
from model import build_options, get_available_models, get_thinking_mode
from chattypes import ChatMessage

# PAGE CONFIG

st.set_page_config(page_title="Ollama Chatbot", page_icon="🤖", layout="wide")

# SESSION STATE INIYIALIZATION

def _init_session_state(model_names: list[str]) -> None:
    """Ensure all required session-state keys exist with sensible defaults."""
    defaults: dict = {
        "messages": [],
        "model_name": model_names[0] if model_names else "",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# SIDEBAR

def _render_sidebar(model_names: list[str]) -> dict:
    """Render the sidebar and return the current UI settings as a plain dict.

    Keeping widget rendering and value collection here means app.py's main
    flow only has to read from this dict — no st.session_state.* scattered
    around the body.
    """
    with st.sidebar:
        st.subheader("Settings")

        # --- Model selector ---
        if model_names:
            current_index = (
                model_names.index(st.session_state.model_name)
                if st.session_state.model_name in model_names
                else 0
            )
            st.session_state.model_name = st.selectbox(
                "Model", model_names, index=current_index
            )
        else:
            st.warning("No models found. Is Ollama running?")

        # --- System prompt (persisted in session state) ---
        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=100,
        )

        st.divider()

        # --- Generation parameters ---
        t_min, t_max = TEMPERATURE_RANGE
        temperature = st.slider(
            "Temperature", t_min, t_max, DEFAULT_TEMPERATURE, step=0.05,
            help="Higher = more creative, lower = more deterministic",
        )

        p_min, p_max = TOP_P_RANGE
        top_p = st.slider(
            "Top-p", p_min, p_max, DEFAULT_TOP_P, step=0.05,
            help="Nucleus sampling — lower focuses on the most likely tokens",
        )

        mt_min, mt_max = MAX_TOKENS_RANGE
        max_tokens = st.slider(
            "Max tokens", mt_min, mt_max, DEFAULT_MAX_TOKENS, step=64,
            help="Cap on reply length",
        )

        stream = st.toggle("Stream responses", value=True)

        # --- Thinking toggle (only shown for supported models) ---
        thinking_mode = get_thinking_mode(st.session_state.model_name)
        if thinking_mode:
            enable_thinking = st.toggle(
                "Enable thinking box",
                value=True,
                help="Show model reasoning in a collapsible expander",
            )
        else:
            enable_thinking = False
            st.caption("💡 Thinking not available for this model.")

        # --- Seed ---
        seed_value = st.number_input(
            "Seed (−1 = random)",
            min_value=-1,
            max_value=2**31 - 1,
            value=-1,
            step=1,
            help="Fix the seed for reproducible outputs.",
        )
        seed = int(seed_value) if seed_value >= 0 else None

        st.divider()

        # --- Context health indicator ---
        msg_count = len(st.session_state.messages)
        if msg_count >= HISTORY_WARNING_THRESHOLD:
            st.warning(
                f"⚠️ {msg_count} messages in context "
                f"(limit: {MAX_HISTORY_MESSAGES}). "
                "Older messages will be trimmed."
            )
        else:
            st.caption(f"Messages in context: {msg_count} / {MAX_HISTORY_MESSAGES}")

        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Model: `{st.session_state.model_name}`")

    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
        "thinking_mode": thinking_mode,
        "enable_thinking": enable_thinking,
        "seed": seed,
    }

# MESSAGE RENDERING

def _render_message(msg: ChatMessage) -> None:
    """Render one chat turn, including any thinking block."""
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("💭 Thoughts", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

def _render_history() -> None:
    for msg in st.session_state.messages:
        _render_message(msg)

# RESPONSE RENDERING

def _render_streaming_response(
    *,
    model: str,
    payload: list[dict],
    options: dict,
    active_mode,
) -> tuple[str, str]:
    """Set up streaming placeholders and delegate to chat.stream_response()."""
    thinking_placeholder = None
    if active_mode:
        with st.expander("💭 Thoughts", expanded=False):
            thinking_placeholder = st.empty()

    reply_placeholder = st.empty()
    placeholders = _StreamPlaceholders(reply_placeholder, thinking_placeholder)

    return stream_response(
        model=model,
        payload=payload,
        options=options,
        thinking_mode=active_mode,
        placeholders=placeholders,
    )

def _render_blocking_response(
    *,
    model: str,
    payload: list[dict],
    options: dict,
    active_mode,
) -> tuple[str, str]:
    """Fetch and render a non-streaming response."""
    with st.spinner("Thinking..."):
        content, thinking = fetch_response(
            model=model,
            payload=payload,
            options=options,
            thinking_mode=active_mode,
        )

    if thinking:
        with st.expander("💭 Thoughts", expanded=False):
            st.markdown(thinking)
    st.markdown(content)
    return content, thinking

# ERROR HANDLING

def _handle_ollama_error(error: Exception) -> None:
    """Surface actionable error messages based on exception type."""
    if isinstance(error, ollama.ResponseError):
        if "model" in str(error).lower():
            st.error(
                f"Model not found: `{st.session_state.model_name}`. "
                "Try running `ollama pull <model>` in your terminal."
            )
        else:
            st.error(f"Ollama API error: {error}")
    elif isinstance(error, ConnectionError):
        st.error("Could not connect to Ollama. Is the daemon running? (`ollama serve`)")
    else:
        st.error(f"Unexpected error ({type(error).__name__}): {error}")

# MAIN

def main() -> None:
    model_names = get_available_models()
    _init_session_state(model_names)

    settings = _render_sidebar(model_names)

    st.title("Ollama Chatbot")
    _render_history()

    if not (prompt := st.chat_input("Ask something...", disabled=not model_names)):
        return

    # Record and display the user turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build API inputs
    try:
        options = build_options(
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_tokens=settings["max_tokens"],
            seed=settings["seed"],
            enable_thinking=settings["enable_thinking"],
            thinking_mode=settings["thinking_mode"],
        )
    except ValueError as e:
        st.error(f"Invalid settings: {e}")
        return

    payload = build_payload(
        st.session_state.messages,
        st.session_state.system_prompt,
    )

    active_mode = settings["thinking_mode"] if settings["enable_thinking"] else None

    # Fetch and render the assistant turn
    content = ""
    thinking = ""
    with st.chat_message("assistant"):
        try:
            if settings["stream"]:
                content, thinking = _render_streaming_response(
                    model=st.session_state.model_name,
                    payload=payload,
                    options=options,
                    active_mode=active_mode,
                )
            else:
                content, thinking = _render_blocking_response(
                    model=st.session_state.model_name,
                    payload=payload,
                    options=options,
                    active_mode=active_mode,
                )
        except Exception as e:  # noqa: BLE001
            _handle_ollama_error(e)

    # Persist the assistant turn only if we got a response
    if content:
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=content, thinking=thinking)
        )

if __name__ == "__main__":
    main()