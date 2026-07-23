# Model registry helpers and API option construction.
#
# This module is the single source of truth for:
#   - which models support thinking and how
#   - how to build a validated options dict before sending to Ollama

import logging

import streamlit as st
import ollama

from config import (
    THINKING_MODELS,
    TEMPERATURE_RANGE,
    TOP_P_RANGE,
    MAX_TOKENS_RANGE,
)
from chattypes import ThinkingMode

logger = logging.getLogger(__name__)

# MODEL LIST

@st.cache_resource
def get_available_models() -> list[str]:
    """Return all models available in the local Ollama instance.

    Cached for the lifetime of the Streamlit process so we don't hammer the
    Ollama daemon on every rerun.
    """
    try:
        model_list = ollama.list()
        return [m.model for m in model_list.models]
    except ollama.ResponseError as e:
        st.error(f"Ollama API error while listing models: {e}")
        return []
    except Exception as e:  # noqa: BLE001
        st.error(
            f"Could not connect to Ollama ({type(e).__name__}). "
            "Is the Ollama daemon running?"
        )
        return []

# THINKING-MODE DETECTION

def get_thinking_mode(model_name: str) -> ThinkingMode | None:
    """Return the thinking mode for *model_name*, or None if unsupported.

    Matching is prefix-based and case-insensitive so that versioned names like
    'gemma3:27b' are still recognised.
    """
    name = model_name.lower()
    for prefix, meta in THINKING_MODELS.items():
        if name.startswith(prefix):
            return meta["thinking_mode"]
    return None

# OPTIONS BUILDER

def build_options(
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    enable_thinking: bool,
    thinking_mode: ThinkingMode | None,
) -> dict:
    """Validate and assemble the options dict sent to ollama.chat().

    Raises ValueError if any parameter is out of the expected range, so callers
    surface problems early rather than sending bad values to the API.
    """
    t_min, t_max = TEMPERATURE_RANGE
    if not (t_min <= temperature <= t_max):
        raise ValueError(f"temperature {temperature} out of range [{t_min}, {t_max}]")

    p_min, p_max = TOP_P_RANGE
    if not (p_min <= top_p <= p_max):
        raise ValueError(f"top_p {top_p} out of range [{p_min}, {p_max}]")

    mt_min, mt_max = MAX_TOKENS_RANGE
    if not (mt_min <= max_tokens <= mt_max):
        raise ValueError(f"max_tokens {max_tokens} out of range [{mt_min}, {mt_max}]")

    options: dict = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
    }

    if seed is not None:
        options["seed"] = seed

    # Only Gemma-style models accept a "thinking" flag in options
    if enable_thinking and thinking_mode == "field":
        options["thinking"] = True

    return options