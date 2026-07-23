# Central configuration for the Ollama chatbot.
#
# All tuneable constants and model registry live here.
# To add support for a new thinking model, add an entry to THINKING_MODELS.

from typing import Any

# MODEL REGISTRY

# Maps model-name prefixes (lowercase) to their thinking-exposure mechanism.
#   "field" — reasoning arrives in chunk.message.thinking (e.g. Gemma)
#   "tags"  — reasoning is embedded as <think>…</think> in content (e.g. Qwen3)
THINKING_MODELS: dict[str, dict[str, Any]] = {
    "gemma": {
        "thinking_mode": "field",
        "max_context_tokens": 8192,
    },
    # Uncomment to enable Qwen3 support:
    # "qwen": {
    #     "thinking_mode": "tags",
    #     "max_context_tokens": 32768,
    # },
}

# CONVERSATION LIMITS

# Maximum number of user+assistant message pairs kept in the payload sent to
# the model. Older messages are silently trimmed. Raise or lower this based on
# the context window of the models you use.
MAX_HISTORY_MESSAGES: int = 40

# Show a sidebar warning when the message count exceeds this threshold.
HISTORY_WARNING_THRESHOLD: int = 30

# UI DEFAULTS

DEFAULT_SYSTEM_PROMPT: str = "You are a helpful assistant."

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_TOKENS: int = 1024

TEMPERATURE_RANGE: tuple[float, float] = (0.0, 2.0)
TOP_P_RANGE: tuple[float, float] = (0.0, 1.0)
MAX_TOKENS_RANGE: tuple[int, int] = (64, 4096)