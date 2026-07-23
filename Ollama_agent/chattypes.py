"""Shared types for the Ollama chatbot."""
from typing import Literal, TypedDict

# The two mechanisms by which a model can expose chain-of-thought reasoning
ThinkingMode = Literal["field", "tags"]


class ChatMessage(TypedDict, total=False):
    """A single turn in the conversation history.

    Required fields: role, content.
    Optional field: thinking (only present on assistant messages from thinking models).
    """

    role: Literal["user", "assistant", "system"]
    content: str
    thinking: str  # optional; empty string means no thinking was captured