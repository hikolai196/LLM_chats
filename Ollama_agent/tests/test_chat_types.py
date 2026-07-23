# Tests for chat_types.py — verify the TypedDict and Literal definitions.

# These tests are deliberately lightweight: they confirm that valid shapes are
# accepted and that the Literal values match what the rest of the codebase
# branches on.

import pytest
from typing import get_args

from Chattypes import ChatMessage, ThinkingMode

class TestThinkingMode:
    def test_valid_values_are_field_and_tags(self):
        assert set(get_args(ThinkingMode)) == {"field", "tags"}

    def test_literal_values_match_config_registry(self):
        """Thinking mode strings used in config.py must be valid ThinkingMode values."""
        import Config
        valid = set(get_args(ThinkingMode))
        for name, meta in Config.THINKING_MODELS.items():
            assert meta["thinking_mode"] in valid, (
                f"Config entry '{name}' uses unknown mode '{meta['thinking_mode']}'"
            )

class TestChatMessage:
    def test_minimal_user_message(self):
        msg: ChatMessage = {"role": "user", "content": "hello"}
        assert msg["role"] == "user"
        assert msg["content"] == "hello"

    def test_assistant_message_with_thinking(self):
        msg: ChatMessage = {
            "role": "assistant",
            "content": "answer",
            "thinking": "chain of thought",
        }
        assert msg["thinking"] == "chain of thought"

    def test_assistant_message_without_thinking(self):
        msg: ChatMessage = {"role": "assistant", "content": "answer"}
        assert msg.get("thinking") is None

    def test_system_message(self):
        msg: ChatMessage = {"role": "system", "content": "Be helpful."}
        assert msg["role"] == "system"

    def test_thinking_field_defaults_to_absent(self):
        """total=False means all keys are optional — thinking should be absent, not ""."""
        msg: ChatMessage = {"role": "user", "content": "hi"}
        assert "thinking" not in msg