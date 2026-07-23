# Tests for chat.py — history management, payload building, and chunk handlers.
#
# ollama.chat() is never called for real here; we test all the logic around it.

import pytest
from unittest.mock import MagicMock, patch

from Chat import (
    _StreamPlaceholders,
    _handle_field_chunk,
    _handle_tags_chunk,
    _parse_thinking_tags,
    build_payload,
    trim_history,
)
from Config import MAX_HISTORY_MESSAGES

# -- TRIM_HISTORY --

class TestTrimHistory:
    def test_short_history_is_unchanged(self, short_history):
        result = trim_history(short_history, limit=10)
        assert result == short_history

    def test_exact_limit_is_unchanged(self):
        msgs = [{"role": "user", "content": str(i)} for i in range(10)]
        assert trim_history(msgs, limit=10) == msgs

    def test_over_limit_trims_from_front(self, long_history):
        result = trim_history(long_history, limit=10)
        assert len(result) == 10
        # Most recent messages are kept
        assert result == long_history[-10:]

    def test_empty_history_returns_empty(self):
        assert trim_history([], limit=10) == []

    def test_default_limit_uses_config(self, long_history):
        """Default limit should come from MAX_HISTORY_MESSAGES (40). Our fixture has 50."""
        result = trim_history(long_history)
        assert len(result) == MAX_HISTORY_MESSAGES

    def test_limit_of_one_keeps_last_message(self, short_history):
        result = trim_history(short_history, limit=1)
        assert len(result) == 1
        assert result[0] == short_history[-1]

# -- BUILD_PAYLOAD --

class TestBuildPayload:
    def test_system_prompt_is_first(self, short_history):
        payload = build_payload(short_history, "Be concise.")
        assert payload[0] == {"role": "system", "content": "Be concise."}

    def test_history_follows_system_prompt(self, short_history):
        payload = build_payload(short_history, "Be concise.")
        assert len(payload) == 1 + len(short_history)

    def test_thinking_field_is_stripped(self, assistant_message_with_thinking):
        history = [assistant_message_with_thinking]
        payload = build_payload(history, "sys")
        assistant_payload = payload[1]
        assert "thinking" not in assistant_payload
        assert assistant_payload["content"] == "The answer is 42."

    def test_empty_history_gives_system_only(self):
        payload = build_payload([], "Only me.")
        assert payload == [{"role": "system", "content": "Only me."}]

    def test_long_history_is_trimmed(self, long_history):
        payload = build_payload(long_history, "sys")
        # 1 system + MAX_HISTORY_MESSAGES
        assert len(payload) == 1 + MAX_HISTORY_MESSAGES

    def test_role_and_content_are_forwarded(self, user_message):
        payload = build_payload([user_message], "sys")
        assert payload[1]["role"] == "user"
        assert payload[1]["content"] == "Hello"

# -- _PARSE_THINKING_TAGS --

class TestParseThinkingTags:
    def test_well_formed_tags_are_split(self):
        text = "<think>some reasoning</think>final answer"
        thinking, reply = _parse_thinking_tags(text)
        assert thinking == "some reasoning"
        assert reply == "final answer"

    def test_multiline_thinking_block(self):
        text = "<think>line one\nline two</think>response"
        thinking, reply = _parse_thinking_tags(text)
        assert "line one" in thinking
        assert "line two" in thinking
        assert reply == "response"

    def test_no_tags_returns_empty_thinking(self):
        text = "just a plain response"
        thinking, reply = _parse_thinking_tags(text)
        assert thinking == ""
        assert reply == "just a plain response"

    def test_only_open_tag_no_close(self):
        """Incomplete tags — treat as no tags, return full text as reply."""
        text = "<think>incomplete"
        thinking, reply = _parse_thinking_tags(text)
        assert thinking == ""
        assert reply == text.strip()

    def test_whitespace_is_stripped(self):
        text = "<think>  spaced  </think>  answer  "
        thinking, reply = _parse_thinking_tags(text)
        assert thinking == "spaced"
        assert reply == "answer"

    def test_empty_think_block(self):
        text = "<think></think>response"
        thinking, reply = _parse_thinking_tags(text)
        assert thinking == ""
        assert reply == "response"

    def test_empty_string_input(self):
        thinking, reply = _parse_thinking_tags("")
        assert thinking == ""
        assert reply == ""

# -- _STREAMPLACEHOLDERS --

class TestStreamPlaceholders:
    def _make_placeholders(self, mock_placeholder):
        reply_ph = MagicMock()
        thinking_ph = MagicMock()
        return _StreamPlaceholders(reply_ph, thinking_ph), reply_ph, thinking_ph

    def test_update_reply_with_cursor(self, mock_placeholder):
        ph = _StreamPlaceholders(mock_placeholder)
        ph.update_reply("hello", cursor=True)
        assert mock_placeholder.last == "hello▌"

    def test_update_reply_without_cursor(self, mock_placeholder):
        ph = _StreamPlaceholders(mock_placeholder)
        ph.update_reply("hello", cursor=False)
        assert mock_placeholder.last == "hello"

    def test_update_thinking_with_cursor(self, mock_placeholder):
        thinking_ph = MagicMock()
        ph = _StreamPlaceholders(MagicMock(), thinking_ph)
        ph.update_thinking("reasoning", cursor=True)
        thinking_ph.markdown.assert_called_once_with("reasoning▌")

    def test_update_thinking_no_op_when_none(self, mock_placeholder):
        """No exception when thinking_placeholder is None."""
        ph = _StreamPlaceholders(mock_placeholder, None)
        ph.update_thinking("reasoning", cursor=True)  # should not raise

    def test_update_thinking_without_cursor(self):
        thinking_ph = MagicMock()
        ph = _StreamPlaceholders(MagicMock(), thinking_ph)
        ph.update_thinking("reasoning", cursor=False)
        thinking_ph.markdown.assert_called_once_with("reasoning")

# -- _HANDLE_FIELD_CHUNK --

class TestHandleFieldChunk:
    def _make_chunk(self, content="", thinking=""):
        chunk = MagicMock()
        chunk.message.content = content
        chunk.message.thinking = thinking
        return chunk

    def test_content_chunk_updates_reply(self):
        reply_ph = MagicMock()
        ph = _StreamPlaceholders(reply_ph, MagicMock())
        chunk = self._make_chunk(content="hello")
        ft, fr = _handle_field_chunk(chunk, "", "", ph)
        assert fr == "hello"
        reply_ph.markdown.assert_called_with("hello▌")

    def test_thinking_chunk_updates_thinking(self):
        thinking_ph = MagicMock()
        ph = _StreamPlaceholders(MagicMock(), thinking_ph)
        chunk = self._make_chunk(thinking="step 1")
        ft, fr = _handle_field_chunk(chunk, "", "", ph)
        assert ft == "step 1"
        thinking_ph.markdown.assert_called_with("step 1▌")

    def test_both_fields_accumulate(self):
        # _handle_field_chunk(chunk, full_thinking, full_response, ph) -> (thinking, response)
        ph = _StreamPlaceholders(MagicMock(), MagicMock())
        chunk = self._make_chunk(content=" world", thinking=" more")
        ft, fr = _handle_field_chunk(chunk, "step 1", "hello", ph)
        assert ft == "step 1 more"
        assert fr == "hello world"

    def test_empty_chunk_leaves_state_unchanged(self):
        # Signature: _handle_field_chunk(chunk, full_thinking, full_response, ph)
        ph = _StreamPlaceholders(MagicMock(), MagicMock())
        chunk = self._make_chunk(content="", thinking="")
        ft, fr = _handle_field_chunk(chunk, "existing_think", "existing", ph)
        assert ft == "existing_think"
        assert fr == "existing"

# -- _HANDLE_TAGS_CHUNK --

class TestHandleTagsChunk:
    def _make_chunk(self, content):
        chunk = MagicMock()
        chunk.message.content = content
        return chunk

    def test_inside_think_block_shows_raw_thinking(self):
        thinking_ph = MagicMock()
        ph = _StreamPlaceholders(MagicMock(), thinking_ph)
        chunk = self._make_chunk("<think>reasoning so far")
        ft, fr = _handle_tags_chunk(chunk, "", "", ph)
        # Thinking placeholder should show content minus the open tag
        thinking_ph.markdown.assert_called()
        assert "reasoning so far" in thinking_ph.markdown.call_args[0][0]

    def test_after_close_tag_reply_is_shown(self):
        reply_ph = MagicMock()
        thinking_ph = MagicMock()
        ph = _StreamPlaceholders(reply_ph, thinking_ph)
        chunk = self._make_chunk("<think>thought</think>answer")
        ft, fr = _handle_tags_chunk(chunk, "", "", ph)
        reply_ph.markdown.assert_called()
        # Reply placeholder should have been updated with the clean answer
        last_reply_call = reply_ph.markdown.call_args[0][0]
        assert "answer" in last_reply_call

    def test_accumulates_content_across_calls(self):
        ph = _StreamPlaceholders(MagicMock(), MagicMock())
        chunk1 = self._make_chunk("<think>part one ")
        _, fr1 = _handle_tags_chunk(chunk1, "", "", ph)
        chunk2 = self._make_chunk("part two</think>reply")
        ft2, fr2 = _handle_tags_chunk(chunk2, "", fr1, ph)
        assert "part one" in fr2
        assert "part two" in fr2