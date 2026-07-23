# Tests for config.py — sanity-check that constants are self-consistent.
#
# These are not unit tests of logic — they guard against accidental misconfiguration,
# such as setting WARNING_THRESHOLD above MAX_HISTORY or inverting a range tuple.

import pytest
import Config

class TestConfigSanity:
    def test_max_history_is_positive(self):
        assert Config.MAX_HISTORY_MESSAGES > 0

    def test_warning_threshold_below_max_history(self):
        assert Config.HISTORY_WARNING_THRESHOLD < Config.MAX_HISTORY_MESSAGES

    def test_temperature_range_is_ordered(self):
        lo, hi = Config.TEMPERATURE_RANGE
        assert lo < hi

    def test_top_p_range_is_ordered(self):
        lo, hi = Config.TOP_P_RANGE
        assert lo < hi

    def test_max_tokens_range_is_ordered(self):
        lo, hi = Config.MAX_TOKENS_RANGE
        assert lo < hi

    def test_default_temperature_within_range(self):
        lo, hi = Config.TEMPERATURE_RANGE
        assert lo <= Config.DEFAULT_TEMPERATURE <= hi

    def test_default_top_p_within_range(self):
        lo, hi = Config.TOP_P_RANGE
        assert lo <= Config.DEFAULT_TOP_P <= hi

    def test_default_max_tokens_within_range(self):
        lo, hi = Config.MAX_TOKENS_RANGE
        assert lo <= Config.DEFAULT_MAX_TOKENS <= hi

    def test_default_system_prompt_is_non_empty(self):
        assert Config.DEFAULT_SYSTEM_PROMPT.strip() != ""

    def test_thinking_models_have_required_keys(self):
        for name, meta in Config.THINKING_MODELS.items():
            assert "thinking_mode" in meta, f"{name} missing 'thinking_mode'"
            assert "max_context_tokens" in meta, f"{name} missing 'max_context_tokens'"

    def test_thinking_modes_are_valid_literals(self):
        valid = {"field", "tags"}
        for name, meta in Config.THINKING_MODELS.items():
            assert meta["thinking_mode"] in valid, (
                f"{name} has unknown thinking_mode '{meta['thinking_mode']}'"
            )

    def test_thinking_model_keys_are_lowercase(self):
        """Prefix matching is case-insensitive only on the input side;
        registry keys should be lowercase to avoid duplicates."""
        for key in Config.THINKING_MODELS:
            assert key == key.lower(), f"Registry key '{key}' should be lowercase"