# Tests for models.py — thinking-mode detection and options builder.
#
# get_available_models() is not tested here because it calls Ollama and st.cache_resource,
# both of which require integration-level setup. build_options() and get_thinking_mode()
# are pure logic and fully unit-testable.

import pytest
from unittest.mock import patch, MagicMock

from Model import build_options, get_thinking_mode

# -- GET_THINKING_MODE --

class TestGetThinkingMode:
    def test_gemma_returns_field(self):
        assert get_thinking_mode("gemma") == "field"

    def test_gemma_versioned_returns_field(self):
        assert get_thinking_mode("gemma3:27b") == "field"

    def test_gemma_uppercase_returns_field(self):
        assert get_thinking_mode("GEMMA3") == "field"

    def test_unknown_model_returns_none(self):
        assert get_thinking_mode("llama3.2") is None

    def test_empty_string_returns_none(self):
        assert get_thinking_mode("") is None

    def test_partial_prefix_match_returns_none(self):
        """'gem' should not match 'gemma'."""
        assert get_thinking_mode("gem") is None

    def test_model_with_gemma_in_middle_returns_none(self):
        """Only prefix matching — 'mygemma' should not match."""
        assert get_thinking_mode("mygemma") is None

# -- BUILD_OPTIONS - VALID INPUTS --

class TestBuildOptionsValid:
    def _base_kwargs(self, **overrides):
        defaults = dict(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            seed=None,
            enable_thinking=False,
            thinking_mode=None,
        )
        return {**defaults, **overrides}

    def test_basic_options_are_set(self):
        opts = build_options(**self._base_kwargs())
        assert opts["temperature"] == 0.7
        assert opts["top_p"] == 0.9
        assert opts["num_predict"] == 1024

    def test_seed_included_when_provided(self):
        opts = build_options(**self._base_kwargs(seed=42))
        assert opts["seed"] == 42

    def test_seed_absent_when_none(self):
        opts = build_options(**self._base_kwargs(seed=None))
        assert "seed" not in opts

    def test_thinking_flag_set_for_field_mode(self):
        opts = build_options(**self._base_kwargs(
            enable_thinking=True, thinking_mode="field"
        ))
        assert opts.get("thinking") is True

    def test_thinking_flag_absent_when_disabled(self):
        opts = build_options(**self._base_kwargs(
            enable_thinking=False, thinking_mode="field"
        ))
        assert "thinking" not in opts

    def test_thinking_flag_absent_for_tags_mode(self):
        """Tags-mode models don't use an options flag."""
        opts = build_options(**self._base_kwargs(
            enable_thinking=True, thinking_mode="tags"
        ))
        assert "thinking" not in opts

    def test_boundary_temperature_min(self):
        opts = build_options(**self._base_kwargs(temperature=0.0))
        assert opts["temperature"] == 0.0

    def test_boundary_temperature_max(self):
        opts = build_options(**self._base_kwargs(temperature=2.0))
        assert opts["temperature"] == 2.0

    def test_boundary_top_p_min(self):
        opts = build_options(**self._base_kwargs(top_p=0.0))
        assert opts["top_p"] == 0.0

    def test_boundary_top_p_max(self):
        opts = build_options(**self._base_kwargs(top_p=1.0))
        assert opts["top_p"] == 1.0

    def test_boundary_max_tokens_min(self):
        opts = build_options(**self._base_kwargs(max_tokens=64))
        assert opts["num_predict"] == 64

    def test_boundary_max_tokens_max(self):
        opts = build_options(**self._base_kwargs(max_tokens=4096))
        assert opts["num_predict"] == 4096

    def test_negative_seed_is_accepted(self):
        """Negative seeds are valid (e.g. -1 used as 'random' sentinel upstream)."""
        opts = build_options(**self._base_kwargs(seed=-1))
        assert opts["seed"] == -1

# -- BUILD_OPTIONS - INVALID INPUTS RAISE ValueError --

class TestBuildOptionsInvalid:
    def _base_kwargs(self, **overrides):
        defaults = dict(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            seed=None,
            enable_thinking=False,
            thinking_mode=None,
        )
        return {**defaults, **overrides}

    def test_temperature_too_low_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            build_options(**self._base_kwargs(temperature=-0.1))

    def test_temperature_too_high_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            build_options(**self._base_kwargs(temperature=2.1))

    def test_top_p_too_low_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            build_options(**self._base_kwargs(top_p=-0.1))

    def test_top_p_too_high_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            build_options(**self._base_kwargs(top_p=1.1))

    def test_max_tokens_too_low_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            build_options(**self._base_kwargs(max_tokens=63))

    def test_max_tokens_too_high_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            build_options(**self._base_kwargs(max_tokens=4097))