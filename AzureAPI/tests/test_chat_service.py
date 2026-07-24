from unittest.mock import MagicMock, patch

import pytest
import requests

from chat_service import (
    AzureOpenAIRequestError,
    AzureOpenAIResponseError,
    build_payload,
    build_token_params,
    build_url,
    call_azure_openai,
    can_delete_last_turn,
    can_regenerate,
    delete_last_turn,
    drop_last_assistant,
    export_chat_json,
    export_chat_markdown,
    extract_assistant_content,
    extract_stream_delta_content,
    format_azure_http_error,
    get_default_messages,
    get_last_assistant_content,
    get_message_count,
    get_missing_fields,
    iter_sse_content_deltas,
    parse_sse_data_line,
    stream_azure_openai,
    sync_system_prompt,
    trim_messages_for_api,
)
from config import (
    DEPLOYMENT_OPTIONS,
    UNRELIABLE_DEPLOYMENTS,
    format_deployment_label,
    resolve_default_deployment_index,
)


class TestBuildUrl:
    def test_strips_trailing_slash_and_whitespace(self):
        url = build_url(
            endpoint=" https://example.com/v1/ ",
            deployment="sbd-gpt-5.2",
            api_version=" 2024-06-01",
        )
        assert url == (
            "https://example.com/v1/openai/deployments/"
            "sbd-gpt-5.2/chat/completions?api-version=2024-06-01"
        )


class TestBuildTokenParams:
    def test_gpt5_uses_max_completion_tokens(self):
        assert build_token_params("sbd-gpt-5.1", 512) == {"max_completion_tokens": 512}

    def test_gpt5_mini_uses_max_completion_tokens(self):
        assert build_token_params("sbd-gpt-5-mini", 256) == {"max_completion_tokens": 256}

    def test_non_gpt5_uses_max_tokens(self):
        assert build_token_params("sbd-gpt-5.2", 1024) == {"max_tokens": 1024}


class TestBuildPayload:
    def test_merges_messages_temperature_and_token_params(self):
        messages = [{"role": "user", "content": "hi"}]
        payload = build_payload(messages, "sbd-gpt-5.2", 100, 0.2)
        assert payload == {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 100,
        }

    def test_trims_history_when_limit_set(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        payload = build_payload(
            messages,
            "sbd-gpt-5.2",
            100,
            0.2,
            max_history_messages=2,
        )
        assert payload["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_stream_flag(self):
        payload = build_payload(
            [{"role": "user", "content": "hi"}],
            "sbd-gpt-5.2",
            100,
            0.2,
            stream=True,
        )
        assert payload["stream"] is True


class TestExtractAssistantContent:
    def test_happy_path(self):
        data = {"choices": [{"message": {"content": "你好"}}]}
        assert extract_assistant_content(data) == "你好"

    def test_empty_choices(self):
        with pytest.raises(AzureOpenAIResponseError, match="choices"):
            extract_assistant_content({"choices": []})

    def test_missing_content(self):
        with pytest.raises(AzureOpenAIResponseError, match="content"):
            extract_assistant_content({"choices": [{"message": {}}]})

    def test_malformed_body(self):
        with pytest.raises(AzureOpenAIResponseError, match="無法解析"):
            extract_assistant_content({})


class TestCallAzureOpenAI:
    @patch("chat_service.requests.post")
    def test_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        result = call_azure_openai(
            url="https://example.com/chat",
            api_key=" secret ",
            payload={"messages": []},
            deployment="sbd-gpt-5.2",
        )

        assert result == "ok"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["api-key"] == "secret"
        assert kwargs["stream"] is False

    @patch("chat_service.requests.post")
    def test_http_error_is_formatted(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.json.return_value = {
            "error": {"message": "DeploymentNotFound", "code": "404"}
        }
        mock_response.text = "not found"
        mock_post.return_value = mock_response

        with pytest.raises(AzureOpenAIRequestError, match="HTTP 404") as exc_info:
            call_azure_openai(
                "https://example.com",
                "key",
                {},
                deployment="sbd-gpt-5-mini",
                max_retries=0,
            )

        assert exc_info.value.status_code == 404
        assert "sbd-gpt-5-mini" in str(exc_info.value)
        assert "DeploymentNotFound" in str(exc_info.value)

    @patch("chat_service.time.sleep")
    @patch("chat_service.requests.post")
    def test_retries_on_429_then_succeeds(self, mock_post, mock_sleep):
        transient = MagicMock()
        transient.status_code = 429
        transient.reason = "Too Many Requests"
        transient.json.return_value = {"error": {"message": "Rate limited"}}
        transient.text = "rate"

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            "choices": [{"message": {"content": "ok after retry"}}]
        }

        mock_post.side_effect = [transient, success]

        result = call_azure_openai(
            "https://example.com",
            "key",
            {},
            deployment="sbd-gpt-5.2",
            max_retries=2,
            backoff_seconds=0.01,
        )

        assert result == "ok after retry"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()
        transient.close.assert_called_once()


class TestStreaming:
    def test_parse_sse_and_delta(self):
        assert parse_sse_data_line("data: [DONE]") == {}
        assert parse_sse_data_line(": comment") is None
        data = parse_sse_data_line(
            'data: {"choices":[{"delta":{"content":"你好"}}]}'
        )
        assert extract_stream_delta_content(data) == "你好"

    def test_iter_sse_content_deltas_and_stop(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"choices":[{"delta":{"content":"B"}}]}',
            "data: [DONE]",
        ]
        assert list(iter_sse_content_deltas(iter(lines))) == ["A", "B"]

        stop_lines = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"choices":[{"delta":{"content":"B"}}]}',
        ]
        stopped = {"n": 0}

        def should_stop():
            stopped["n"] += 1
            return stopped["n"] > 1

        assert list(iter_sse_content_deltas(iter(stop_lines), should_stop=should_stop)) == ["A"]

    @patch("chat_service.requests.post")
    def test_stream_azure_openai_yields_tokens(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":"!"}}]}',
                "data: [DONE]",
            ]
        )
        mock_post.return_value = mock_response

        chunks = list(
            stream_azure_openai(
                url="https://example.com",
                api_key="key",
                payload={"messages": []},
                deployment="sbd-gpt-5.2",
                max_retries=0,
            )
        )
        assert chunks == ["Hello", "!"]
        _, kwargs = mock_post.call_args
        assert kwargs["stream"] is True
        assert kwargs["json"]["stream"] is True
        mock_response.close.assert_called()

    @patch("chat_service.requests.post")
    def test_stream_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        with pytest.raises(AzureOpenAIRequestError, match="連線逾時"):
            list(
                stream_azure_openai(
                    "https://example.com",
                    "key",
                    {},
                    deployment="sbd-gpt-5.2",
                    max_retries=0,
                )
            )


class TestFormatAzureHttpError:
    def test_includes_status_deployment_and_message(self):
        response = MagicMock()
        response.status_code = 401
        response.reason = "Unauthorized"
        response.json.return_value = {"error": {"message": "Access denied"}}
        response.text = ""
        text = format_azure_http_error(response, deployment="sbd-gpt-5.2")
        assert "HTTP 401" in text
        assert "sbd-gpt-5.2" in text
        assert "Access denied" in text


class TestGetMissingFields:
    def test_all_present(self):
        assert get_missing_fields("https://x", "2024-06-01", "key", "dep") == []

    def test_reports_each_blank_field(self):
        missing = get_missing_fields(" ", "", "  ", "")
        assert "AZURE_OPENAI_ENDPOINT" in missing
        assert "AZURE_OPENAI_API_VERSION" in missing
        assert "AZURE_OPENAI_API_KEY(.env)" in missing
        assert "AZURE_OPENAI_DEPLOYMENT" in missing


class TestMessageHelpers:
    def test_default_messages_include_system(self):
        msgs = get_default_messages("系統提示")
        assert msgs == [{"role": "system", "content": "系統提示"}]

    def test_message_count_skips_system(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        assert get_message_count(messages) == 2

    def test_sync_system_prompt_updates_first_system_message(self):
        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "hi"},
        ]
        updated = sync_system_prompt(messages, "new prompt")
        assert updated[0]["content"] == "new prompt"
        assert updated[1]["content"] == "hi"

    def test_trim_messages_unlimited_when_zero(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        assert trim_messages_for_api(messages, 0) == messages

    def test_trim_messages_keeps_newest(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        trimmed = trim_messages_for_api(messages, 1)
        assert trimmed == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u2"},
        ]

    def test_delete_last_turn_removes_user_and_assistant(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        updated = delete_last_turn(messages)
        assert get_message_count(updated) == 2
        assert updated[-1]["content"] == "a1"

    def test_delete_last_turn_handles_orphan_user(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "only"},
        ]
        updated = delete_last_turn(messages)
        assert updated == [{"role": "system", "content": "s"}]

    def test_regenerate_helpers(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert can_regenerate(messages) is True
        assert can_delete_last_turn(messages) is True
        assert drop_last_assistant(messages)[-1]["role"] == "user"
        assert get_last_assistant_content(messages) == "a"

    def test_cannot_regenerate_without_assistant(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
        ]
        assert can_regenerate(messages) is False

    def test_export_markdown_and_json(self):
        messages = [
            {"role": "system", "content": "系統"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"},
        ]
        md = export_chat_markdown(messages)
        assert "## 使用者" in md
        assert "你好" in md
        assert "## Assistant" in md

        raw = export_chat_json(messages)
        assert '"role": "user"' in raw
        assert "exported_at" in raw


class TestConfigHelpers:
    def test_resolve_known_default_deployment(self):
        idx = resolve_default_deployment_index(
            deployment_options=DEPLOYMENT_OPTIONS,
            default_deployment="sbd-gpt-5.2",
        )
        assert idx == DEPLOYMENT_OPTIONS.index("sbd-gpt-5.2")

    def test_resolve_unknown_default_falls_back_to_zero(self):
        idx = resolve_default_deployment_index(
            deployment_options=DEPLOYMENT_OPTIONS,
            default_deployment="does-not-exist",
        )
        assert idx == 0

    def test_unreliable_deployments_are_labeled(self):
        for name in UNRELIABLE_DEPLOYMENTS:
            label = format_deployment_label(name)
            assert "可能無法使用" in label
            assert name in label

    def test_reliable_deployment_label_unchanged(self):
        assert format_deployment_label("sbd-gpt-5.2") == "sbd-gpt-5.2"
