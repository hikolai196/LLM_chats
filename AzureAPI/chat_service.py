import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone

import requests

from config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    RETRYABLE_STATUS_CODES,
)


class AzureOpenAIResponseError(ValueError):
    """Raised when the Azure response JSON is missing expected fields."""


class AzureOpenAIRequestError(Exception):
    """Raised for HTTP/network failures talking to Azure OpenAI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        deployment: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.deployment = deployment


class GenerationCancelled(Exception):
    """Raised when the caller asks to stop an in-flight stream."""


def get_default_messages(system_prompt: str) -> list:
    return [{"role": "system", "content": system_prompt}]


def get_message_count(messages: list) -> int:
    return sum(1 for msg in messages if msg["role"] != "system")


def estimate_message_chars(messages: list, *, include_system: bool = True) -> int:
    total = 0
    for msg in messages:
        if not include_system and msg.get("role") == "system":
            continue
        total += len(str(msg.get("content") or ""))
    return total


def estimate_tokens_rough(messages: list, *, include_system: bool = True) -> int:
    """Rough token estimate for awareness only (CJK-ish: ~1 token per 2 chars)."""
    chars = estimate_message_chars(messages, include_system=include_system)
    return max(0, (chars + 1) // 2)


def format_usage_summary(messages: list) -> str:
    count = get_message_count(messages)
    chars = estimate_message_chars(messages, include_system=True)
    tokens = estimate_tokens_rough(messages, include_system=True)
    return f"訊息 {count}｜約 {chars} 字｜約 {tokens} tokens（粗估）"


def sync_system_prompt(messages: list, system_prompt: str) -> list:
    """Ensure the first system message matches the current prompt."""
    updated = [dict(msg) for msg in messages]
    prompt = system_prompt.strip() or system_prompt
    if updated and updated[0].get("role") == "system":
        updated[0]["content"] = prompt
        return updated
    return [{"role": "system", "content": prompt}, *updated]


def trim_messages_for_api(messages: list, max_non_system: int) -> list:
    """Keep system message(s) plus the newest non-system messages.

    max_non_system <= 0 means no trimming.
    """
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    other_messages = [msg for msg in messages if msg.get("role") != "system"]
    if max_non_system <= 0 or len(other_messages) <= max_non_system:
        return [*system_messages, *other_messages]
    return [*system_messages, *other_messages[-max_non_system:]]


def delete_last_turn(messages: list) -> list:
    """Remove the last user/assistant exchange (assistant first if present)."""
    updated = [dict(msg) for msg in messages]
    if len(updated) <= 1:
        return updated

    if updated[-1].get("role") == "assistant":
        updated.pop()
    if len(updated) > 1 and updated[-1].get("role") == "user":
        updated.pop()
    return updated


def can_regenerate(messages: list) -> bool:
    """True when the last visible turn is an assistant reply to a user message."""
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    if len(non_system) < 2:
        return False
    return non_system[-1].get("role") == "assistant" and non_system[-2].get("role") == "user"


def drop_last_assistant(messages: list) -> list:
    """Remove the trailing assistant message so regenerate can re-ask."""
    updated = [dict(msg) for msg in messages]
    if updated and updated[-1].get("role") == "assistant":
        updated.pop()
    return updated


def can_delete_last_turn(messages: list) -> bool:
    return get_message_count(messages) > 0


def get_last_assistant_content(messages: list) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content")
    return None


def export_chat_markdown(messages: list) -> str:
    lines = ["# 對話匯出", ""]
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            lines.extend(["## System", content, ""])
        elif role == "user":
            lines.extend(["## 使用者", content, ""])
        elif role == "assistant":
            lines.extend(["## Assistant", content, ""])
    return "\n".join(lines).rstrip() + "\n"


def export_chat_json(messages: list) -> str:
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_url(endpoint: str, deployment: str, api_version: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    return (
        f"{endpoint}/openai/deployments/"
        f"{deployment}/chat/completions"
        f"?api-version={api_version.strip()}"
    )


def build_token_params(deployment: str, max_output_tokens: int) -> dict:
    deployment_lower = deployment.lower()
    if "gpt-5" in deployment_lower:
        return {"max_completion_tokens": max_output_tokens}
    return {"max_tokens": max_output_tokens}


def build_payload(
    messages: list,
    deployment: str,
    max_output_tokens: int,
    temperature: float,
    max_history_messages: int = 0,
    stream: bool = False,
) -> dict:
    token_params = build_token_params(deployment, max_output_tokens)
    api_messages = trim_messages_for_api(messages, max_history_messages)
    payload = {
        "messages": api_messages,
        "temperature": temperature,
        **token_params,
    }
    if stream:
        payload["stream"] = True
    return payload


def extract_assistant_content(data: dict) -> str:
    """Pull assistant text from a chat completions JSON body."""
    try:
        choices = data["choices"]
        if not choices:
            raise AzureOpenAIResponseError("回應缺少 choices。")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None or content == "":
            raise AzureOpenAIResponseError("回應缺少 message.content。")
        return content
    except (KeyError, TypeError, IndexError) as exc:
        raise AzureOpenAIResponseError("無法解析模型回應格式。") from exc


def extract_stream_delta_content(data: dict) -> str | None:
    """Pull incremental content from a streaming chat chunk."""
    try:
        choices = data.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content is None or content == "":
            return None
        return content
    except (TypeError, IndexError, AttributeError):
        return None


def parse_sse_data_line(line: str) -> dict | None:
    """Parse one SSE line. Returns {} for [DONE], dict for JSON, None to skip."""
    text = line.strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if text == "[DONE]":
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AzureOpenAIResponseError(f"無法解析串流資料：{text[:120]}") from exc


def format_azure_http_error(
    response: requests.Response,
    *,
    deployment: str = "",
) -> str:
    """Build a readable error string from an Azure HTTP response."""
    status = response.status_code
    detail = ""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            detail = error.get("message") or error.get("code") or str(error)
        elif body:
            detail = str(body)
    except Exception:
        detail = (response.text or "").strip()

    parts = [f"HTTP {status}"]
    if deployment:
        parts.append(f"deployment={deployment}")
    if detail:
        parts.append(str(detail))
    else:
        parts.append(response.reason or "請求失敗")
    return "｜".join(parts)


def _post_azure(
    url: str,
    api_key: str,
    payload: dict,
    *,
    stream: bool,
    timeout: int,
    max_retries: int,
    backoff_seconds: float,
    deployment: str,
) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key.strip(),
    }
    attempts = max(0, max_retries) + 1
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=stream,
            )
        except requests.exceptions.Timeout as exc:
            last_error = AzureOpenAIRequestError(
                f"連線逾時（第 {attempt + 1}/{attempts} 次）｜deployment={deployment}｜{exc}",
                deployment=deployment,
            )
            if attempt >= attempts - 1:
                raise last_error from exc
            time.sleep(backoff_seconds * (2**attempt))
            continue
        except requests.exceptions.RequestException as exc:
            last_error = AzureOpenAIRequestError(
                f"連線錯誤｜deployment={deployment}｜{exc}",
                deployment=deployment,
            )
            if attempt >= attempts - 1:
                raise last_error from exc
            time.sleep(backoff_seconds * (2**attempt))
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts - 1:
            # Consume/close before retrying so connections are not leaked.
            response.close()
            time.sleep(backoff_seconds * (2**attempt))
            continue

        if response.status_code >= 400:
            message = format_azure_http_error(response, deployment=deployment)
            response.close()
            raise AzureOpenAIRequestError(
                message,
                status_code=response.status_code,
                deployment=deployment,
            )

        return response

    raise last_error or AzureOpenAIRequestError(
        f"請求失敗｜deployment={deployment}",
        deployment=deployment,
    )


def call_azure_openai(
    url: str,
    api_key: str,
    payload: dict,
    *,
    deployment: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    timeout: int = 60,
) -> str:
    response = _post_azure(
        url,
        api_key,
        payload,
        stream=False,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        deployment=deployment,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise AzureOpenAIResponseError("無法解析模型回應 JSON。") from exc
    return extract_assistant_content(data)


def iter_sse_content_deltas(
    lines: Iterator[str],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Yield content deltas from an SSE line iterator."""
    for raw_line in lines:
        if should_stop and should_stop():
            return
        parsed = parse_sse_data_line(raw_line)
        if parsed is None:
            continue
        if parsed == {}:
            break
        delta = extract_stream_delta_content(parsed)
        if delta:
            yield delta


def stream_azure_openai(
    url: str,
    api_key: str,
    payload: dict,
    *,
    deployment: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    timeout: int = 120,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Stream assistant tokens from Azure OpenAI chat completions."""
    stream_payload = {**payload, "stream": True}
    response = _post_azure(
        url,
        api_key,
        stream_payload,
        stream=True,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        deployment=deployment,
    )

    try:
        lines = response.iter_lines(decode_unicode=True)
        yielded_any = False
        for chunk in iter_sse_content_deltas(lines, should_stop=should_stop):
            yielded_any = True
            yield chunk
        if not yielded_any and not (should_stop and should_stop()):
            raise AzureOpenAIResponseError("串流回應沒有可顯示的內容。")
    finally:
        response.close()


def get_missing_fields(endpoint: str, api_version: str, api_key: str, deployment: str) -> list:
    missing_fields = []

    if not endpoint.strip():
        missing_fields.append("AZURE_OPENAI_ENDPOINT")
    if not api_version.strip():
        missing_fields.append("AZURE_OPENAI_API_VERSION")
    if not api_key.strip():
        missing_fields.append("AZURE_OPENAI_API_KEY(.env)")
    if not deployment.strip():
        missing_fields.append("AZURE_OPENAI_DEPLOYMENT")

    return missing_fields
