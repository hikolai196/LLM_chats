import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant."

APP_TITLE = "Azure OpenAI Chat"
APP_SUBTITLE = "API 測試用對話介面"

SAMPLE_QUESTIONS = [
    "用一句話介紹你自己。",
    "回覆一個簡短的 JSON，包含 name 與 status 兩個欄位。",
    "把「Hello, Azure OpenAI」翻譯成繁體中文。",
]

# Prompt starters for quick API smoke tests.
PROMPT_TEMPLATES = [
    {
        "id": "summarize",
        "label": "摘要",
        "prompt": "請用三點摘要下列內容：\n\n",
    },
    {
        "id": "json",
        "label": "JSON 輸出",
        "prompt": "只回傳 JSON（不要 markdown），格式如下：\n"
        '{"answer": "...", "confidence": 0.0}\n\n問題：',
    },
    {
        "id": "rewrite",
        "label": "改寫",
        "prompt": "請把下面文字改寫得更清楚精簡：\n\n",
    },
    {
        "id": "debug",
        "label": "除錯說明",
        "prompt": "請分析下列錯誤訊息，並給出可能原因與排查步驟：\n\n",
    },
]

# Deployments known to return 404 on this endpoint (kept selectable but labeled in UI).
UNRELIABLE_DEPLOYMENTS = {
    "sbd-gpt-5-mini",
    "sbd-gpt-5-nano",
}

DEPLOYMENT_OPTIONS = [
    "sbd-gpt-5.2",
    "sbd-gpt-5-mini",
    "sbd-gpt-5-nano",
    "sbd-gpt-5.1",
    "sbd-gpt-5.4",
]

DEPLOYMENT_NOTES = {
    "sbd-gpt-5-mini": {
        "token_param": "max_completion_tokens",
        "temperature": "視部署而定",
        "status": "可能無法使用",
        "note": "此環境曾出現 404，建議先選其他模型。",
    },
    "sbd-gpt-5-nano": {
        "token_param": "max_completion_tokens",
        "temperature": "視部署而定",
        "status": "可能無法使用",
        "note": "此環境曾出現 404，建議先選其他模型。",
    },
    "sbd-gpt-5.1": {
        "token_param": "max_completion_tokens",
        "temperature": "視部署而定",
        "status": "可用",
        "note": "較新模型；輸出上限使用 max_completion_tokens。",
    },
    "sbd-gpt-5.2": {
        "token_param": "max_completion_tokens",
        "temperature": "視部署而定",
        "status": "可用",
        "note": "較新模型；可測較長或較複雜回覆。",
    },
    "sbd-gpt-5.4": {
        "token_param": "max_completion_tokens",
        "temperature": "視部署而定",
        "status": "可用",
        "note": "較新模型；可測長文與結構化輸出。",
    },
}

DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEFAULT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource-name.openai.azure.com")
DEFAULT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEFAULT_DEPLOYMENT", DEPLOYMENT_OPTIONS[0])
# Optional gate for shared machines. Empty = no password prompt.
APP_ACCESS_PASSWORD = os.getenv("APP_ACCESS_PASSWORD", "").strip()
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1
# How many non-system messages to send to the API (0 = unlimited).
DEFAULT_MAX_HISTORY_MESSAGES = 20
# Transient Azure failures (429/5xx) retry policy.
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def get_deployment_note(deployment: str) -> dict:
    return DEPLOYMENT_NOTES.get(
        deployment,
        {
            "token_param": "依模型而定",
            "temperature": "依模型而定",
            "status": "未知",
            "note": "尚無此部署的使用備註。",
        },
    )


def resolve_default_deployment_index(
    deployment_options: list[str] | None = None,
    default_deployment: str | None = None,
) -> int:
    """Return selectbox index for the configured default deployment."""
    options = deployment_options if deployment_options is not None else DEPLOYMENT_OPTIONS
    preferred = (default_deployment if default_deployment is not None else DEFAULT_DEPLOYMENT).strip()
    if preferred in options:
        return options.index(preferred)
    return 0


def format_deployment_label(deployment: str) -> str:
    """Label unreliable deployments so users know they may fail."""
    if deployment in UNRELIABLE_DEPLOYMENTS:
        return f"{deployment}（可能無法使用）"
    return deployment
