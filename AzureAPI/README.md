# Azure OpenAI Chat (API test UI)

Streamlit app for smoke-testing Azure OpenAI chat deployments. Dark theme, multi-session chat, streaming replies, and helpers for everyday API checks.

## Features

- Select Azure OpenAI deployment (with per-model notes; unreliable ones labeled)
- Default deployment / API version from `.env`
- Streaming replies (`st.write_stream` + Azure SSE)
- Multi named chat sessions (create / rename / delete / switch)
- Editable system prompt (with safe reset)
- History window limit for outbound context
- Regenerate last reply / delete last turn / clear chat
- Export Markdown or JSON; per-reply download & copy helpers
- Quick templates (summarize, JSON, rewrite, debug) and sample prompts
- Rough usage chip (messages / chars / tokens)
- Retries with backoff on transient 429 / 5xx
- Clearer HTTP errors (status + deployment + Azure message)
- Best-effort stop generation
- Demo-mode banner when API key is missing
- Optional password gate (`APP_ACCESS_PASSWORD`)
- Unit tests with `pytest`

## Requirements

- Python 3.12+ recommended
- Azure OpenAI endpoint, API key, and deployed model names

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and fill in your values:

```env
AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_ENDPOINT=https://YOUR_ENDPOINT.EXAMPLE.com
AZURE_OPENAI_API_VERSION=RESOURCE_DEPLOYED_DATE
AZURE_OPENAI_DEFAULT_DEPLOYMENT=GPT_DEPLOTMENT_NAME

```

### Deployments in the sidebar

- `sbd-gpt-5-mini` / `sbd-gpt-5-nano` (may be unavailable / 404 on this endpoint)
- `sbd-gpt-5.1` / `sbd-gpt-5.2` / `sbd-gpt-5.4`

GPT-5* deployments use `max_completion_tokens`; others use `max_tokens`.

## Run

```powershell
streamlit run app.py
```

Theme is forced dark via `.streamlit/config.toml`.

## Test

```powershell
pytest
```

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit entry / orchestration |
| `ui.py` | Sidebar, chat UI, styles |
| `chat_service.py` | Azure URL/payload/stream/retry helpers |
| `sessions.py` | In-memory multi-session helpers |
| `config.py` | Env, models, templates, notes |
| `.streamlit/config.toml` | Dark theme |
| `.env.example` | Env template |
| `tests/` | Unit tests |
| `requirements.txt` | Pinned dependencies |

## Notes

- API key stays in `.env` (never commit `.env`).
- Chat history lives in Streamlit `session_state` (cleared when the browser session ends).
- “停止產生” is best-effort under Streamlit’s execution model.
