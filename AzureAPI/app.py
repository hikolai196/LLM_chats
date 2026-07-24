import streamlit as st

from config import (
    SYSTEM_PROMPT,
    SAMPLE_QUESTIONS,
    PROMPT_TEMPLATES,
    DEPLOYMENT_OPTIONS,
    DEFAULT_API_VERSION,
    DEFAULT_API_KEY,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_HISTORY_MESSAGES,
    APP_ACCESS_PASSWORD,
    APP_TITLE,
    APP_SUBTITLE,
    resolve_default_deployment_index,
    format_deployment_label,
    get_deployment_note,
)
from chat_service import (
    get_default_messages,
    get_message_count,
    sync_system_prompt,
    build_url,
    build_token_params,
    build_payload,
    stream_azure_openai,
    get_missing_fields,
    AzureOpenAIResponseError,
    AzureOpenAIRequestError,
    can_regenerate,
    can_delete_last_turn,
    delete_last_turn,
    drop_last_assistant,
    export_chat_markdown,
    export_chat_json,
    get_last_assistant_content,
    format_usage_summary,
)
from sessions import (
    create_session,
    ensure_default_session,
    save_active_session,
    rename_session,
    delete_session,
)
from ui import (
    apply_custom_styles,
    render_sidebar,
    render_header,
    render_top_info,
    render_advanced_api_url,
    render_chat_history,
    render_empty_state,
    render_last_reply_copy_box,
    render_prompt_templates,
    render_access_gate,
    render_demo_mode_banner,
)

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
apply_custom_styles()

if not render_access_gate(APP_ACCESS_PASSWORD):
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = get_default_messages(SYSTEM_PROMPT)
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "pending_regenerate" not in st.session_state:
    st.session_state.pending_regenerate = False
if "awaiting_stream" not in st.session_state:
    st.session_state.awaiting_stream = False
if "cancel_generation" not in st.session_state:
    st.session_state.cancel_generation = False
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "template_draft" not in st.session_state:
    st.session_state.template_draft = ""

# Apply deferred system_prompt updates BEFORE the text_area widget is created.
if "pending_system_prompt" in st.session_state:
    st.session_state.system_prompt = st.session_state.pop("pending_system_prompt")

# Apply deferred template draft updates BEFORE the draft text_area is created.
if "pending_template_draft" in st.session_state:
    st.session_state.template_draft = st.session_state.pop("pending_template_draft")

# Bootstrap multi-session store from current messages on first run.
if "chat_sessions" not in st.session_state or "active_session_id" not in st.session_state:
    sessions, active_id = ensure_default_session(
        None,
        None,
        system_prompt=st.session_state.get("system_prompt", SYSTEM_PROMPT),
        messages=st.session_state.messages,
    )
    st.session_state.chat_sessions = sessions
    st.session_state.active_session_id = active_id


def _should_stop_generation() -> bool:
    return bool(st.session_state.get("cancel_generation"))


def _queue_system_prompt(value: str):
    """Queue a system_prompt change for the next run (before widget bind)."""
    st.session_state.pending_system_prompt = value


def _persist_active_session(system_prompt: str | None = None):
    prompt = system_prompt if system_prompt is not None else st.session_state.system_prompt
    st.session_state.chat_sessions = save_active_session(
        st.session_state.chat_sessions,
        st.session_state.active_session_id,
        messages=st.session_state.messages,
        system_prompt=prompt,
    )


def _switch_session(target_id: str, current_system_prompt: str):
    if target_id == st.session_state.active_session_id:
        return
    _persist_active_session(current_system_prompt)
    target = st.session_state.chat_sessions.get(target_id)
    if not target:
        return
    st.session_state.active_session_id = target_id
    st.session_state.messages = list(target.get("messages") or get_default_messages(SYSTEM_PROMPT))
    _queue_system_prompt(target.get("system_prompt") or SYSTEM_PROMPT)
    st.session_state.pending_prompt = None
    st.session_state.pending_regenerate = False
    st.session_state.awaiting_stream = False


def _stream_assistant_reply(
    *,
    url: str,
    api_key: str,
    messages: list,
    deployment: str,
    max_output_tokens: int,
    temperature: float,
    max_history_messages: int,
) -> str | None:
    payload = build_payload(
        messages=messages,
        deployment=deployment,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        max_history_messages=max_history_messages,
        stream=True,
    )
    st.session_state.cancel_generation = False
    st.session_state.is_generating = True
    try:
        with st.chat_message("assistant"):
            st.caption("串流回覆中…（可點側邊欄「停止產生」嘗試中斷）")
            try:
                answer = st.write_stream(
                    stream_azure_openai(
                        url=url,
                        api_key=api_key,
                        payload=payload,
                        deployment=deployment,
                        should_stop=_should_stop_generation,
                    )
                )
            except AzureOpenAIRequestError as e:
                st.error(str(e))
                return None
            except AzureOpenAIResponseError as e:
                st.error(f"回應解析錯誤：{e}")
                return None
            except Exception as e:
                st.error(f"發生未預期錯誤：{e}")
                return None

        if st.session_state.get("cancel_generation"):
            st.info("已停止產生。若已有部分文字，仍會保留在對話中。")

        if isinstance(answer, str) and answer.strip():
            return answer
        return None
    finally:
        st.session_state.is_generating = False


_export_md = export_chat_markdown(st.session_state.messages)
_export_json = export_chat_json(st.session_state.messages)
_usage = format_usage_summary(st.session_state.messages)

sidebar_values = render_sidebar(
    default_endpoint=DEFAULT_ENDPOINT,
    default_api_version=DEFAULT_API_VERSION,
    deployment_options=DEPLOYMENT_OPTIONS,
    default_max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    default_temperature=DEFAULT_TEMPERATURE,
    default_system_prompt=SYSTEM_PROMPT,
    default_max_history_messages=DEFAULT_MAX_HISTORY_MESSAGES,
    sessions=st.session_state.chat_sessions,
    active_session_id=st.session_state.active_session_id,
    get_deployment_note=get_deployment_note,
    default_deployment_index=resolve_default_deployment_index(),
    format_deployment_label=format_deployment_label,
    can_regenerate=can_regenerate(st.session_state.messages),
    can_delete_last_turn=can_delete_last_turn(st.session_state.messages),
    export_markdown=_export_md,
    export_json=_export_json,
    usage_summary=_usage,
)

azure_endpoint = sidebar_values["azure_endpoint"]
azure_api_version = sidebar_values["azure_api_version"]
azure_deployment = sidebar_values["azure_deployment"]
max_output_tokens = sidebar_values["max_output_tokens"]
temperature = sidebar_values["temperature"]
max_history_messages = sidebar_values["max_history_messages"]
system_prompt = sidebar_values["system_prompt"]
regenerate_clicked = sidebar_values["regenerate_clicked"]
delete_last_clicked = sidebar_values["delete_last_clicked"]
clear_clicked = sidebar_values["clear_clicked"]
selected_session_id = sidebar_values["selected_session_id"]
session_title_draft = sidebar_values["session_title_draft"]
create_session_clicked = sidebar_values["create_session_clicked"]
rename_session_clicked = sidebar_values["rename_session_clicked"]
delete_session_clicked = sidebar_values["delete_session_clicked"]

st.session_state.messages = sync_system_prompt(st.session_state.messages, system_prompt)

# Session management actions.
if create_session_clicked:
    _persist_active_session(system_prompt)
    new_session = create_session(
        title=f"對話 {len(st.session_state.chat_sessions) + 1}",
        system_prompt=SYSTEM_PROMPT,
        messages=get_default_messages(SYSTEM_PROMPT),
    )
    st.session_state.chat_sessions[new_session["id"]] = new_session
    st.session_state.active_session_id = new_session["id"]
    st.session_state.messages = list(new_session["messages"])
    _queue_system_prompt(SYSTEM_PROMPT)
    st.rerun()

if rename_session_clicked:
    st.session_state.chat_sessions = rename_session(
        st.session_state.chat_sessions,
        st.session_state.active_session_id,
        session_title_draft,
    )
    st.rerun()

if delete_session_clicked and len(st.session_state.chat_sessions) > 1:
    _persist_active_session(system_prompt)
    current_id = st.session_state.active_session_id
    st.session_state.chat_sessions = delete_session(st.session_state.chat_sessions, current_id)
    next_id = next(iter(st.session_state.chat_sessions.keys()))
    st.session_state.active_session_id = next_id
    nxt = st.session_state.chat_sessions[next_id]
    st.session_state.messages = list(nxt.get("messages") or get_default_messages(SYSTEM_PROMPT))
    _queue_system_prompt(nxt.get("system_prompt") or SYSTEM_PROMPT)
    st.rerun()

if selected_session_id != st.session_state.active_session_id:
    _switch_session(selected_session_id, system_prompt)
    st.rerun()

if clear_clicked:
    st.session_state.messages = get_default_messages(system_prompt)
    st.session_state.pending_prompt = None
    st.session_state.pending_regenerate = False
    st.session_state.awaiting_stream = False
    st.session_state.cancel_generation = True
    st.session_state.is_generating = False
    _persist_active_session(system_prompt)
    st.rerun()

if delete_last_clicked and can_delete_last_turn(st.session_state.messages):
    st.session_state.messages = delete_last_turn(st.session_state.messages)
    _persist_active_session(system_prompt)
    st.rerun()

if regenerate_clicked and can_regenerate(st.session_state.messages):
    st.session_state.pending_regenerate = True
    st.rerun()

url = build_url(
    endpoint=azure_endpoint,
    deployment=azure_deployment,
    api_version=azure_api_version,
)

token_param_name = list(build_token_params(azure_deployment, max_output_tokens).keys())[0]

missing_fields = get_missing_fields(
    endpoint=azure_endpoint,
    api_version=azure_api_version,
    api_key=DEFAULT_API_KEY,
    deployment=azure_deployment,
)

render_header(APP_TITLE, APP_SUBTITLE)
render_demo_mode_banner(missing_fields)
render_top_info(
    azure_deployment=azure_deployment,
    azure_api_version=azure_api_version,
    token_param_name=token_param_name,
    temperature=temperature,
)
render_advanced_api_url(url)

if missing_fields and "AZURE_OPENAI_API_KEY(.env)" not in missing_fields:
    st.warning(f"請先完成設定：{', '.join(missing_fields)}")
elif missing_fields and len(missing_fields) > 1:
    st.warning(f"請先完成設定：{', '.join(missing_fields)}")

template_prompt = render_prompt_templates(PROMPT_TEMPLATES)
if template_prompt is not None:
    st.session_state.pending_template_draft = template_prompt
    st.rerun()

# Convert pending actions into a stream job after config is known.
if st.session_state.pending_regenerate:
    st.session_state.pending_regenerate = False
    if missing_fields:
        st.error("Azure OpenAI 設定尚未完成，請先檢查左側欄位與 .env。")
    elif can_regenerate(st.session_state.messages):
        st.session_state.messages = drop_last_assistant(st.session_state.messages)
        st.session_state.awaiting_stream = True

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    if missing_fields:
        st.error("Azure OpenAI 設定尚未完成，請先檢查左側欄位與 .env。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.awaiting_stream = True

visible_count = get_message_count(st.session_state.messages)
if visible_count == 0 and not st.session_state.awaiting_stream:
    sample = render_empty_state(SAMPLE_QUESTIONS)
    if sample:
        st.session_state.pending_prompt = sample
        st.rerun()
else:
    render_chat_history(st.session_state.messages)
    if not st.session_state.awaiting_stream:
        render_last_reply_copy_box(get_last_assistant_content(st.session_state.messages))

if st.session_state.awaiting_stream:
    st.session_state.awaiting_stream = False
    answer = _stream_assistant_reply(
        url=url,
        api_key=DEFAULT_API_KEY,
        messages=st.session_state.messages,
        deployment=azure_deployment,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        max_history_messages=max_history_messages,
    )
    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})
    _persist_active_session(system_prompt)
    st.rerun()

# Show template draft editor when a template was chosen.
if st.session_state.template_draft:
    st.text_area(
        "模板草稿（可修改後送出）",
        key="template_draft",
        height=120,
    )

    def _send_template_draft():
        draft = (st.session_state.get("template_draft") or "").strip()
        if draft:
            st.session_state.pending_prompt = draft
            st.session_state.pending_template_draft = ""

    def _clear_template_draft():
        st.session_state.pending_template_draft = ""

    col_send, col_clear = st.columns(2)
    with col_send:
        st.button(
            "送出模板草稿",
            type="primary",
            use_container_width=True,
            on_click=_send_template_draft,
        )
    with col_clear:
        st.button(
            "清除模板草稿",
            use_container_width=True,
            on_click=_clear_template_draft,
        )

prompt = st.chat_input("請輸入你的問題，或先選上方快捷模板")
if prompt:
    st.session_state.pending_prompt = prompt
    st.session_state.pending_template_draft = ""
    st.rerun()

_persist_active_session(system_prompt)
