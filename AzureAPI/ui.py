import html

import streamlit as st

from sessions import list_session_labels


def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Keep accents compatible with Streamlit dark theme; do not force light backgrounds. */
        .hero-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: inherit;
            margin-bottom: 0.15rem;
        }
        .hero-subtitle {
            color: rgba(250, 250, 250, 0.65);
            font-size: 0.95rem;
            margin-bottom: 0.9rem;
        }
        .mini-card {
            padding: 10px 12px;
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 10px;
            background-color: rgba(250, 250, 250, 0.04);
            margin-bottom: 8px;
        }
        .mini-label {
            font-size: 0.78rem;
            color: rgba(250, 250, 250, 0.55);
            margin-bottom: 4px;
            line-height: 1.2;
        }
        .mini-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: inherit;
            line-height: 1.3;
            word-break: break-word;
        }
        .empty-hint {
            color: rgba(250, 250, 250, 0.65);
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        }
        .model-note {
            border-left: 3px solid #4f8cff;
            background: rgba(79, 140, 255, 0.12);
            padding: 0.65rem 0.8rem;
            border-radius: 0 10px 10px 0;
            color: inherit;
            font-size: 0.9rem;
            margin: 0.4rem 0 0.8rem 0;
        }
        .usage-chip {
            display: inline-block;
            background: rgba(250, 250, 250, 0.06);
            color: inherit;
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.82rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_mini_info(label: str, value: str):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{safe_label}</div>
            <div class="mini-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_access_gate(expected_password: str) -> bool:
    """Return True when access is allowed."""
    if not expected_password:
        return True
    if st.session_state.get("access_granted"):
        return True

    st.markdown('<div class="hero-title">Azure OpenAI Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">此環境已啟用簡易保護，請輸入通行碼後繼續。</div>',
        unsafe_allow_html=True,
    )
    password = st.text_input("通行碼", type="password")
    if st.button("進入", type="primary"):
        if password == expected_password:
            st.session_state.access_granted = True
            st.rerun()
        st.error("通行碼不正確。")
    return False


def render_demo_mode_banner(missing_fields: list[str]):
    if "AZURE_OPENAI_API_KEY(.env)" not in missing_fields:
        return
    st.info(
        "目前為示範模式：可以瀏覽介面與切換對話，但尚未設定 API Key，"
        "因此無法呼叫模型。請在 `.env` 填入 `AZURE_OPENAI_API_KEY`。"
    )


def render_sidebar(
    default_endpoint,
    default_api_version,
    deployment_options,
    default_max_output_tokens,
    default_temperature,
    default_system_prompt,
    default_max_history_messages,
    sessions: dict,
    active_session_id: str,
    deployment_note: dict | None = None,
    get_deployment_note=None,
    default_deployment_index=0,
    format_deployment_label=None,
    can_regenerate=False,
    can_delete_last_turn=False,
    export_markdown="",
    export_json="",
    usage_summary="",
):
    with st.sidebar:
        st.header("對話工作區")
        session_controls = _render_session_controls(sessions, active_session_id)

        st.markdown("---")
        st.header("模型設定")

        azure_deployment, max_output_tokens, temperature, max_history_messages = (
            _render_model_controls(
                deployment_options=deployment_options,
                default_deployment_index=default_deployment_index,
                format_deployment_label=format_deployment_label,
                default_max_output_tokens=default_max_output_tokens,
                default_temperature=default_temperature,
                default_max_history_messages=default_max_history_messages,
            )
        )

        if callable(get_deployment_note):
            render_model_note(get_deployment_note(azure_deployment))
        elif deployment_note:
            render_model_note(deployment_note)

        system_prompt = _render_system_prompt_controls(default_system_prompt)

        with st.expander("進階設定", expanded=False):
            azure_endpoint = st.text_input(
                "Endpoint",
                value=default_endpoint,
                help="例如：https://your-resource-name.openai.azure.com",
            )
            azure_api_version = st.text_input(
                "API version",
                value=default_api_version,
            )
            st.caption("API Key 由 .env 載入")

        st.markdown("---")
        st.subheader("對話操作")

        regenerate_clicked = st.button(
            "重新產生最後回覆",
            use_container_width=True,
            disabled=not can_regenerate,
        )
        delete_last_clicked = st.button(
            "刪除上一輪對話",
            use_container_width=True,
            disabled=not can_delete_last_turn,
        )
        clear_clicked = st.button("清除目前對話", use_container_width=True)

        st.markdown("---")
        render_stop_generation_button()

        st.markdown("---")
        st.subheader("匯出目前對話")
        st.download_button(
            "匯出 Markdown",
            data=export_markdown,
            file_name="chat_export.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not export_markdown.strip(),
        )
        st.download_button(
            "匯出 JSON",
            data=export_json,
            file_name="chat_export.json",
            mime="application/json",
            use_container_width=True,
            disabled=not export_json.strip(),
        )

        message_count_placeholder = st.empty()
        if usage_summary:
            message_count_placeholder.markdown(
                f'<div class="usage-chip">{html.escape(usage_summary)}</div>',
                unsafe_allow_html=True,
            )

    return {
        "azure_endpoint": azure_endpoint,
        "azure_api_version": azure_api_version,
        "azure_deployment": azure_deployment,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "max_history_messages": max_history_messages,
        "system_prompt": system_prompt,
        "regenerate_clicked": regenerate_clicked,
        "delete_last_clicked": delete_last_clicked,
        "clear_clicked": clear_clicked,
        "message_count_placeholder": message_count_placeholder,
        **session_controls,
    }


def _render_session_controls(sessions: dict, active_session_id: str) -> dict:
    labels = list_session_labels(sessions)
    ids = [item[0] for item in labels]
    titles = [item[1] for item in labels]
    current_index = ids.index(active_session_id) if active_session_id in ids else 0

    selected_title = st.selectbox(
        "目前對話",
        titles,
        index=current_index,
        key="session_select_title",
    )
    selected_id = ids[titles.index(selected_title)]

    new_title = st.text_input(
        "對話名稱",
        value=sessions.get(selected_id, {}).get("title", ""),
        key=f"session_title_input_{selected_id}",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        create_clicked = st.button("新建", use_container_width=True)
    with col2:
        rename_clicked = st.button("重新命名", use_container_width=True)
    with col3:
        delete_clicked = st.button(
            "刪除",
            use_container_width=True,
            disabled=len(sessions) <= 1,
        )

    return {
        "selected_session_id": selected_id,
        "session_title_draft": new_title,
        "create_session_clicked": create_clicked,
        "rename_session_clicked": rename_clicked,
        "delete_session_clicked": delete_clicked,
    }


def render_model_note(note: dict):
    status = html.escape(str(note.get("status", "")))
    token_param = html.escape(str(note.get("token_param", "")))
    temperature = html.escape(str(note.get("temperature", "")))
    detail = html.escape(str(note.get("note", "")))
    st.markdown(
        f"""
        <div class="model-note">
            <strong>模型備註｜{status}</strong><br/>
            Token 參數：{token_param}　｜　溫度：{temperature}<br/>
            {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_controls(
    deployment_options,
    default_deployment_index,
    format_deployment_label,
    default_max_output_tokens,
    default_temperature,
    default_max_history_messages,
):
    label_fn = format_deployment_label or (lambda name: name)
    deployment_labels = [label_fn(name) for name in deployment_options]
    selected_label = st.selectbox(
        "Model deployment",
        deployment_labels,
        index=default_deployment_index,
    )
    azure_deployment = deployment_options[deployment_labels.index(selected_label)]

    max_output_tokens = st.number_input(
        "Max Output Tokens",
        min_value=1,
        max_value=8192,
        value=default_max_output_tokens,
        step=100,
    )

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=default_temperature,
        step=0.1,
    )

    max_history_messages = st.number_input(
        "送出歷史訊息上限",
        min_value=0,
        max_value=200,
        value=default_max_history_messages,
        step=2,
        help="僅送出最近 N 則非 system 訊息給模型。0 表示不限制。",
    )

    return azure_deployment, max_output_tokens, temperature, int(max_history_messages)


def _reset_system_prompt(default_system_prompt: str):
    # Must run via on_click so the value is set before the text_area widget binds.
    st.session_state.system_prompt = default_system_prompt


def _render_system_prompt_controls(default_system_prompt: str):
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = default_system_prompt

    system_prompt = st.text_area(
        "系統提示（System Prompt）",
        key="system_prompt",
        height=100,
    )
    st.button(
        "重設系統提示",
        use_container_width=True,
        on_click=_reset_system_prompt,
        args=(default_system_prompt,),
    )
    return system_prompt


def render_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="hero-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="hero-subtitle">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )


@st.fragment
def render_stop_generation_button():
    """Best-effort stop control; fragment can update while the main script streams."""
    generating = bool(st.session_state.get("is_generating"))
    st.caption("產生中可點此嘗試停止" if generating else "停止目前產生（盡力而為）")
    if st.button(
        "停止產生",
        use_container_width=True,
        key="stop_generation_btn",
    ):
        st.session_state.cancel_generation = True
        st.warning("已送出停止請求…")


def render_top_info(azure_deployment: str, azure_api_version: str, token_param_name: str, temperature: float):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_mini_info("Deployment", azure_deployment)
    with col2:
        render_mini_info("API Version", azure_api_version)
    with col3:
        render_mini_info("Token Param", token_param_name)
    with col4:
        render_mini_info("Temperature", str(temperature))


def render_advanced_api_url(url: str):
    with st.expander("進階：API URL", expanded=False):
        render_mini_info("API URL", url)


def render_prompt_templates(templates: list[dict]) -> str | None:
    st.markdown("**快捷模板**")
    selected = None
    cols = st.columns(min(4, max(1, len(templates))))
    for index, template in enumerate(templates):
        with cols[index % len(cols)]:
            if st.button(
                template["label"],
                key=f"template_{template['id']}",
                use_container_width=True,
                help=template["prompt"][:80],
            ):
                selected = template["prompt"]
    return selected


def render_empty_state(sample_questions: list[str]) -> str | None:
    st.markdown(
        '<p class="empty-hint">還沒有對話。可以直接輸入問題，點快捷模板，或選下方範例開始：</p>',
        unsafe_allow_html=True,
    )
    selected = None
    cols = st.columns(min(3, max(1, len(sample_questions))))
    for index, question in enumerate(sample_questions):
        with cols[index % len(cols)]:
            if st.button(question, key=f"sample_q_{index}", use_container_width=True):
                selected = question
    return selected


def render_chat_history(messages: list):
    assistant_index = 0
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        with st.chat_message(role):
            content = msg.get("content", "")
            st.markdown(content)
            if role == "assistant":
                st.download_button(
                    label="下載此則回覆",
                    data=content,
                    file_name=f"assistant_reply_{assistant_index + 1}.txt",
                    mime="text/plain",
                    key=f"download_assistant_{assistant_index}",
                )
                with st.expander("方便複製的純文字", expanded=False):
                    st.text_area(
                        "assistant_copy",
                        value=content,
                        height=120,
                        label_visibility="collapsed",
                        key=f"copy_assistant_{assistant_index}",
                    )
                assistant_index += 1


def render_last_reply_copy_box(content: str | None):
    if not content:
        return
    with st.expander("最後一則回覆（全選複製）", expanded=False):
        st.text_area(
            "last_reply_copy",
            value=content,
            height=160,
            label_visibility="collapsed",
            key="last_reply_copy_box",
        )
