from chat_service import (
    estimate_message_chars,
    estimate_tokens_rough,
    format_usage_summary,
    get_default_messages,
)
from config import get_deployment_note, PROMPT_TEMPLATES
from sessions import (
    create_session,
    delete_session,
    ensure_default_session,
    list_session_labels,
    rename_session,
    save_active_session,
)


class TestSessions:
    def test_create_and_list_sessions(self):
        session = create_session(title="測試對話", system_prompt="助手", messages=[])
        store = {session["id"]: session}
        labels = list_session_labels(store)
        assert labels[0][0] == session["id"]
        assert labels[0][1] == "測試對話"

    def test_ensure_default_session_bootstraps(self):
        messages = get_default_messages("系統")
        store, active = ensure_default_session(
            None,
            None,
            system_prompt="系統",
            messages=messages,
        )
        assert active in store
        assert store[active]["messages"] == messages

    def test_save_rename_delete(self):
        session = create_session(title="A", system_prompt="s", messages=[])
        store = {session["id"]: session}
        store = save_active_session(
            store,
            session["id"],
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="s2",
        )
        assert store[session["id"]]["messages"][0]["content"] == "hi"

        store = rename_session(store, session["id"], "新名稱")
        assert store[session["id"]]["title"] == "新名稱"

        other = create_session(title="B", system_prompt="s", messages=[])
        store[other["id"]] = other
        store = delete_session(store, session["id"])
        assert session["id"] not in store
        assert other["id"] in store


class TestUsageEstimates:
    def test_estimate_and_format(self):
        messages = [
            {"role": "system", "content": "abcd"},
            {"role": "user", "content": "你好世界"},
        ]
        assert estimate_message_chars(messages) == 8
        assert estimate_tokens_rough(messages) == 4
        summary = format_usage_summary(messages)
        assert "訊息 1" in summary
        assert "tokens" in summary


class TestTeachingConfig:
    def test_templates_exist(self):
        ids = {item["id"] for item in PROMPT_TEMPLATES}
        assert {"summarize", "json", "rewrite", "debug"} <= ids

    def test_deployment_note_known_and_fallback(self):
        note = get_deployment_note("sbd-gpt-5.2")
        assert note["status"] == "建議"
        unknown = get_deployment_note("does-not-exist")
        assert "尚無" in unknown["note"]
