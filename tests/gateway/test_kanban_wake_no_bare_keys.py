"""A bare i18n key must never reach a delivered notification.

When a locale catalog is missing/incomplete, ``t()`` falls back to the key itself.
For chat-facing strings that leaks a raw ``gateway.*`` token into the message the
user reads. ``_t_or`` substitutes an English sentence instead.
"""

import agent.i18n as i18n
from gateway.kanban_watchers_notifier import _t_or


def test_t_or_substitutes_fallback_on_catalog_miss(monkeypatch):
    monkeypatch.setattr(i18n, "get_language", lambda: "en")
    i18n._catalog_cache[(i18n._current_home(), "en")] = {}  # simulate a missing/empty configured catalog
    assert _t_or("gateway.kanban.wake.message", "FALLBACK {task_id}", task_id="t_1") == "FALLBACK t_1"
    assert _t_or("nope.key", "plain sentence") == "plain sentence"
    # A present key still resolves to the catalog sentence.
    i18n._catalog_cache.pop((i18n._current_home(), "en"), None)
    assert _t_or("gateway.kanban.wake.completed", "FALLBACK") == "completed"


def test_wake_text_uses_sentences_when_catalog_empty(monkeypatch):
    """The kanban wake builder must emit sentences, never bare keys."""
    monkeypatch.setattr(i18n, "get_language", lambda: "en")
    i18n._catalog_cache[(i18n._current_home(), "en")] = {}

    from gateway.kanban_watchers_notifier import _KanbanNotification

    class Sub(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    class Task:
        title = "T"
        assignee = "mk-szybki"
        status = "done"
        result = None
        session_id = "s"

    class Ev:
        kind = "completed"
        payload = {"summary": "Wynik gotowy."}
        id = 1

    sub = Sub(task_id="t_x", platform="telegram", chat_id="1", thread_id="",
              delivery_mode="notify+wake", notifier_profile="mk", last_ping_event_id=0)
    n = _KanbanNotification.__new__(_KanbanNotification)
    n.d = {"sub": sub, "task": Task(), "events": [Ev()]}
    n.sub, n.task, n.title = sub, Task(), "T"
    n.board_slug, n.task_id, n.sub_profile = None, "t_x", "mk"
    n.board_tag, n.head = "", "Kanban t_x"
    n.wake_agent, n.is_push_adapter = True, True
    n.wake_handoff = n.wake_review_detail = n.session_key = n.synth = ""
    for ev in n.d["events"]:
        n.format_event(ev)
    n.build_wake_text()
    assert "gateway.kanban.wake" not in n.synth
    assert "t_x" in n.synth and "completed" in n.synth
