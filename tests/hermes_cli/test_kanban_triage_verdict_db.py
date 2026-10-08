"""Tests for kb.apply_triage_verdict — the DB-layer triage verdict used by
kanban_triage_assign (mk-minimax): assignee mk-duzy|mk-szybki, status ready.
LLM-free by design. Mirrors test_kanban_specify_db.py conventions.

NOTE (spec mismatch, 2026-10-05): acceptance criteria in t_804f81b9 expect
status ``todo`` after triage; the implementation lands on ``ready``
(hermes_cli/kanban_db.py, UPDATE ... SET status = 'ready'). These tests
assert the implemented behavior (ready) — see card comments by mk-minimax.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", assignee=None):
    return kb.create_task(conn, title=title, assignee=assignee, triage=True)


def test_verdict_promotes_triage_to_ready(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="worker card", assignee="mk-duzy")
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.get_task(conn, tid).assignee == "mk-minimax"
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        ok = kb.apply_triage_verdict(conn, tid, "mk-duzy", author="triage-bot")
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.assignee == "mk-duzy"


def test_verdict_accepts_only_work_profiles(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    for bad in ("mk-minimax", "researcher-x", ""):
        with kbc.connect() as conn, pytest.raises(ValueError):
            kb.apply_triage_verdict(conn, tid, bad, author="triage-bot")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_verdict_requires_own_env_task(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        other = _create_triage(conn, title="other")
    monkeypatch.setenv("HERMES_KANBAN_TASK", other)
    with kbc.connect() as conn, pytest.raises(PermissionError):
        kb.apply_triage_verdict(conn, tid, "mk-duzy", author="triage-bot")
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"


def test_verdict_picks_up_blocked_card(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        ok = kb.apply_triage_verdict(conn, tid, "mk-szybki", author="triage-bot")
    assert ok is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert task.assignee == "mk-szybki"


def test_verdict_blocked_refused_when_not_allowed(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        ok = kb.apply_triage_verdict(
            conn, tid, "mk-szybki", author="triage-bot", allow_blocked=False
        )
    assert ok is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_verdict_ignores_card_not_owned_by_triage(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain card")  # not triage=True
        conn.execute("UPDATE tasks SET assignee = 'mk-duzy' WHERE id = ?", (tid,))
        conn.commit()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with kbc.connect() as conn:
        ok = kb.apply_triage_verdict(conn, tid, "mk-szybki", author="triage-bot")
    assert ok is False
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.assignee == "mk-duzy"
    assert task.status != "ready"
