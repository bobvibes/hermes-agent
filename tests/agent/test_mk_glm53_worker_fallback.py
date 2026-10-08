"""mk 08.10 (decyzja Michała): w workerze kanbana mk-duzy (glm-5.3) po 429 spada na MiniMax jak dziś;
poza workerem (gateway, cron) reguła glm-5.3 działa jak dziś (zostaje po 429, nie schodzi na MiniMax)."""
import types

import pytest

from agent import chat_completion_helpers as cch


class Reached(Exception):
    pass


def _agent(model="glm-5.3"):
    a = types.SimpleNamespace()
    a.model = model
    a._fallback_index = 0
    a._fallback_chain = [{"provider": "minimax", "model": "MiniMax-M3"}]
    a._unavailable_fallback_keys = set()
    return a


RATE = types.SimpleNamespace(value="rate_limit")


@pytest.fixture
def seen(monkeypatch):
    import agent.fallback_cooldown as fc
    monkeypatch.setattr(fc, "switch_deferred_by_reset", lambda *a, **k: False)
    monkeypatch.setattr(fc, "_arm_rate_limit_cooldown", lambda *a, **k: 0)
    monkeypatch.setattr(cch, "_fallback_chain_exhausted", lambda agent, reason: "EXHAUSTED")
    got = []

    def candidate(agent, fb, key, prov, model, unavailable):
        got.append(model)
        raise Reached()

    monkeypatch.setattr(cch, "_should_skip_fallback_candidate", candidate)
    return got


def test_worker_glm53_429_falls_back_to_minimax(monkeypatch, seen):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    with pytest.raises(Reached):
        cch.try_activate_fallback(_agent(), RATE)
    assert seen == ["MiniMax-M3"]


def test_worker_glm53_other_reason_reaches_minimax(monkeypatch, seen):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    with pytest.raises(Reached):
        cch.try_activate_fallback(_agent(), None)
    assert seen == ["MiniMax-M3"]


def test_gateway_glm53_429_stays_on_glm53(monkeypatch, seen):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cch.try_activate_fallback(_agent(), RATE) is False
    assert seen == []


def test_gateway_glm53_never_takes_minimax(monkeypatch, seen):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert cch.try_activate_fallback(_agent(), None) == "EXHAUSTED"
    assert seen == []


@pytest.mark.parametrize("worker", [True, False])
def test_other_models_unaffected(monkeypatch, seen, worker):
    if worker:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with pytest.raises(Reached):
        cch.try_activate_fallback(_agent("glm-5.3-flash"), RATE)
    assert seen == ["MiniMax-M3"]
