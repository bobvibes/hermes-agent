"""Drain the kanban triage column before any new ready work is spawned.

Called from the gateway dispatcher. A free slot routes one triage card
(batch size = free slots). Ready work waits until triage is empty.
Blocked cards are reassessed every 12 hours.
"""
from __future__ import annotations

import time
from pathlib import Path

STAMP = Path("/home/hermes/mk-data/tmp/kanban-blocked-sweep.stamp")
SWEEP_SECONDS = 12 * 3600


def _comments(conn, task_id: str) -> str:
    rows = conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return "\n".join((r[0] or "") for r in rows)


def _do_not_resume(text: str) -> bool:
    t = text.lower()
    return "nie wznawia" in t or "zastąpione paczkami" in t or "zastapione paczkami" in t


def _is_human_wait(title: str, blocked_payloads: str) -> bool:
    if (title or "").lower().startswith("whatsapp"):
        return True
    return "initial_status" in blocked_payloads and '"actor": "mk"' in blocked_payloads


def _assignee_for(title: str, body: str) -> str:
    blob = f"{title or ''}\n{body or ''}".lower()
    if "weryfik" in blob or "zweryfikuj" in blob:
        return "mk-szybki"
    if "autor" in blob or "pozycje kolejki" in blob:
        return "mk-duzy"
    return "mk-szybki"


def _blocked_payloads(conn, task_id: str) -> str:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 5",
        (task_id,),
    ).fetchall()
    return "\n".join((r[0] or "") for r in rows)


def park_blocked_into_triage(conn, task_ids: list[str] | None = None) -> list[str]:
    """Move blocked cards to triage, clear assignee, reset the failure counter."""
    from hermes_cli.kanban_db import _append_event, add_comment
    from hermes_cli.kanban_db_connect import write_txn

    if task_ids is None:
        rows = conn.execute("SELECT id FROM tasks WHERE status = 'blocked'").fetchall()
        task_ids = [r[0] for r in rows]
    moved = []
    for task_id in task_ids:
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'triage', assignee = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "worker_started_at = NULL, consecutive_failures = 0 "
                "WHERE id = ? AND status = 'blocked'",
                (task_id,),
            )
            if cur.rowcount != 1:
                continue
            _append_event(conn, task_id, "triaged", {
                "source": "triage-drain",
                "reason": "blocked card returned to triage for assignment",
            })
        add_comment(
            conn, task_id, "triage-drain",
            "Wracam z blocked do triage. Assignee wyczyszczony, licznik porażek wyzerowany. "
            "Wolny slot przypisze kartkę albo odłoży ją, jeśli ma czekać na człowieka.",
        )
        moved.append(task_id)
    return moved


def _archive(conn, task_id: str, why: str) -> None:
    from hermes_cli.kanban_db import add_comment, archive_task

    add_comment(conn, task_id, "triage-drain", why)
    archive_task(conn, task_id)


def _park_human(conn, task_id: str, why: str) -> None:
    from hermes_cli.kanban_db import _append_event, add_comment
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'blocked', assignee = NULL, "
            "block_kind = 'needs_input', block_recurrences = 0, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status = 'triage'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return
        _append_event(conn, task_id, "blocked", {
            "kind": "needs_input", "reason": why, "source": "triage-drain",
        })
    add_comment(conn, task_id, "triage-drain", why)


def _promote(conn, task_id: str, assignee: str) -> None:
    from hermes_cli.kanban_db import specify_triage_task

    conn.execute(
        "UPDATE tasks SET consecutive_failures = 0 WHERE id = ? AND status = 'triage'",
        (task_id,),
    )
    specify_triage_task(conn, task_id, assignee=assignee, author="triage-drain")


def drain_triage(conn, limit: int) -> int:
    """Route up to ``limit`` triage cards. Returns how many triage cards remain."""
    if limit is None or limit < 1:
        row = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'triage'").fetchone()
        return int(row[0])
    rows = conn.execute(
        "SELECT id, title, body FROM tasks WHERE status = 'triage' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC LIMIT ?",
        (int(limit),),
    ).fetchall()
    for row in rows:
        task_id, title, body = row[0], row[1] or "", row[2] or ""
        comments = _comments(conn, task_id)
        if _do_not_resume(comments):
            _archive(
                conn, task_id,
                "Nie wznawiam. W komentarzu jest, że paczka została zastąpiona albo kartki nie wolno wznawiać. Archiwum.",
            )
            continue
        if _is_human_wait(title, _blocked_payloads(conn, task_id)):
            _park_human(
                conn, task_id,
                "To czeka na Michała, nie na agenta. Zostawiam jako blocked/needs_input, poza triage, "
                "żeby nie blokowało kolejki.",
            )
            continue
        # Leave it in triage. mk-minimax claims and reassigns. Do not
        # keyword-promote onto a work profile.
        continue
    return 0


def sweep_blocked(conn, *, force: bool = False) -> list[str]:
    """Every 12h, move blocked cards that are not a human wait back to triage."""
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if not force and STAMP.exists() and now - STAMP.stat().st_mtime < SWEEP_SECONDS:
        return []
    rows = conn.execute(
        "SELECT id, title, COALESCE(block_kind, '') FROM tasks WHERE status = 'blocked'"
    ).fetchall()
    chosen = []
    for task_id, title, kind in rows:
        if kind == "needs_input":
            continue
        if _is_human_wait(title or "", _blocked_payloads(conn, task_id)):
            continue
        if _do_not_resume(_comments(conn, task_id)):
            continue
        chosen.append(task_id)
    moved = park_blocked_into_triage(conn, chosen) if chosen else []
    STAMP.write_text(str(int(now)))
    return moved
