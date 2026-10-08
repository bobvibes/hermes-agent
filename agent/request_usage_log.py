"""Append-only API request log. No prompt text, no headers, no secrets."""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

DB_PATH = "/home/hermes/mk-data/usage/requests.db"
_CARD = re.compile(r"^t_[0-9a-fA-F]{6,}$")
_RETENTION = timedelta(days=7)
_EXTRA = (
    ("prompt_tokens", "INTEGER"),
    ("output_tokens", "INTEGER"),
    ("total_tokens", "INTEGER"),
    ("duration_s", "REAL"),
)


def source_for(task_id, platform) -> str:
    """Card id when the worker owns one; else telegram for a Telegram turn."""
    env_tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    tid = str(task_id or "").strip()
    for cand in (env_tid, tid):
        if _CARD.match(cand):
            return cand
    if str(platform or "").strip().lower() == "telegram":
        return "telegram"
    if env_tid.startswith("t_"):
        return env_tid
    if tid.startswith("t_"):
        return tid
    plat = str(platform or "").strip().lower()
    return plat or "other"


def _as_status(status_code):
    if status_code is None or status_code == "":
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_duration(value):
    if value is None or value == "":
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def status_from_response(response):
    seen = set()
    stack = [response]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        ident = id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        code = getattr(obj, "status_code", None)
        parsed = _as_status(code)
        if parsed is not None:
            return parsed
        for attr in ("response", "_raw_response", "http_response", "raw"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                stack.append(child)
    # post_api_request runs only after a parsed HTTP response.
    return 200 if response is not None else None


def _outcome(status_code, failed: bool) -> str:
    if status_code == 429:
        return "429"
    if failed or (isinstance(status_code, int) and status_code >= 400):
        return "error"
    return "ok"


def _usage_numbers(agent, response):
    summary = None
    try:
        fn = getattr(agent, "_usage_summary_for_api_request_hook", None)
        if fn is not None:
            summary = fn(response)
    except Exception:
        summary = None
    if not isinstance(summary, dict):
        return None, None, None
    prompt = _as_int(summary.get("prompt_tokens"))
    if prompt is None:
        prompt = _as_int(summary.get("input_tokens"))
    output = _as_int(summary.get("output_tokens"))
    if output is None:
        output = _as_int(summary.get("completion_tokens"))
    total = _as_int(summary.get("total_tokens"))
    if total is None and prompt is not None and output is not None:
        total = prompt + output
    return prompt, output, total


def _drop_foreign_sidecars() -> None:
    """Empty WAL/SHM owned by another user blocks this logger (other has no write).

    Drop them only when the WAL has no frames, then the next connect recreates
    both as the current user. A non-empty WAL is left alone.
    """
    wal = DB_PATH + "-wal"
    shm = DB_PATH + "-shm"
    blocked = False
    for path in (wal, shm):
        if os.path.exists(path) and not os.access(path, os.W_OK):
            blocked = True
    if not blocked:
        return
    wal_size = os.path.getsize(wal) if os.path.exists(wal) else 0
    if wal_size != 0:
        return
    for path in (wal, shm):
        try:
            os.remove(path)
        except OSError:
            pass


def _chmod_db() -> None:
    for path in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"):
        try:
            if os.path.exists(path) and os.access(path, os.W_OK):
                os.chmod(path, 0o664)
        except OSError:
            pass


def log_request(
    *,
    task_id,
    platform,
    model,
    status_code,
    failed: bool,
    prompt_tokens=None,
    output_tokens=None,
    total_tokens=None,
    duration_s=None,
) -> None:
    try:
        status = _as_status(status_code)
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_for(task_id, platform),
            (str(model).strip() if model else "") or None,
            status,
            _outcome(status, failed),
            _as_int(prompt_tokens),
            _as_int(output_tokens),
            _as_int(total_tokens),
            _as_duration(duration_s),
        )
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _drop_foreign_sidecars()
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS requests ("
                "timestamp TEXT NOT NULL, source TEXT NOT NULL, model TEXT, "
                "status_code INTEGER, outcome TEXT NOT NULL)"
            )
            have = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
            for name, decl in _EXTRA:
                if name not in have:
                    conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {decl}")
            cutoff = (datetime.now(timezone.utc) - _RETENTION).isoformat(timespec="seconds")
            conn.execute("DELETE FROM requests WHERE timestamp < ?", (cutoff,))
            conn.execute(
                "INSERT INTO requests ("
                "timestamp, source, model, status_code, outcome, "
                "prompt_tokens, output_tokens, total_tokens, duration_s) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()
        finally:
            conn.close()
        _chmod_db()
    except Exception:
        return


def log_api_response(agent, response, *, task_id, duration_s=None) -> None:
    model = getattr(response, "model", None) or getattr(agent, "model", None)
    prompt, output, total = _usage_numbers(agent, response)
    log_request(
        task_id=task_id,
        platform=getattr(agent, "platform", None),
        model=model,
        status_code=status_from_response(response),
        failed=False,
        prompt_tokens=prompt,
        output_tokens=output,
        total_tokens=total,
        duration_s=duration_s,
    )


def log_api_error(*, task_id, platform, model, status_code, duration_s=None) -> None:
    log_request(
        task_id=task_id,
        platform=platform,
        model=model,
        status_code=status_code,
        failed=True,
        duration_s=duration_s,
    )
