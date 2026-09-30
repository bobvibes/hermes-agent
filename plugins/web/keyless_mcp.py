"""Keyless web search/extract via public free-tier endpoints (Exa, Parallel, Firecrawl, Keenable).
Resolved strictly LAST — after every keyed backend, the managed gateway, ddgs and custom plugin
providers — so it never pre-empts a deliberate setup. Privacy: no user identifiers are sent;
Parallel gets a random per-process ``session_id`` (rate limiting only) and its optional
``model_name`` analytics field is deliberately omitted. Disable with ``web.keyless_fallback: false``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from plugins.web._common import document as _page, page_error as _page_error, search_fail, search_ok, web_hit as _row

logger = logging.getLogger(__name__)

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
KEENABLE_API_URL = "https://api.keenable.ai"
_KEENABLE_TITLE = "hermes-agent"

# Parallel free-tier rate-limit correlation id — random per process, never persisted.
_SESSION_ID = uuid.uuid4().hex

_TIMEOUT_SECONDS = 30


class KeylessMCPError(RuntimeError):
    """A keyless MCP call failed (transport, rate limit, or tool error)."""


_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "ratelimit", "too many requests", "429", "quota exceeded", "slow down")

# vendor -> (display label, env key, signup URL) for the standard failure hint.
_VENDOR_HINTS = {
    "exa": ("Exa", "EXA_API_KEY", "https://exa.ai"), "parallel": ("Parallel", "PARALLEL_API_KEY", "https://parallel.ai"),
    "firecrawl": ("Firecrawl", "FIRECRAWL_API_KEY", "https://firecrawl.dev"), "keenable": ("Keenable", "KEENABLE_API_KEY", "https://keenable.ai"),
}


def _is_rate_limitish(message: str) -> bool:
    """Heuristic: does an error message look like free-tier throttling?"""
    return any(marker in (message or "").lower() for marker in _RATE_LIMIT_MARKERS)


# A free tier that refuses this client (401/402/403: anonymous access revoked, IP reputation gate)
# or errors server-side (5xx) fails every query from here, while another vendor may still serve it.
# Anchored to the status the error starts with (after the ``Keyless <Vendor> search failed:`` prefix),
# so a terminal error that echoes the query ("HTTP 400: invalid query 'http 503'") does not match.
_VENDOR_REFUSAL_RE = re.compile(
    r"\s*(?:keyless\s+\w+\s+search\s+failed:\s*)?"
    r"(?:http(?:\s+status)?|status(?:\s+code)?|client\s+error|error\s+code)"
    r"\s*[:=']*\s*(?:40[123]|5\d\d)\b",
    re.IGNORECASE,
)


def _is_search_failover_eligible(message: str) -> bool:
    """Return whether another anonymous vendor may serve the search.

    Rate limits and structured HTTP 401/402/403/5xx vendor refusals are local to
    one free search endpoint. Free-text markers are deliberately ignored:
    vendors may echo the query in an otherwise terminal error.
    """
    return _is_rate_limitish(message) or bool(_VENDOR_REFUSAL_RE.match(message or ""))


def _fail_msg(vendor: str, kind: str, exc: Any, *, other_backends: bool = True) -> str:
    label, env_key, site = _VENDOR_HINTS[vendor]
    alt = " or another web backend via `hermes tools`" if other_backends else ""
    return f"Keyless {label} {kind} failed: {exc}. Set {env_key} ({site}){alt} for reliable service."


def _search(vendor: str, rows: Callable[[], List[Dict[str, Any]]], catch: Any = (), fmt: Optional[Callable[[Exception], str]] = None) -> Dict[str, Any]:
    """``search_ok(rows())``; :class:`KeylessMCPError` → standard vendor hint, exception
    types in ``catch`` → ``fmt(exc)``; anything else propagates."""
    try:
        return search_ok(rows())
    except KeylessMCPError as exc:
        return search_fail(_fail_msg(vendor, "search", exc))
    except catch as exc:
        return search_fail(fmt(exc))


def _per_url(urls: List[str], fetch: Callable[[str], Dict[str, Any]], vendor: str, catch: Any = Exception, hint: bool = False) -> List[Dict[str, Any]]:
    """Per-URL extract loop: a ``catch`` failure becomes an error entry (``hint`` adds the ``hermes tools`` hint)."""
    def _one(url: str) -> Dict[str, Any]:
        try:
            return fetch(url)
        except catch as exc:  # noqa: BLE001 — per-URL error entry
            return _page_error(url, _fail_msg(vendor, "extract", exc, other_backends=hint))

    return [_one(u) for u in urls]


# --- Tier / config ------------------------------------------------------------
def keyless_enabled() -> bool:
    """Delegates to the registry so the ``web.keyless_fallback`` (default on) chokepoint lives with backend resolution."""
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — resolver optional in stripped envs
        logger.debug("keyless_enabled(): registry helper unavailable: %s", exc)
        return True


_BACKEND_KEYS = ("backend", "search_backend", "extract_backend")


def _web_config_selects(name: str) -> bool:
    """True when any ``web.backend`` / ``search_backend`` / ``extract_backend`` names *name*."""
    import tools.web_tools as _wt
    web_cfg = _wt._load_web_config()
    return any((web_cfg.get(key) or "").lower().strip() == name for key in _BACKEND_KEYS)


def provider_tier(name: str) -> str:
    """``web.provider_tier.<name>`` (``hermes tools`` Free/Paid rows): ``free``, ``paid``, or ``auto`` (anything else/unset)."""
    try:
        from hermes_cli.config import load_config
        tiers = (load_config().get("web") or {}).get("provider_tier") or {}
        value = str(tiers.get(name, "") or "").lower().strip()
        return value if value in ("free", "paid") else "auto"
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("provider_tier(%r) config read failed: %s", name, exc)
        return "auto"


def use_keyless(name: str, api_key: str) -> bool:
    """Single chokepoint for search + extract: ``free`` → keyless even with a key; ``paid`` → keyed even
    without one (the keyed path raises its usual missing-key error); ``auto`` → keyless only when no key + tier enabled."""
    tier = provider_tier(name)
    if tier in ("free", "paid"):
        return tier == "free"
    return not api_key and keyless_enabled()


# --- MCP transport ------------------------------------------------------------
def _parse_mcp_body(body: str) -> str:
    """First text content item from an MCP tools/call response — plain-JSON bodies
    (Parallel) or SSE ``data: {...}`` lines (Exa). Raises :class:`KeylessMCPError` for
    JSON-RPC errors and ``isError`` tool results (e.g. Exa's free-tier rate limit).

    SSE frames are split on the SSE line terminators (CRLF, CR, LF) only:
    ``str.splitlines()`` also breaks on U+0085 / U+2028 / U+2029, which legitimately
    occur inside CJK page text and would cut a ``data:`` line in two. A parsed envelope with no text is reported as such — it is
    the vendor's answer, not an unrecognized shape."""

    def _from_payload(payload: str) -> Optional[str]:
        payload = payload.strip()
        if not payload.startswith("{"):
            return None
        data = json.loads(payload)
        err = data.get("error")
        if err:
            raise KeylessMCPError(str(err.get("message") or err))
        result = data.get("result") or {}
        texts = [c.get("text", "") for c in result.get("content") or [] if isinstance(c, dict)]
        if result.get("isError"):
            raise KeylessMCPError(" ".join(t for t in texts if t) or "MCP tool call failed")
        return next((str(t) for t in texts if t), "")

    stripped = body.strip()
    candidates = [stripped] if stripped.startswith("{") else []
    candidates += [line[len("data: "):] for line in re.split(r"\r\n|\r|\n", body) if line.startswith("data: ")]
    envelope_seen = False
    for candidate in candidates:
        try:
            text = _from_payload(candidate)
        except json.JSONDecodeError:
            continue
        if text:
            return text
        envelope_seen = envelope_seen or text == ""
    raise KeylessMCPError("MCP response contained no text content" if envelope_seen else "Unrecognized MCP response shape")


def _response_text(response: Any) -> str:
    """Body decoded with the declared charset, else UTF-8. JSON-RPC and SSE bodies are UTF-8 by
    spec, but ``text/event-stream`` carries no charset and ``requests`` then decodes ``.text`` as
    ISO-8859-1 — mojibake for every non-ASCII result and, for CJK, stray U+0085 line breaks that
    made the envelope unparseable."""
    from requests.utils import get_encoding_from_headers
    content_type = response.headers.get("Content-Type", "")
    declared = get_encoding_from_headers({"content-type": content_type}) if "charset=" in content_type.lower() else None
    return response.content.decode(declared or "utf-8", errors="replace")


def mcp_call(url: str, tool: str, arguments: Dict[str, Any], timeout: int = _TIMEOUT_SECONDS) -> str:
    """POST a JSON-RPC ``tools/call`` and return the text payload. Raises
    :class:`KeylessMCPError` on transport failures, non-2xx, JSON-RPC and tool errors."""
    import requests
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "User-Agent": "hermes-agent"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise KeylessMCPError(f"HTTP {response.status_code}: {_response_text(response)[:300]}")
    return _parse_mcp_body(_response_text(response))


# --- Parallel (search.parallel.ai) — JSON text payloads -----------------------
def parallel_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    def _rows() -> List[Dict[str, Any]]:
        text = mcp_call(PARALLEL_MCP_URL, "web_search", {"objective": query, "search_queries": [query], "session_id": _SESSION_ID})
        results = json.loads(text).get("results") or []
        return [
            _row(r.get("url") or "", r.get("title") or "", " ".join(r.get("excerpts") or []), i + 1)
            for i, r in enumerate(results[:max(limit, 0)] if limit else results)
        ]

    return _search("parallel", _rows, (json.JSONDecodeError, TypeError, KeyError), lambda exc: f"Keyless Parallel search returned an unexpected payload: {exc}")


def parallel_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    try:
        data = json.loads(mcp_call(PARALLEL_MCP_URL, "web_fetch", {"urls": list(urls), "objective": "Full page content", "session_id": _SESSION_ID}))
    except (KeylessMCPError, json.JSONDecodeError, TypeError) as exc:
        message = _fail_msg("parallel", "extract", exc)
        return [_page_error(u, message) for u in urls]
    results = [
        _page(r.get("url") or "", r.get("title") or "", r.get("full_content") or r.get("content") or "\n\n".join(r.get("excerpts") or []))
        for r in data.get("results") or []
    ]
    for error in data.get("errors") or []:
        url = error.get("url") or ""
        results.append({**_page_error(url, str(error.get("content") or error.get("error_type") or "extraction failed")), "metadata": {"sourceURL": url}})
    # URLs the endpoint silently dropped still get an error entry (per-URL contract).
    seen = {r["url"] for r in results}
    results.extend(_page_error(u, "no content returned") for u in urls if u not in seen)
    return results


# --- Exa (mcp.exa.ai) — formatted plain-text payloads -------------------------
def _after(line: str, prefix: str) -> str:
    return line[len(prefix):].strip()


_EXA_LABELS = ("Title:", "URL:", "Highlights:", "Published:", "Author:")


def _parse_exa_search_text(text: str, limit: int) -> List[Dict[str, Any]]:
    """Parse Exa's ``---``-separated ``Title:/URL:/Published:/Author:/Highlights:`` blocks."""
    results: List[Dict[str, Any]] = []
    for block in text.split("\n---\n"):
        title = url = ""
        highlight_lines: List[str] = []
        in_highlights = False
        for stripped in map(str.strip, block.splitlines()):
            if stripped.startswith("Title:"):
                title = _after(stripped, "Title:")
            elif stripped.startswith("URL:"):
                url = _after(stripped, "URL:")
            elif in_highlights and stripped and not stripped.startswith(_EXA_LABELS):
                highlight_lines.append(stripped)
            # Highlights run until the next labelled field.
            if stripped.startswith(_EXA_LABELS):
                in_highlights = stripped.startswith("Highlights:")
        if url:
            results.append(_row(url, title, " ".join(highlight_lines), len(results) + 1))
        if limit and len(results) >= limit:
            break
    return results


def exa_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    return _search("exa", lambda: _parse_exa_search_text(mcp_call(EXA_MCP_URL, "web_search_exa", {"query": query, "numResults": max(1, int(limit))}), limit))


def exa_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Called per-URL; the tool returns one combined text payload."""
    def _fetch(url: str) -> Dict[str, Any]:
        text = mcp_call(EXA_MCP_URL, "web_fetch_exa", {"urls": [url]})
        # Title: first markdown H1 or ``Title:`` line, whichever comes first.
        titles = (_after(s, "# " if s.startswith("# ") else "Title:") for s in map(str.strip, text.splitlines()) if s.startswith(("# ", "Title:")))
        return _page(url, next(titles, ""), text)

    return _per_url(urls, _fetch, "exa", catch=KeylessMCPError, hint=True)


# --- Firecrawl keyless (public cloud API, no auth header) ---------------------
def firecrawl_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_web_search_results
    rows = lambda: _extract_web_search_results(_KeylessFirecrawlClient().search(query=query, limit=limit))  # noqa: E731
    return _search("firecrawl", rows, Exception, lambda exc: _fail_msg("firecrawl", "search", exc))


def firecrawl_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_scrape_payload
    client = _KeylessFirecrawlClient()

    def _fetch(url: str) -> Dict[str, Any]:
        payload = _extract_scrape_payload(client.scrape(url=url, formats=["markdown"])) or {}
        metadata = payload.get("metadata") or {}
        title = metadata.get("title") if isinstance(metadata, dict) else None
        return _page(url, title or "", payload.get("markdown") or payload.get("html") or "")

    return _per_url(urls, _fetch, "firecrawl")


# --- Keenable keyless (api.keenable.ai public endpoints) ----------------------
def _keenable_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    """Call a Keenable public endpoint with the mandatory X-Keenable-Title app id."""
    import requests
    headers = {"X-Keenable-Title": _KEENABLE_TITLE}
    if method == "post":
        headers["Content-Type"] = "application/json"
    response = getattr(requests, method)(f"{KEENABLE_API_URL}{path}", headers=headers, timeout=_TIMEOUT_SECONDS, **kwargs)
    if response.status_code >= 400:
        raise KeylessMCPError(f"HTTP {response.status_code}: {_response_text(response).strip()[:300]}")
    return response.json()


def keenable_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    def _rows() -> List[Dict[str, Any]]:
        data = _keenable_request("post", "/v1/search/public", json={"query": query, "max_results": max(1, int(limit))})
        return [
            _row(r.get("url") or "", r.get("title") or "", r.get("snippet") or r.get("description") or "", i + 1)
            for i, r in enumerate(data.get("results") or [])
        ]

    return _search("keenable", _rows, Exception, lambda exc: f"Keyless Keenable search failed: {exc}.")


def keenable_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    def _fetch(url: str) -> Dict[str, Any]:
        data = _keenable_request("get", "/v1/fetch/public", params={"url": url})
        return _page(data.get("url") or url, data.get("title") or "", data.get("content") or "", source_url=url)

    return _per_url(urls, _fetch, "keenable")


# --- Quarantine: time-based cooldown for hard-failed vendors (mk addition) ----
# A vendor that failed HARD (HTTP error / timeout / captcha / success:false —
# NOT a mere empty result) is benched for QUARANTINE_SECONDS so the next
# queries skip it instead of re-hitting a dead endpoint. Purely local state in
# ~/mk-data/cache/szukaj-kwarantanna.json; when every vendor sits in
# quarantine the earliest-expiring one is used so a query never dead-ends.
_QUARANTINE_FILE = Path(os.environ.get("MK_SZUKAJ_KWARANTANNA")
                         or os.path.expanduser("~/mk-data/cache/szukaj-kwarantanna.json"))
_QUARANTINE_SECONDS = 2 * 60 * 60  # 2h, per task t_f34f58b5
_quarantine_lock = threading.Lock()


def _is_hard_failure(result: Any) -> bool:
    """A vendor-transport failure (the result carries success:false), as opposed
    to an empty-but-successful answer. Empty results never quarantine; non-dict
    results (the per-URL extract lists) are never hard failures either."""
    if not isinstance(result, dict):
        return False
    return not result.get("success", True)


def quarantine_until(vendor: str, seconds: int = _QUARANTINE_SECONDS) -> float:
    """Bench *vendor* until now+seconds; returns the expiry epoch."""
    until = time.time() + seconds
    try:
        _QUARANTINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _quarantine_lock:
            state = _load_quarantine()
            state[vendor] = until
            _QUARANTINE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError as exc:
        logger.debug("quarantine write failed for %s: %s", vendor, exc)
    return until


def _load_quarantine() -> Dict[str, float]:
    try:
        state = json.loads(_QUARANTINE_FILE.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            now = time.time()
            live = {str(k): float(v) for k, v in state.items() if float(v) > now}
            if len(live) != len(state):  # expired entries dropped on read
                _QUARANTINE_FILE.write_text(json.dumps(live), encoding="utf-8")
            return live
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("quarantine read failed: %s", exc)
    return {}


def _quarantined_vendors() -> Dict[str, float]:
    """Currently-benched vendors with their expiry epochs (fresh read)."""
    with _quarantine_lock:
        return _load_quarantine()


def _filter_ring_by_quarantine(order: List[str]) -> List[str]:
    """Ring order minus vendors still in quarantine. When that would empty the
    list, keep only the vendor whose bench expires earliest (a search must
    always have somewhere to go)."""
    benched = _quarantined_vendors()
    if not benched:
        return order
    live = [v for v in order if v not in benched]
    if live:
        return live
    earliest = min(benched, key=benched.get)
    logger.info("all keyless vendors quarantined; using earliest-expiring %s", earliest)
    return [earliest]


# --- Round-robin ring + next-in-line failover (rate-limited free tiers) -------
_KEYLESS_RING = ("exa", "parallel", "firecrawl", "keenable")

# Late-bound lookups (not bare references) so ``patch.object(keyless_mcp, "<vendor>_search_keyless")``
# is honored at call time. Tests also ``setitem`` these dicts directly.
_KEYLESS_SEARCHERS: Dict[str, Callable[[str, int], Dict[str, Any]]] = {v: (lambda query, limit, _v=v: globals()[f"{_v}_search_keyless"](query, limit)) for v in _KEYLESS_RING}
_KEYLESS_EXTRACTORS: Dict[str, Callable[[List[str]], List[Dict[str, Any]]]] = {v: (lambda urls, _v=v: globals()[f"{_v}_extract_keyless"](urls)) for v in _KEYLESS_RING}

# Per-process round-robin cursor, seeded by the random session id so the fleet
# spreads across vendors; advances once per unpinned keyless request.
_ring_lock = threading.Lock()
_ring_cursor = int(_SESSION_ID, 16) % len(_KEYLESS_RING)


def _vendor_pinned(name: str) -> bool:
    """True when config explicitly routes web traffic to *name* (backend keys or a
    ``free`` tier pin). A pinned vendor starts every keyless request."""
    if provider_tier(name) == "free":
        return True
    try:
        return _web_config_selects(name)
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("_vendor_pinned(%r) config read failed: %s", name, exc)
        return False


def _ring_order(name: str) -> List[str]:
    """Vendor walk order: pinned → start at *name* (its ring position fixes the failover
    succession); else round-robin from the cursor, advancing it per request. Vendors
    pinned ``paid`` are excluded (explicit paid opts their free endpoint out), and so
    are vendors still in quarantine (hard-failure cooldown, see above)."""
    global _ring_cursor
    if _vendor_pinned(name):
        start = _KEYLESS_RING.index(name) if name in _KEYLESS_RING else 0
    else:
        with _ring_lock:
            start = _ring_cursor
            _ring_cursor = (_ring_cursor + 1) % len(_KEYLESS_RING)
    ordered = _KEYLESS_RING[start:] + _KEYLESS_RING[:start]
    return _filter_ring_by_quarantine([v for v in ordered if provider_tier(v) != "paid"])


_ALL_PAID_MSG = "All keyless web providers are pinned to paid tiers."


def _walk_ring(name: str, kind: str, call, throttled) -> tuple:
    """Call each vendor from :func:`_ring_order` until a result is not ``throttled``.
    Returns ``(order, vendor, result, exhausted)``; ``order`` is empty (result None)
    when every vendor is pinned paid. A hard failure (HTTP error, timeout, captcha,
    success:false — anything with ``success`` not True) also advances to the next
    vendor within the SAME query instead of ending the walk: like OpenClaw, when a
    provider is down you try the next one. Hard failures additionally bench the
    vendor in quarantine for 2h so subsequent queries skip it; an empty but
    successful result is a real answer and never quarantines."""
    order = _ring_order(name)
    vendor, result = None, None
    for i, vendor in enumerate(order):
        result = call(vendor)
        if not throttled(result) and not _is_hard_failure(result):
            return order, vendor, result, False
        if _is_hard_failure(result):
            quarantine_until(vendor)
            if i + 1 < len(order):
                logger.info("keyless %s %s hard-failed (%s); failing over to %s",
                            vendor, kind, (result.get("error") or "")[:120], order[i + 1])
                continue
        if i + 1 < len(order):
            logger.info("keyless %s %s unavailable; failing over to %s", vendor, kind, order[i + 1])
    return order, vendor, result, True


def search_with_failover(name: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """Chain entry point. The managed chain (Synthetic → MiniMax → z.ai MCP, see the
    ``_chain_search`` helpers below) serves first whenever it is configured; the keyless
    ring below is the last-resort tier. ``data.served_by`` records who actually answered."""

    def _throttled(result: Dict[str, Any]) -> bool:
        return not result.get("success") and _is_rate_limitish(result.get("error", ""))

    chained = _chain_search(query, limit)
    if chained is not None:
        return chained
    order, vendor, result, exhausted = _walk_ring(name, "search", lambda v: _KEYLESS_SEARCHERS[v](query, limit), _throttled)
    if not order:
        return search_fail(_ALL_PAID_MSG)
    if exhausted:
        result["error"] = f"{result.get('error', '')} (all keyless vendors failed or throttled: {', '.join(order)})"
    elif result.get("success") and vendor != name:
        result.setdefault("data", {})["served_by"] = vendor
    return result


def extract_with_failover(name: str, urls: List[str]) -> List[Dict[str, Any]]:
    """Advances to the next ring vendor only when EVERY url in a batch comes
    back with a rate-limit-shaped error — partial failures and HTTP 403s can
    be page problems, not provider throttling, and return as-is."""

    def _all_throttled(results: List[Dict[str, Any]]) -> bool:
        return bool(results) and all(r.get("error", "") and _is_rate_limitish(r.get("error", "")) for r in results)

    order, _vendor, results, _exhausted = _walk_ring(name, "extract", lambda v: _KEYLESS_EXTRACTORS[v](list(urls)), _all_throttled)
    if not order:
        return [_page_error(u, _ALL_PAID_MSG) for u in urls]
    return results


# --- Managed search chain: Synthetic → MiniMax → z.ai MCP (mk task t_f34f58b5) ----
# Serves keyed/managed searches BEFORE the keyless ring, per Michał's fixed order:
#   1. Synthetic  POST https://api.synthetic.new/v2/search   (Bearer SYNTHETIC_API_KEY;
#      search has its OWN counter — 250 req/h, GET /v2/quotas is free) — run it until
#      the hourly window is full; skip this query when requests == limit.
#   2. MiniMax    POST https://api.minimax.io/v1/coding_plan/search  (body {"q": ...} —
#      NOT "query"; no "model" field — the endpoint rejects/ignores it). Subscription
#      key: env MINIMAX_CODE_PLAN_KEY, else one line of
#      ~/.hermes/profiles/mk/secrets/minimax-code-plan.key. 5h window from
#      GET /v1/coding_plan/remains (model "general"); go to 95% used, skip above.
#      GET remains does not consume the search quota.
#   3. z.ai Web Search MCP (config mcp_servers.zai_search, streamable HTTP JSON-RPC;
#      Authorization from config — Bearer ${GLM_API_KEY}). Usage from the z.ai quota
#      endpoint (stan.json written by ~/mk-stack/scripts/usage.py — same numbers the
#      operator sees; reading it is a plain local file read). Hard stop at 80% of the
#      MCP pool; never assume a numeric limit.
#   4. Keyless ring (exa/parallel/keenable; firecrawl tier-pinned paid) — the caller's
#      existing fallback, unchanged.
# SearXNG and DuckDuckGo stay disabled. Only a HARD member failure advances the chain;
# an empty result is an answer. The chain never raises — it returns None so the ring
# takes over, or a search_fail(...) shaped result.

MINIMAX_SEARCH_URL = "https://api.minimax.io/v1/coding_plan/search"
MINIMAX_REMAINS_URL = "https://api.minimax.io/v1/coding_plan/remains"
SYNTHETIC_SEARCH_URL = "https://api.synthetic.new/v2/search"
SYNTHETIC_QUOTAS_URL = "https://api.synthetic.new/v2/quotas"
_MINIMAX_KEY_FILE = Path("/home/hermes/.hermes/profiles/mk/secrets/minimax-code-plan.key")
_ZAI_MCP_TIMEOUT = 60          # per-request seconds; web_search_prime can be slow
_ZAI_MCP_PROTOCOL_VERSION = "2025-03-26"
_MINIMAX_WINDOW_SECONDS = 5 * 60 * 60
_ZAI_MCP_STOP_PCT = 80.0       # hard stop: skip z.ai when MCP usage >= 80%
_MINIMAX_STOP_PCT = 95.0       # skip MiniMax when the 5h window is >= 95% used
_ZAI_SESSION_LOCK = threading.Lock()
_ZAI_SESSION: Dict[str, Optional[str]] = {"id": None}

_chain_logger = logging.getLogger(__name__ + ".chain")


def _chain_http_json(url: str, *, key: str = "", body: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Tuple[int, str]:
    """POST/GET JSON; returns (status, body-text). Raises KeylessMCPError on transport failure."""
    import requests
    req_headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "hermes-agent"}
    if key:
        req_headers["Authorization"] = "Bearer " + key
    if headers:
        req_headers.update(headers)
    try:
        if body is None:
            response = requests.get(url, headers=req_headers, timeout=timeout)
        else:
            response = requests.post(url, json=body, headers=req_headers, timeout=timeout)
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    return response.status_code, response.content.decode("utf-8", errors="replace")


def _provider_env(name: str) -> str:
    """Config-aware env lookup (process env, then the profile .env). Empty when unset."""
    try:
        from agent.web_search_provider import get_provider_env
        return get_provider_env(name)
    except Exception:  # noqa: BLE001 — stripped envs: plain os.environ
        return os.environ.get(name, "")


# --- 1. Synthetic -----------------------------------------------------------------
def _synthetic_ready() -> Tuple[bool, str]:
    """(callable_now, reason). Reads GET /v2/quotas (free); search.hourly governs."""
    key = _provider_env("SYNTHETIC_API_KEY")
    if not key:
        return False, "no SYNTHETIC_API_KEY"
    try:
        status, text = _chain_http_json(SYNTHETIC_QUOTAS_URL, key=key)
    except KeylessMCPError as exc:
        return False, f"quotas: {exc}"
    if status >= 400:
        return False, f"quotas HTTP {status}"
    try:
        hourly = ((json.loads(text) or {}).get("search") or {}).get("hourly") or {}
        requests_made, limit = int(hourly.get("requests", 0)), int(hourly.get("limit", 0))
    except (ValueError, TypeError):
        return False, "quotas: unexpected shape"
    if limit and requests_made >= limit:
        return False, f"search.hourly full ({requests_made}/{limit})"
    return True, f"search.hourly {requests_made}/{limit}"


def _synthetic_search(query: str, limit: int) -> Dict[str, Any]:
    key = _provider_env("SYNTHETIC_API_KEY")
    if not key:
        return search_fail("synthetic: no SYNTHETIC_API_KEY")
    status, text = _chain_http_json(SYNTHETIC_SEARCH_URL, key=key, body={"query": query, "num_results": max(1, int(limit))})
    if status >= 400:
        return search_fail(f"synthetic: HTTP {status}: {text[:200]}")
    try:
        rows = json.loads(text).get("results") or []
    except ValueError:
        return search_fail(f"synthetic: unexpected payload: {text[:200]}")
    return search_ok([
        _row(r.get("url") or "", r.get("title") or "", (r.get("text") or "")[:500], i + 1)
        for i, r in enumerate(rows[: max(1, int(limit))] if limit else rows)
    ])


# --- 2. MiniMax -------------------------------------------------------------------
def _minimax_key() -> str:
    """Subscription key: env MINIMAX_CODE_PLAN_KEY, else one line of the secrets file."""
    key = _provider_env("MINIMAX_CODE_PLAN_KEY")
    if key:
        return key
    try:
        for line in _MINIMAX_KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                return line
    except OSError:
        pass
    return ""


def _minimax_ready(key: str) -> Tuple[bool, str]:
    """(callable_now, reason) from GET /v1/coding_plan/remains, model 'general' (5h window).
    Prefers the API's own remaining percent; falls back to usage/total inside the window."""
    if not key:
        return False, "no MINIMAX_CODE_PLAN_KEY and no secrets file"
    try:
        status, text = _chain_http_json(MINIMAX_REMAINS_URL, key=key)
    except KeylessMCPError as exc:
        return False, f"remains: {exc}"
    if status >= 400:
        return False, f"remains HTTP {status}"
    try:
        data = json.loads(text)
        general = next(m for m in (data.get("model_remains") or []) if m.get("model_name") == "general")
    except (ValueError, StopIteration, TypeError):
        return False, "remains: no 'general' bucket"
    remaining_pct = general.get("current_interval_remaining_percent")
    if isinstance(remaining_pct, (int, float)):
        if remaining_pct <= 100.0 - _MINIMAX_STOP_PCT:
            return False, f"5h window {100.0 - remaining_pct:.0f}% used (>= {_MINIMAX_STOP_PCT:.0f}%)"
        return True, f"5h window {100.0 - remaining_pct:.0f}% used"
    now = time.time()
    start, total, used = general.get("start_time"), general.get("current_interval_total_count"), general.get("current_interval_usage_count")
    if not (isinstance(start, (int, float)) and now * 1000 - start <= _MINIMAX_WINDOW_SECONDS * 1000):
        return True, "5h window boundary: treating as fresh"
    if isinstance(total, (int, float)) and total > 0 and used / total >= _MINIMAX_STOP_PCT / 100.0:
        return False, f"5h window {used}/{total} used (>= {_MINIMAX_STOP_PCT:.0f}%)"
    return True, "5h window below stop threshold"


def _minimax_search(query: str, limit: int) -> Dict[str, Any]:
    key = _minimax_key()
    if not key:
        return search_fail("minimax: no subscription key (MINIMAX_CODE_PLAN_KEY / secrets file)")
    status, text = _chain_http_json(MINIMAX_SEARCH_URL, key=key, body={"q": query})
    if status >= 400:
        return search_fail(f"minimax: HTTP {status}: {text[:200]}")
    try:
        data = json.loads(text)
    except ValueError:
        return search_fail(f"minimax: unexpected payload: {text[:200]}")
    base = data.get("base_resp") or {}
    if base.get("status_code") not in (0, None):
        return search_fail(f"minimax: success false ({base.get('status_msg') or base.get('status_code')})")
    return search_ok([
        _row(r.get("link") or "", r.get("title") or "", r.get("snippet") or "", i + 1)
        for i, r in enumerate((data.get("organic") or [])[: max(1, int(limit))] if limit else (data.get("organic") or []))
    ])


# --- 3. z.ai Web Search MCP (streamable HTTP JSON-RPC) -----------------------------
def _zai_mcp_config() -> Tuple[str, str]:
    """(url, Authorization header) from config ``mcp_servers.zai_search`` — the same
    expansion the gateway uses (``${VAR}`` refs resolved via the config env layer)."""
    try:
        from hermes_cli.config import load_config
        cfg = (load_config().get("mcp_servers") or {}).get("zai_search") or {}
        url = str(cfg.get("url") or "")
        auth = str((cfg.get("headers") or {}).get("Authorization") or "")

        def _expand(value: str) -> str:
            return re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda m: _provider_env(m.group(1)),
                value,
            )

        return _expand(url), _expand(auth)
    except Exception as exc:  # noqa: BLE001 — config layer optional
        _chain_logger.debug("zai mcp config read failed: %s", exc)
        return "", ""


def _zai_ready(url: str, auth: str) -> Tuple[bool, str]:
    """(callable_now, reason): usable credentials AND MCP pool below the hard stop.
    Usage comes from the z.ai quota endpoint via the stan.json the usage monitor
    maintains — the same numbers the operator sees; never a guessed limit."""
    if not url or not auth:
        return False, "no zai_search mcp config/credentials"
    try:
        from hermes_cli.config import get_env_value
        mcp_pct = (get_env_value("ZAI_MCP_PCT") or "").strip()
    except Exception:  # noqa: BLE001 — fall through to os.environ
        mcp_pct = ""
    if not mcp_pct:
        mcp_pct = os.environ.get("ZAI_MCP_PCT", "")
    if not mcp_pct:
        try:
            stan = json.loads(Path("~/mk-data/usage/stan.json").expanduser().read_text(encoding="utf-8"))
            mcp_pct = str(stan.get("mcp_pct", "") or "")
        except (OSError, ValueError):
            mcp_pct = ""
    try:
        if mcp_pct and float(mcp_pct) >= _ZAI_MCP_STOP_PCT:
            return False, f"z.ai MCP pool at {mcp_pct}% (>= {_ZAI_MCP_STOP_PCT:.0f}%)"
    except ValueError:
        pass
    return True, "below MCP stop threshold"


def _zai_session_id(url: str, auth: str) -> Optional[str]:
    """Initialize once per process and keep the server-issued session id."""
    with _ZAI_SESSION_LOCK:
        if _ZAI_SESSION["id"]:
            return _ZAI_SESSION["id"]
        status, text = _chain_http_json(
            url, body={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": _ZAI_MCP_PROTOCOL_VERSION, "capabilities": {},
                           "clientInfo": {"name": "hermes-agent", "version": "1.0"}},
            }, headers={"Authorization": auth, "Accept": "application/json, text/event-stream"},
            timeout=_ZAI_MCP_TIMEOUT)
        if status >= 400:
            raise KeylessMCPError(f"HTTP {status}: {text[:200]}")
        if '"error"' in text or '"code":40' in text:
            raise KeylessMCPError(f"initialize rejected: {text[:200]}")
        # The server may be stateless (no mcp-session-id issued — z.ai is): keep None
        # and never send a made-up session header.
        session = None
        header_names = ("mcp-session-id", "Mcp-Session-Id")
        try:
            import requests as _requests
            probe = _requests.post(
                url, json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                         "Authorization": auth}, timeout=_ZAI_MCP_TIMEOUT)
            session = next((probe.headers[n] for n in header_names if probe.headers.get(n)), None)
        except _requests.RequestException:
            pass
        _ZAI_SESSION["id"] = session
        return session


def _zai_search(query: str, limit: int) -> Dict[str, Any]:
    url, auth = _zai_mcp_config()
    if not url or not auth:
        return search_fail("z.ai: no zai_search mcp config/credentials")
    try:
        session = _zai_session_id(url, auth)
    except KeylessMCPError as exc:
        return search_fail(f"z.ai: initialize failed: {exc}")
    headers = {"Authorization": auth}
    if session:
        headers["mcp-session-id"] = session
    status, text = _chain_http_json(
        url, body={
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "web_search_prime",
                       "arguments": {"search_query": query[:70], "location": "us"}},
        }, headers=headers, timeout=_ZAI_MCP_TIMEOUT)
    if status >= 400:
        return search_fail(f"z.ai: HTTP {status}: {text[:200]}")
    payload = ""
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                data = json.loads(line[len("data: "):])
            except ValueError:
                continue
            if data.get("error"):
                return search_fail(f"z.ai: rpc error: {json.dumps(data['error'])[:200]}")
            for item in ((data.get("result") or {}).get("content") or []):
                if isinstance(item, dict) and item.get("type") == "text":
                    payload += item.get("text") or ""
    if not payload:
        try:
            data = json.loads(text)
            if data.get("error"):
                return search_fail(f"z.ai: rpc error: {json.dumps(data['error'])[:200]}")
            for item in ((data.get("result") or {}).get("content") or []):
                if isinstance(item, dict) and item.get("type") == "text":
                    payload += item.get("text") or ""
        except ValueError:
            return search_fail(f"z.ai: unrecognized response: {text[:200]}")
    try:
        rows = json.loads(payload)
    except ValueError:
        return search_fail(f"z.ai: unexpected tool payload: {payload[:200]}")
    if isinstance(rows, dict):
        rows = rows.get("results") or rows.get("data") or []
    return search_ok([
        _row(r.get("link") or r.get("url") or "", r.get("title") or "", r.get("content") or r.get("snippet") or "", i + 1)
        for i, r in enumerate(rows[: max(1, int(limit))] if limit else rows)
        if isinstance(r, dict)
    ])


# --- chain driver -------------------------------------------------------------------
# Late-bound member table (same convention as _KEYLESS_SEARCHERS): the driver looks the
# callable up at call time, so tests can rebind module attrs or setitem this dict.
_CHAIN_MEMBERS = ("synthetic", "minimax", "z.ai")
_CHAIN_SEARCHERS: Dict[str, Callable[[str, int], Dict[str, Any]]] = {
    "synthetic": _synthetic_search, "minimax": _minimax_search, "z.ai": _zai_search,
}


def _chain_search(query: str, limit: int) -> Optional[Dict[str, Any]]:
    """Synthetic → MiniMax → z.ai; None when the chain is unconfigured/exhausted so the
    keyless ring takes over. Only hard failures advance; empty results are answers."""
    zai_url, zai_auth = _zai_mcp_config()
    readiness = {
        "synthetic": _synthetic_ready,
        "minimax": lambda: _minimax_ready(_minimax_key()),
        "z.ai": lambda: _zai_ready(zai_url, zai_auth),
    }
    for name in _CHAIN_MEMBERS:
        check = readiness[name]
        try:
            ok, reason = check()
        except Exception as exc:  # noqa: BLE001 — a broken readiness probe must not kill the chain
            _chain_logger.debug("chain %s readiness probe raised: %s", name, exc)
            ok, reason = True, "probe error — attempting anyway"
        if not ok:
            _chain_logger.info("chain %s skipped: %s", name, reason)
            continue
        try:
            result = _CHAIN_SEARCHERS[name](query, limit)
        except Exception as exc:  # noqa: BLE001 — treat as a hard member failure
            _chain_logger.info("chain %s raised (%s); advancing", name, exc)
            result = search_fail(f"{name}: {exc}")
        if isinstance(result, dict) and result.get("success"):
            result.setdefault("data", {})["served_by"] = name
            return result
        _chain_logger.info("chain %s failed (%s); advancing", name, (result or {}).get("error", ""))
    return None
