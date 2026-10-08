"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Backend name stays searxng. Search order is the keyless chain.

        Synthetic until its hourly quota (250), then MiniMax (field q), then z.ai
        until 80% of the MCP pool, then free Exa, Parallel, Keenable.
        """
        from plugins.web import keyless_mcp as km

        chained = km._chain_search(query, limit)
        if isinstance(chained, dict) and chained.get("success"):
            return chained
        errors = []
        for vendor in ("exa", "parallel", "keenable"):
            try:
                out = km._KEYLESS_SEARCHERS[vendor](query, limit)
            except Exception as exc:  # noqa: BLE001
                errors.append("%s: %s" % (vendor, exc))
                continue
            if isinstance(out, dict) and out.get("success"):
                out.setdefault("data", {})["served_by"] = vendor
                return out
            if isinstance(out, dict) and out.get("error"):
                errors.append("%s: %s" % (vendor, out.get("error")))
        detail = "; ".join(errors)[:300] if errors else "no member answered"
        return search_fail("chain exhausted: " + detail)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
        )
