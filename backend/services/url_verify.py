"""
HTTP-based URL verification for cited papers, datasets, and algorithms.

Deliberately independent of citation generation: a citation can be
well-formed while its link is dead (moved paper, renamed dataset repo,
etc.), and callers need that distinction for evidence-quality scoring.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

_TIMEOUT = 6.0
_MAX_WORKERS = 8
_MAX_RETRIES = 2
_HEADERS = {"User-Agent": "ARES-URLVerifier/1.0 (+https://github.com)"}


def _check_once(url: str) -> bool:
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS) as client:
        resp = client.head(url)
        if resp.status_code < 400:
            return True
        with client.stream("GET", url) as stream_resp:
            return stream_resp.status_code < 400


def verify_url(url: str) -> Optional[bool]:
    """True/False when we got a real HTTP response (resolves vs. a genuine
    4xx/5xx); None when we couldn't get any response at all, even after
    retries — this is NOT the same thing as a dead link. Tries HEAD first
    (cheap); always confirms with a streamed GET before concluding a URL is
    dead, since HEAD support is unreliable across hosts in both directions —
    some reject it outright with 403/405, and some (e.g. kaggle.com) return
    a bare 404 on HEAD for pages that resolve fine on GET.

    Retries transport-level failures (connection reset, timeout), but on
    this network arXiv connections can reset for several seconds straight
    (observed WinError 10054 persisting across retries with delays, while
    google.com resolved fine throughout — a local network/VPN issue, not a
    dead paper). Collapsing "couldn't reach the server at all" to the same
    False as "server said 404" wrongly zeroed out evidence-quality scoring
    for citations that are actually fine. Returning None here lets callers
    (services/collaboration.py evidence_quality_literature) fall back to
    "URL is present" credit instead of "confirmed dead" — matching the
    Optional[bool] semantics models.schemas.Paper.url_verified already
    documents (None = not yet checked / couldn't determine)."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    last_was_transport_error = False
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _check_once(url)
        except httpx.HTTPError:
            last_was_transport_error = True
            if attempt >= _MAX_RETRIES:
                break
            time.sleep(0.5 * (attempt + 1))
    return None if last_was_transport_error else False


def verify_urls_concurrently(urls: list[str]) -> dict[str, Optional[bool]]:
    """Verify a batch of (possibly duplicate) URLs concurrently. Returns a
    map of url -> resolvable (True/False/None — see verify_url), so callers
    can look up results by URL."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(unique))) as pool:
        results = list(pool.map(verify_url, unique))
    return dict(zip(unique, results))
