"""
HTTP-based URL verification for cited papers, datasets, and algorithms.

Deliberately independent of citation generation: a citation can be
well-formed while its link is dead (moved paper, renamed dataset repo,
etc.), and callers need that distinction for evidence-quality scoring.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import httpx

_TIMEOUT = 6.0
_MAX_WORKERS = 8
_HEADERS = {"User-Agent": "ARES-URLVerifier/1.0 (+https://github.com)"}


def verify_url(url: str) -> bool:
    """True if the URL resolves with a non-error status. Tries HEAD first
    (cheap); always confirms with a streamed GET before concluding a URL is
    dead, since HEAD support is unreliable across hosts in both directions —
    some reject it outright with 403/405, and some (e.g. kaggle.com) return
    a bare 404 on HEAD for pages that resolve fine on GET. A single false
    "unreachable" verdict would wrongly tank evidence-quality scoring and
    mislabel a live link as dead in the UI, so the extra request is worth it."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.head(url)
            if resp.status_code < 400:
                return True
            with client.stream("GET", url) as stream_resp:
                return stream_resp.status_code < 400
    except httpx.HTTPError:
        return False


def verify_urls_concurrently(urls: list[str]) -> dict[str, bool]:
    """Verify a batch of (possibly duplicate) URLs concurrently. Returns a
    map of url -> resolvable, so callers can look up results by URL."""
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(unique))) as pool:
        results = list(pool.map(verify_url, unique))
    return dict(zip(unique, results))
