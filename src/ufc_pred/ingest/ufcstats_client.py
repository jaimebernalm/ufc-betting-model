"""HTTP client for ufcstats.com that solves the JS proof-of-work challenge.

ufcstats.com serves a small SHA-256 hash-cash challenge on the first request
of a session. This module wraps `requests.Session` and solves it transparently.
"""

from __future__ import annotations

import hashlib
import re
import time

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

_NONCE_RE = re.compile(r'nonce="([0-9a-f]+)"')
_TARGET_RE = re.compile(r"new Array\((\d+)\+1\)\.join\('0'\)")


def _solve_challenge(session: requests.Session, response: requests.Response) -> None:
    nonce = _NONCE_RE.search(response.text).group(1)
    zeros = int(_TARGET_RE.search(response.text).group(1))
    target = "0" * zeros
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1
    session.post(
        "http://ufcstats.com/__c",
        data={"nonce": nonce, "n": n},
        headers={"Referer": response.url},
        timeout=20,
    )


class UFCStatsClient:
    """Thin wrapper around requests.Session that handles the PoW challenge.

    Use as a context manager or instantiate directly. A small pause between
    requests is applied to avoid hammering the server.
    """

    def __init__(self, request_delay: float = 0.4):
        self._session = requests.Session()
        self._session.headers["User-Agent"] = UA
        self._delay = request_delay
        self._last_request_at = 0.0

    def get(self, url: str, *, max_retries: int = 2) -> str:
        for _attempt in range(max_retries + 1):
            self._respect_delay()
            r = self._session.get(url, timeout=30)
            if "Checking your browser" in r.text:
                _solve_challenge(self._session, r)
                continue
            r.raise_for_status()
            return r.text
        raise RuntimeError(f"Failed to fetch {url} after solving challenge")

    def _respect_delay(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_at = time.monotonic()

    def __enter__(self) -> UFCStatsClient:
        return self

    def __exit__(self, *exc) -> None:
        self._session.close()
