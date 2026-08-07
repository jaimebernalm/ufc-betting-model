"""Minimal Kalshi REST client with RSA-PSS request signing.

Auth model (Kalshi v2): every request includes three headers —
  KALSHI-ACCESS-KEY        : the Key ID (UUID)
  KALSHI-ACCESS-TIMESTAMP  : current epoch in milliseconds
  KALSHI-ACCESS-SIGNATURE  : base64( RSA-PSS-SHA256( timestamp + method + path ) )

The signed message uses the path *without* query string and the HTTP method
in uppercase.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class KalshiConfig:
    key_id: str
    private_key_path: Path
    env: str  # "prod" | "demo"

    @property
    def base_url(self) -> str:
        return DEMO_BASE if self.env == "demo" else PROD_BASE

    @classmethod
    def from_env(cls) -> KalshiConfig:
        key_id = os.environ.get("KALSHI_KEY_ID", "").strip()
        path = os.path.expanduser(os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip())
        env = os.environ.get("KALSHI_ENV", "prod").strip() or "prod"
        if not key_id or not path:
            raise RuntimeError(
                "Missing KALSHI_KEY_ID or KALSHI_PRIVATE_KEY_PATH in env. Check your .env file."
            )
        return cls(key_id=key_id, private_key_path=Path(path), env=env)


class KalshiClient:
    def __init__(self, config: KalshiConfig | None = None, session: requests.Session | None = None):
        self.config = config or KalshiConfig.from_env()
        self.session = session or requests.Session()
        with open(self.config.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        msg = f"{timestamp_ms}{method.upper()}{path}".encode()
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        url = f"{self.config.base_url}{path}"
        sign_path = f"/trade-api/v2{path}"
        # Retry on 429 with exponential backoff (Kalshi rate-limits aggressively)
        backoff = 1.0
        for _attempt in range(6):
            ts = str(int(time.time() * 1000))
            headers = {
                "KALSHI-ACCESS-KEY": self.config.key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, sign_path),
                "Accept": "application/json",
            }
            if json is not None:
                headers["Content-Type"] = "application/json"
            resp = self.session.request(method, url, params=params, json=json, headers=headers, timeout=30)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Kalshi {method} {path} -> {resp.status_code}: {resp.text[:500]}")
            return resp.json()
        raise RuntimeError(f"Kalshi {method} {path} -> 429 after retries")

    # ---- convenience wrappers ----

    def get_balance(self) -> dict:
        return self.request("GET", "/portfolio/balance")

    def get_exchange_status(self) -> dict:
        return self.request("GET", "/exchange/status")

    def list_series(self, *, category: str | None = None) -> dict:
        params = {"category": category} if category else None
        return self.request("GET", "/series", params=params)

    def list_events(
        self, *, series_ticker: str | None = None, status: str = "open", limit: int = 200
    ) -> dict:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self.request("GET", "/events", params=params)

    def list_markets(
        self,
        *,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
    ) -> dict:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self.request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        return self.request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 50) -> dict:
        """Returns {'yes': [(price_$, qty), ...], 'no': [(price_$, qty), ...]}.

        Kalshi v2 response shape is
        `{"orderbook_fp": {"yes_dollars": [["0.51","100"], ...], "no_dollars": ...}}`.
        Each side represents RESTING BUY orders on that outcome. To BUY YES, lift NO bids
        (a NO bid at price p == a YES ask at price 1-p).
        """
        raw = self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}

        def conv(side_key: str) -> list[tuple[float, float]]:
            out = []
            for entry in ob.get(side_key) or []:
                try:
                    out.append((float(entry[0]), float(entry[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            return out

        return {
            "yes": conv("yes_dollars") or conv("yes"),
            "no": conv("no_dollars") or conv("no"),
        }
