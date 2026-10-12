"""Finnhub company-news (per-ticker news with summaries).

Added 2026-10-12 as the primary ``news_data`` vendor: Alpha Vantage
NEWS_SENTIMENT became premium-only for this deployment's key, and Yahoo's
per-ticker feed gives few headlines and no summaries. Measured on the same 40
names and window: 5+ relevant headlines for 58% of names (Yahoo search 33%),
none for 10% (Yahoo 20%), every kept article carrying a summary.

Free tier: 60 calls/minute. Calls are spaced FINNHUB_MIN_INTERVAL_SECONDS apart
under a lock -- the four analysts run in parallel threads (TODOS #21), so a
per-thread pause would not hold the global rate.
"""

import os
import threading
import time
from datetime import datetime, timezone

import requests

from .config import get_config
from .errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorNotEntitledError,
    VendorRateLimitError,
)
from .yfinance_news import _in_news_window

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_MIN_INTERVAL_SECONDS = 1.1
_rate_lock = threading.Lock()
_last_call = 0.0


def _throttled_get(params: dict) -> requests.Response:
    global _last_call
    with _rate_lock:
        wait = FINNHUB_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
    return requests.get(FINNHUB_NEWS_URL, params=params, timeout=20)


def get_news_finnhub(ticker: str, start_date: str, end_date: str) -> str:
    """News about ``ticker`` published in [start_date, end_date] (yyyy-mm-dd),
    newest first, in the same markdown shape as get_news_yfinance().

    Raises the router's vendor errors so ``news_data``'s chain can fall
    through: no key -> VendorNotConfiguredError, HTTP 429 ->
    VendorRateLimitError, 401/403 -> VendorNotEntitledError, no articles at
    all -> NoMarketDataError (the next vendor may still have some).
    """
    token = os.environ.get("FINNHUB_API_KEY")
    if not token:
        raise VendorNotConfiguredError("FINNHUB_API_KEY is not set")
    symbol = ticker.strip().upper()
    resp = _throttled_get({"symbol": symbol, "from": start_date, "to": end_date, "token": token})
    if resp.status_code == 429:
        raise VendorRateLimitError("Finnhub rate limit (429)")
    if resp.status_code in (401, 403):
        raise VendorNotEntitledError(f"Finnhub refused the request ({resp.status_code})")
    resp.raise_for_status()
    items = resp.json()
    if not isinstance(items, list):
        raise ValueError(f"unexpected Finnhub response: {str(items)[:120]}")
    if not items:
        raise NoMarketDataError(ticker, symbol, "Finnhub returned no news for the window")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    limit = get_config()["news_article_limit"]

    news_str = ""
    count = 0
    for item in sorted(items, key=lambda x: x.get("datetime") or 0, reverse=True):
        ts = item.get("datetime")
        pub_date = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        # Same look-ahead guard as Yahoo news: never keep an article stamped
        # after the requested window.
        if not _in_news_window(pub_date, start_dt, end_dt):
            continue
        news_str += f"### {item.get('headline') or 'No title'} (source: {item.get('source') or 'Unknown'})\n"
        if item.get("summary"):
            news_str += f"{item['summary']}\n"
        if item.get("url"):
            news_str += f"Link: {item['url']}\n"
        news_str += "\n"
        count += 1
        if count >= limit:
            break

    if count == 0:
        return f"No news found for {ticker} between {start_date} and {end_date}"
    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"
