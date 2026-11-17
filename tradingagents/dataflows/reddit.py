"""Reddit search fetcher for ticker-specific discussion posts.

Default path is Reddit's public Atom/RSS search feed
(``reddit.com/r/{sub}/search.rss``). The richer JSON search endpoint
(``/search.json``) is reliably WAF-blocked (``HTTP 403``) for public clients
(issue #862), and probing it on every call only doubled our request volume
against Reddit's per-IP rate limit — tripping ``429`` on the RSS fallback — so
it is kept (``_fetch_subreddit_json``) but not used by default.

All subreddits are searched in ONE combined feed (``r/a+b+c``, 2026-11-17,
upstream 241638d): anonymous RSS allows about one request per minute per IP
(upstream #1132), so a request per subreddit spent the budget three times per
ticker. Each entry names its subreddit (``atom:category``) and posts are
grouped back by it. On a 429 we back off ONCE for 60s (honouring
``Retry-After``): upstream #1295 measured that a retry at 8/10/30s still 429s
and only ~60s succeeds, so the old 5s x2 retries were wasted requests that
also consumed the limit. RSS lacks score / comment counts, so those posts
are marked and the formatter omits the metrics rather than printing fake
zeros.

No API key required (see TODOS.md #13 for the pending OAuth
registration, which needs Reddit's 2-4 week manual approval). Returns
formatted plaintext blocks ready for prompt injection and degrades
gracefully — returns a placeholder string rather than raising, so callers
never special-case missing data. A fetch that fails outright is worded as
"data unavailable", distinct from a genuinely empty result — conflating the
two previously let a rate-limited fetch read to the LLM as confirmed market
silence (2026-07-29 finding).
"""

from __future__ import annotations

import html
import http.client
import json
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 60s.

    The cap matches ``_RETRY_FALLBACK_SECONDS``: honouring less than we would
    wait on our own would spend the single retry on a request already known to
    be too early.

    Returns ``None`` when the header is absent, unparseable, negative or NaN; a
    valid ``Retry-After: 0`` returns ``0.0`` (retry at once), not ``None`` -- the
    old ``if val else None`` treated "0" as falsy and waited the 5s fallback
    anyway. Negative/NaN delays are rejected because ``time.sleep()`` raises
    ValueError on them, which would escape the 429 handler (upstream #1601).
    """
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        if val is None:
            return None
        secs = float(val)
        if secs != secs or secs < 0:   # NaN or negative
            return None
        return min(secs, 60.0)
    except (ValueError, TypeError, AttributeError):
        return None


# Headerless-429 backoff when Reddit gives no Retry-After. Measured upstream
# (#1295): a retry still 429s at 8s, 10s and 30s of spacing and succeeds at
# 60s, so a shorter wait spends the retry on a request that cannot succeed.
# Jittered so several analyses sharing an IP don't retry in lockstep and
# re-collide on the limit.
_RETRY_FALLBACK_SECONDS = 60.0

# Reddit's maximum page size. A week of posts for a ticker across the default
# subreddits fits well inside one page, which keeps a high-volume subreddit
# from crowding the others out of the combined search.
_FEED_PAGE = 100


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """Return ``seconds`` with +/-``frac`` random jitter, to desynchronize
    concurrent runs pacing against the same per-IP limit."""
    return seconds * (1.0 + random.uniform(-frac, frac))


# Reddit search feeds are small (a page of results); cap the read so a
# misbehaving endpoint can't stream an unbounded body into memory before we
# parse it. Overflow raises http.client.HTTPException, which both fetch paths
# already treat as a failed fetch (degrade to empty / RSS fallback).
_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp) -> bytes:
    """Read a response body bounded to ``_MAX_FEED_BYTES``, raising on overflow."""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(
            f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse"
        )
    return data


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    attempts_left: int = 1,
) -> list[dict] | None:
    """Default path: parse the public Atom search feed.

    ``sub`` may be one subreddit or several joined with ``+``. Carries no
    score / comment counts, so those fields are left None and the post is
    tagged ``source="rss"`` for honest display. On a 429 (Reddit's per-IP rate
    limit) we back off and retry — honouring ``Retry-After`` when present —
    up to ``attempts_left`` additional times (default one, after a ~60s wait;
    2026-11-17: see module docstring) before giving up, so a transient burst
    doesn't blank the feed.

    Returns ``None`` (not ``[]``) when every attempt failed -- distinct from
    a request that genuinely succeeded and found zero matching posts.
    Conflating the two (2026-07-29 finding) let a rate-limited fetch get
    reported to the sentiment analyst as "no posts found", which it then
    read as real market silence and folded into a bearish signal. Callers
    (fetch_reddit_posts()) must preserve this distinction rather than
    treating None like an empty list.
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and attempts_left > 0:
            # Honour a server-supplied Retry-After exactly (including 0);
            # jitter only our own fallback so concurrent runs don't retry in
            # lockstep and re-collide on the per-IP limit.
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying (%d attempt(s) left)",
                sub, ticker, wait, attempts_left,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, attempts_left=attempts_left - 1)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        category_el = entry.find("atom:category", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
            # A combined feed names each entry's subreddit; a single-subreddit
            # feed may omit it, and then it can only be that one.
            "subreddit": category_el.get("term") if category_el is not None
            else (sub if "+" not in sub else ""),
        })
    return posts


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Richer JSON search path (carries score / comment counts).

    Reddit's WAF currently returns ``403 Blocked`` on this endpoint for
    non-OAuth clients (issue #862), so it is NOT used by default — calling it on
    every request only doubled our volume against the per-IP rate limit and
    triggered 429s on the RSS fallback. Kept for the day the WAF relaxes or an
    OAuth token is wired in; degrades to RSS on failure.
    """
    url = _API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(_read_capped(resp))
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict] | None:
    """Fetch one subreddit, RSS-first.

    The JSON search endpoint is reliably WAF-blocked (403) for public clients,
    so we go straight to the RSS feed — which serves our identified User-Agent
    reliably — halving our request volume against Reddit's per-IP rate limit.
    """
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    All subreddits go out as one combined request (``r/a+b+c``, newest
    ``_FEED_PAGE`` matches) and are grouped back per subreddit by each entry's
    ``atom:category``, then cut to ``limit_per_sub`` each (the feed is
    newest-first). Nothing is dropped: an entry whose subreddit is unknown gets
    its own "unknown" block.

    A failed fetch is rendered as "<data unavailable>", never folded into
    "<no posts found>" -- the latter previously read to the sentiment analyst as
    confirmed market silence when it was actually just a failed HTTP request
    (2026-07-29 finding). A subreddit with no posts in a full page is "not among
    the newest N matches", not "no posts": the page limit may have hidden it.
    """
    subreddits = list(subreddits)
    # Crypto reaches us as a Yahoo pair (BTC-USD); search Reddit for the base
    # ("BTC") so the query actually matches discussion instead of near-nothing.
    ticker = crypto_base(ticker) or ticker
    label = ", ".join(f"r/{s}" for s in subreddits)
    fetched = _fetch_subreddit(ticker, "+".join(subreddits), _FEED_PAGE, timeout)
    if fetched is None:
        return (
            f"<Reddit data unavailable -- fetch failed ({label}; rate limit or network "
            f"error), NOT evidence of an absence of {ticker.upper()} discussion>"
        )
    if not fetched:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{label} in the past 7 days>"
        )

    def sub_of(p):
        return p.get("subreddit") or (subreddits[0] if len(subreddits) == 1 else "unknown")

    # Group by the subreddit each entry names, in the requested order.
    by_sub = {s.lower(): (s, []) for s in subreddits}
    for p in fetched:
        by_sub.setdefault(sub_of(p).lower(), (sub_of(p), []))[1].append(p)

    page_full = len(fetched) >= _FEED_PAGE
    blocks = []
    for sub, posts in by_sub.values():
        if not posts:
            blocks.append(
                f"r/{sub}: <not among the newest {_FEED_PAGE} matches across {label}>"
                if page_full else
                f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>"
            )
            continue
        posts = posts[:limit_per_sub]
        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
