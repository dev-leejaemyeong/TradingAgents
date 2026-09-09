"""Manifold Markets prediction-market vendor.

Surfaces live, market-implied probabilities for forward-looking events (Fed
decisions, recession, elections, geopolitics, crypto) to the news analyst, as a
complement to news (what happened) and FRED macro data (where things stand):
what the crowd actually prices to happen next.

Default `prediction_markets` vendor as of 2026-09-09 (TODOS #114) — polymarket.py
(the original vendor) is blocked outright from this deployment's network
(HTTP 451, confirmed a Cloudflare country-block, not fixable client-side; see
polymarket.py's docstring). Manifold was chosen after live verification against
this project's actual historical query topics (2026-09-08 logs): same topical
breadth as Polymarket (macro/political/geopolitical/crypto covered; company/
sector-specific topics mostly return nothing, same known gap Polymarket already
had), no auth required, explicitly permits automated/bot read access (500
requests/min per IP), and cross-checked its Fed-meeting probabilities against
Kalshi's real-money regulated market for the same events (same order of
magnitude, same relative ordering across meetings — reasonably calibrated, not
guaranteed-accurate the way a real-money market is).

Uses Manifold's public API (https://api.manifold.markets) — no key, no auth.
``probability`` is already a 0-1 float for BINARY markets (no JSON-string-array
decoding needed, unlike Polymarket's ``outcomePrices``).

Volume is in Manifold's play-money currency ("mana", displayed as "M$"), NOT
real dollars — unlike Polymarket's real-USD volume. Labelled explicitly in the
output so the news analyst doesn't read it as real financial stakes.

Restricted to ``contractType=BINARY`` (deliberately excludes MULTIPLE_CHOICE/
BOUNTY/POLL markets) to keep parsing to a single probability per market, same
shape as Polymarket's Yes/No markets -- simplest option covering the large
majority of relevant macro/event questions.

No server-side volume sort: passing Manifold's own `sort=24-hour-vol` was
tested live and degrades topic relevance (surfaces tangentially-matching but
recently-active markets over the actual best topical match) -- same lesson
Polymarket's own design already encodes: fetch by relevance, sort by volume
client-side among already-relevant results, not the other way around.
"""
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.manifold.markets/v0"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

# Default number of markets to return, ranked by traded volume.
DEFAULT_LIMIT = 6

# How many candidates to fetch before client-side sorting/limiting -- same
# role as Polymarket's limit_per_type=20.
_SEARCH_FETCH_LIMIT = 20

# 2026-09-09, TODOS #114: below this many unique bettors, a market's
# calibration is meaningfully worse -- backed by published prediction-market
# research (contracts below ~$10k volume show "systematic biases"; Brier
# score materially worsens below a liquidity threshold) and, specific to
# Manifold itself, the platform's own finding that calibration stops
# improving somewhere between 10-20 traders (i.e. below that range it's
# still improving, so still noisier than a mature market). 15 splits that
# range; not independently back-tested against this project's own data --
# revisit once debug_events (see below) accumulates enough real observations.
_LOW_LIQUIDITY_BETTOR_THRESHOLD = 15


def _request(params: dict) -> list:
    response = requests.get(
        f"{API_BASE}/search-markets", params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _is_forward_looking(market: dict, now: datetime) -> bool:
    """Belt-and-suspenders check on top of the server-side filter=open param
    -- a plain, unfiltered search was observed live to leak resolved and
    past-closeTime markets, so this mirrors Polymarket's own defensive
    ``_is_forward_looking()`` rather than trusting the server filter alone."""
    if market.get("isResolved"):
        return False
    close_time = market.get("closeTime")
    if isinstance(close_time, (int, float)):
        if datetime.fromtimestamp(close_time / 1000, tz=timezone.utc) < now:
            return False
    return market.get("probability") is not None


def get_prediction_markets(topic: str, limit: int | None = None) -> str:
    """Return live prediction-market probabilities for an event topic.

    Args:
        topic: Event keyword(s), e.g. "Fed rate cut", "recession 2026",
            "US election", or a sector/company event.
        limit: Max markets to return (ranked by traded volume); ``None`` uses
            DEFAULT_LIMIT.

    Returns:
        A markdown report of the most-traded open markets matching the topic,
        each with its implied probability, traded volume (in mana, Manifold's
        play-money currency), and resolution date.
    """
    if limit is None:
        limit = DEFAULT_LIMIT

    try:
        data = _request({
            "term": topic,
            "filter": "open",
            "contractType": "BINARY",
            "limit": _SEARCH_FETCH_LIMIT,
        })
    except requests.RequestException as e:
        logger.warning("Manifold search failed for %r: %s", topic, e)
        return (
            f"Manifold data is currently unavailable (network error: {e}). "
            f"Proceed without prediction-market signal for '{topic}'."
        )

    now = datetime.now(timezone.utc)
    candidates = [m for m in data if _is_forward_looking(m, now)]
    candidates.sort(key=lambda m: m.get("volume") or 0, reverse=True)

    header = (
        f'## Manifold prediction markets: "{topic}"\n'
        f"Live, market-implied probabilities (higher traded volume = deeper, "
        f"more reliable). Volume is in Manifold's play-money currency (M$), "
        f"not real dollars. A probability is the crowd's priced odds of the "
        f"event, not a forecast you should take as certain.\n\n"
    )

    if not candidates:
        return header + (
            f"No open prediction markets matched '{topic}'. Manifold coverage "
            f"is concentrated in macro, political, geopolitical, and crypto "
            f"events; a specific equity may have none."
        )

    lines = []
    for m in candidates[:limit]:
        prob = m["probability"]
        volume = m.get("volume") or 0
        close_time = m.get("closeTime")
        close_date = (
            datetime.fromtimestamp(close_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if isinstance(close_time, (int, float))
            else "unknown"
        )
        bettors = m.get("uniqueBettorCount")
        low_liquidity_note = (
            " ⚠ low liquidity (few traders) — weight this less"
            if isinstance(bettors, int) and bettors < _LOW_LIQUIDITY_BETTOR_THRESHOLD
            else ""
        )
        lines.append(
            f"- **{m.get('question')}** — Yes {prob:.0%} "
            f"(M${volume:,.0f} volume, resolves {close_date}){low_liquidity_note}"
        )

    return header + "\n".join(lines) + "\n"
