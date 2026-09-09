"""Fed rate-decision probabilities from Kalshi's KXFEDDECISION market.

2026-09-09, TODOS #114. Not a general-purpose prediction-market vendor (see
manifold.py for that) -- Kalshi has no full-text search endpoint (confirmed:
no ``/search`` route, only listing endpoints filtered by ``series_ticker``/
``event_ticker``/tags), so it can't serve arbitrary LLM-chosen topics the way
Manifold's ``search-markets`` can. It IS a clean fit for this one narrow,
well-known question: query the single known series ticker
``KXFEDDECISION`` directly, no search needed.

Chosen over reconstructing a probability from raw CME Fed Funds futures
pricing (yfinance) + a hand-written day-weighted-average formula: live
verification (2026-09-09) found that approach's accuracy degrades for
meetings later in a contract month (computed 42.8% hike probability for
Kalshi's own real market pricing 27-28% for the same event -- the small
"days after the meeting" denominator for that month amplified estimation
error). Kalshi's contracts already ARE the finished, real-money-backed
probability for exactly this question -- no derivation, no formula to get
subtly wrong.

Also means no FOMC meeting-date list needs to be hardcoded/maintained here:
Kalshi's own open KXFEDDECISION listings implicitly ARE the calendar (the
earliest close_time among them is the next meeting), so this stays accurate
automatically as meetings resolve and new ones list -- no staleness/expiry
risk, no scheduled-maintenance task needed.

Uses Kalshi's public trade API (https://api.elections.kalshi.com) -- no key,
no account. Reading public market data (not trading) requires no
authentication by design, consistent with every other vendor in this
project. Verified live: 60 open KXFEDDECISION markets, ~112KB response
(not the ~14,000-series full catalog -- server-side ``series_ticker``
filtering keeps this small and targeted).
"""
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

SERIES_TICKER = "KXFEDDECISION"

# Ticker suffix -> human label. Kalshi's own ticker format is
# "KXFEDDECISION-<YYMON>-<SUFFIX>" (confirmed live, e.g.
# "KXFEDDECISION-26SEP-H25"); order is most-cut to most-hike for stable,
# readable rendering regardless of dict/response ordering.
_OUTCOME_LABELS = {
    "C26": "Cut >25bps",
    "C25": "Cut 25bps",
    "H0": "Hold (no change)",
    "H25": "Hike 25bps",
    "H26": "Hike >25bps",
}
_OUTCOME_ORDER = ["C26", "C25", "H0", "H25", "H26"]


def _request(params: dict) -> dict:
    response = requests.get(f"{API_BASE}/markets", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_fed_rate_probability() -> str:
    """Return live, real-money market-implied probabilities for the next
    FOMC rate decision.

    Returns:
        A markdown report of each outcome (cut/hold/hike, by size) with its
        current market-implied probability (bid/ask midpoint), for the
        nearest upcoming FOMC meeting.
    """
    try:
        data = _request({"series_ticker": SERIES_TICKER, "status": "open", "limit": 200})
    except requests.RequestException as e:
        logger.warning("Kalshi Fed rate lookup failed: %s", e)
        return (
            f"Fed rate decision data is currently unavailable (network error: {e}). "
            f"Proceed without this signal."
        )

    markets = data.get("markets", [])
    if not markets:
        # Either a transient gap right after a meeting resolves and before
        # the next one lists, or Kalshi renamed/discontinued this series
        # ticker (they've done this before -- e.g. KXGOVTSHUTDOWN/
        # KXSHUTDOWNBYDATE/KXGOVSHUT coexist for the same government-shutdown
        # topic). Either way: degrade to unavailable, don't crash.
        return (
            "No open Fed rate decision markets found on Kalshi (series "
            f"'{SERIES_TICKER}') -- possibly a gap between meetings, or the "
            "series was renamed. Proceed without this signal."
        )

    # Earliest close_time across all returned legs = the next upcoming
    # meeting; each meeting has up to 5 legs (one per outcome) sharing that
    # same close_time.
    next_close = min(m["close_time"] for m in markets)
    meeting_markets = [m for m in markets if m["close_time"] == next_close]

    by_outcome = {}
    for m in meeting_markets:
        suffix = m["ticker"].rsplit("-", 1)[-1]
        by_outcome[suffix] = m

    meeting_date = datetime.fromisoformat(
        next_close.replace("Z", "+00:00")
    ).strftime("%Y-%m-%d")
    header = (
        f"## Fed rate decision odds (Kalshi, real-money regulated market)\n"
        f"Next FOMC meeting: {meeting_date}. Each probability is the "
        f"market's live implied odds for that outcome (bid/ask midpoint) — "
        f"a real-money regulated exchange (CFTC), not crowd/play-money "
        f"betting.\n\n"
    )

    lines = []
    for suffix in _OUTCOME_ORDER:
        m = by_outcome.get(suffix)
        if m is None:
            continue
        # Kalshi returns these as dollar-formatted strings (e.g. "0.1000"),
        # not floats -- confirmed live 2026-09-09.
        bid = float(m.get("yes_bid_dollars") or 0.0)
        ask = float(m.get("yes_ask_dollars") or 0.0)
        mid = (bid + ask) / 2
        lines.append(f"- **{_OUTCOME_LABELS[suffix]}**: {mid:.0%}")

    if not lines:
        return header + "No priced outcomes available for this meeting."

    return header + "\n".join(lines) + "\n"
