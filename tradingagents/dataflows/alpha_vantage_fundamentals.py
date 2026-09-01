import json

from .alpha_vantage_common import _make_api_request


def _filter_reports_by_date(result, curr_date: str):
    """Drop annual/quarterly reports dated after curr_date to prevent look-ahead.

    ``_make_api_request`` returns the fundamentals payload as a JSON string, so
    parse, filter, and re-serialize. A non-JSON body or an unset ``curr_date`` is
    returned unchanged.
    """
    if not curr_date or not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if isinstance(payload.get(key), list):
            payload[key] = [
                r for r in payload[key]
                if r.get("fiscalDateEnding", "") <= curr_date
            ]
    return json.dumps(payload)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_earnings_calendar(symbol: str, horizon: str = "3month") -> str:
    """Next scheduled earnings report date and consensus EPS estimate.

    TODOS.md #89: forward-looking (the next NOT-yet-reported date), distinct
    from ``get_income_statement()``/``get_recent_negative_earnings_surprise()``
    (already-reported results) -- context only, not a hard gate, so an
    analyst can flag "position entry lands right before an earnings report"
    as a risk factor without the system automatically excluding/resizing
    anything on it (no backtested evidence yet that doing so helps).

    Args:
        symbol: Ticker symbol.
        horizon: How far forward to look: "3month" (default), "6month", or "12month".

    Returns:
        CSV: symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay
        -- already scoped to this one symbol by Alpha Vantage.
    """
    return _make_api_request(
        "EARNINGS_CALENDAR", {"symbol": symbol, "horizon": horizon}
    )


def get_recent_negative_earnings_surprise(symbol: str, lookback_days: int = 30) -> float | None:
    """The surprise fraction if the most recent past earnings report (within
    ``lookback_days`` of now) had a negative surprise, else None.

    Alpha Vantage EARNINGS fallback for screener.py's
    ``recent_negative_earnings_surprise()`` (TODOS.md #87) -- mirrors that
    function's exact contract (float|None, negative-only, "no surprise" and
    "any failure" both collapse to None) so callers can't tell which vendor
    answered. ``quarterlyEarnings`` is already sorted most-recent-first, so
    the first entry that has actually been reported (``reportedDate`` not in
    the future) settles the answer -- if it's older than the lookback
    window, nothing newer qualifies either.

    Args:
        symbol: Ticker symbol.
        lookback_days: Only consider a report dated within this many days of now.

    Returns:
        The surprise fraction (e.g. -0.065 for a 6.5% miss), or None.
    """
    from datetime import datetime, timedelta, timezone

    try:
        payload = json.loads(_make_api_request("EARNINGS", {"symbol": symbol}))
        quarterly = payload.get("quarterlyEarnings") or []
    except Exception:  # noqa: BLE001 - fallback path; caller treats any failure as "no surprise"
        return None

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    for report in quarterly:
        reported_date_str = report.get("reportedDate")
        if not reported_date_str:
            continue
        try:
            reported_date = datetime.strptime(reported_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if reported_date > now:
            continue  # not yet actually reported -- keep looking for a real one
        if reported_date < cutoff:
            return None  # most-recent-first: nothing newer qualifies either
        surprise_pct = report.get("surprisePercentage")
        try:
            surprise_pct = float(surprise_pct)
        except (TypeError, ValueError):
            return None
        return surprise_pct / 100.0 if surprise_pct < 0 else None
    return None

