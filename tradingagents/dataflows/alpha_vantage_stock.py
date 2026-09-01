from datetime import datetime

from .alpha_vantage_common import _filter_csv_by_date_range, _make_api_request


def get_stock(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Returns raw daily OHLCV values, adjusted close values, and historical split/dividend events
    filtered to the specified date range.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily adjusted time series data filtered to the date range.
    """
    # Parse dates to determine the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 days
    # Compact returns latest 100 data points, so check if start_date is recent enough
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)


def get_daily_close(symbol: str) -> float:
    """Latest daily close price via TIME_SERIES_DAILY (not ADJUSTED).

    A lightweight, non-premium alternative to ``get_stock()``:
    TIME_SERIES_DAILY_ADJUSTED requires a paid plan (TODOS.md #86, "This is
    a premium endpoint" regardless of daily-request quota), while the plain
    TIME_SERIES_DAILY is free. No history, no split/dividend adjustment --
    just the single number entrypoint.fetch_current_prices() needs as its
    Alpha Vantage fallback when yfinance fails a stop-loss/take-profit
    price check (TODOS.md #90).

    Args:
        symbol: The ticker symbol.

    Returns:
        The most recent close price.

    Raises:
        ValueError: no parseable data row in the response.
        AlphaVantageRateLimitError / AlphaVantageNotEntitledError /
        AlphaVantageNotConfiguredError: propagated from ``_make_api_request``.
    """
    response = _make_api_request(
        "TIME_SERIES_DAILY",
        {"symbol": symbol, "outputsize": "compact", "datatype": "csv"},
    )
    # Header: timestamp,open,high,low,close,volume -- most recent row first.
    lines = response.strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"Alpha Vantage TIME_SERIES_DAILY returned no data for {symbol!r}")
    fields = lines[1].split(",")
    return float(fields[4])
