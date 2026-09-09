import json
from datetime import datetime, timedelta

from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .config import get_config


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _make_api_request("NEWS_SENTIMENT", params)

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(
    symbol: str,
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """Returns recent insider transactions by key stakeholders, bounded to a
    recent window.

    Covers transactions by founders, executives, board members, etc. The raw
    endpoint has no server-side date/limit support and returns a ticker's
    ENTIRE insider-transaction history since listing -- confirmed 2026-09-09
    that this can reach hundreds of thousands of tokens for a heavily-traded,
    long-listed company (e.g. AAPL: 7,148 records back to 2004, ~645K tokens
    unfiltered), which would exceed the model's context window. Filtered and
    explicitly sorted here rather than trusting the vendor's ordering.

    Args:
        symbol: Ticker symbol. Example: "IBM".
        curr_date: Current date (yyyy-mm-dd), anchor for look_back_days.
        look_back_days: Days of history to include. ``None`` falls back to
            ``insider_transactions_lookback_days`` from the active config.
        limit: Maximum number of most-recent transactions to return. ``None``
            falls back to ``insider_transactions_limit`` from the active config.

    Returns:
        JSON string containing the filtered insider transaction data.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["insider_transactions_lookback_days"]
    if limit is None:
        limit = config["insider_transactions_limit"]

    params = {
        "symbol": symbol,
    }

    response_text = _make_api_request("INSIDER_TRANSACTIONS", params)
    transactions = json.loads(response_text).get("data", [])

    cutoff = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    recent = sorted(
        (tx for tx in transactions if tx.get("transaction_date", "") >= cutoff),
        key=lambda tx: tx.get("transaction_date", ""),
        reverse=True,
    )[:limit]

    return json.dumps({"data": recent})
