"""Alpha Vantage request hardening.

Regressions for #990 (no request timeout -> can hang), #991 (invalid-key
responses mislabeled as rate limits and silently treated as transient),
#1115 (fundamentals look-ahead filter never ran because the payload is a JSON
string, not a dict), and TODOS.md #86 (2026-08-31: a premium-endpoint
response -- e.g. TIME_SERIES_DAILY_ADJUSTED without a paid plan -- was
mislabeled as a rate limit, making a permanent entitlement gap look
transient).
"""
import json

import pytest

import tradingagents.dataflows.alpha_vantage_common as av
import tradingagents.dataflows.alpha_vantage_fundamentals as avf
import tradingagents.dataflows.alpha_vantage_stock as avs


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _patched_get(body, capture=None):
    def fake_get(url, params=None, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return _FakeResponse(body)
    return fake_get


@pytest.mark.unit
def test_request_passes_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(av.requests, "get", _patched_get("Date,Close\n2025-01-02,1.0", captured))
    av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    assert captured.get("timeout") == av.REQUEST_TIMEOUT  # #990


@pytest.mark.unit
def test_rate_limit_detected(monkeypatch):
    body = '{"Information": "Our standard API rate limit is 25 requests per day. ... your API key ..."}'
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageRateLimitError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_invalid_key_not_mislabeled_as_rate_limit(monkeypatch):
    # AV's invalid-key notice mentions "API key"; it must NOT be treated as a
    # (transient) rate limit, but surface as a real configuration error (#991).
    body = ('{"Information": "the parameter apikey is invalid or missing. '
            'Please claim your free API key on (https://www.alphavantage.co/support/#api-key)."}')
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotConfiguredError):
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
    with pytest.raises(av.AlphaVantageRateLimitError):  # sanity: rate-limit path still distinct
        monkeypatch.setattr(av.requests, "get", _patched_get('{"Note": "API call frequency is 5 calls per minute."}'))
        av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})


@pytest.mark.unit
def test_premium_endpoint_not_mislabeled_as_rate_limit(monkeypatch):
    # TODOS.md #86 (2026-08-31 live probe against TIME_SERIES_DAILY_ADJUSTED):
    # a premium-only endpoint is a permanent entitlement gap, not a transient
    # throttle -- retrying (or an approved higher daily quota) never fixes it.
    body = (
        '{"Information": "Thank you for using Alpha Vantage! This is a premium '
        'endpoint. You may subscribe to any of the premium plans at '
        'https://www.alphavantage.co/premium/ to instantly unlock all premium '
        'endpoints"}'
    )
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    with pytest.raises(av.AlphaVantageNotEntitledError):
        av._make_api_request("TIME_SERIES_DAILY_ADJUSTED", {"symbol": "AAPL"})


@pytest.mark.unit
def test_get_daily_close_parses_latest_row(monkeypatch):
    # TODOS.md #90: entrypoint.fetch_current_prices()'s Alpha Vantage
    # fallback -- TIME_SERIES_DAILY (free, unlike ADJUSTED) is CSV with the
    # most recent row first.
    body = (
        "timestamp,open,high,low,close,volume\r\n"
        "2026-08-31,319.5400,321.2350,312.8000,316.8500,41209669\r\n"
        "2026-08-28,316.8450,322.3700,315.4504,319.7000,38649398\r\n"
    )
    monkeypatch.setattr(av.requests, "get", _patched_get(body))
    assert avs.get_daily_close("AAPL") == 316.85


@pytest.mark.unit
def test_get_daily_close_raises_on_no_data_row(monkeypatch):
    monkeypatch.setattr(av.requests, "get", _patched_get("timestamp,open,high,low,close,volume\r\n"))
    with pytest.raises(ValueError, match="no data"):
        avs.get_daily_close("BOGUS")


_FUNDAMENTALS_JSON = json.dumps({
    "symbol": "AAPL",
    "annualReports": [
        {"fiscalDateEnding": "2025-12-31", "totalAssets": "1"},   # future -> must drop
        {"fiscalDateEnding": "2023-12-31", "totalAssets": "2"},   # past   -> must keep
    ],
    "quarterlyReports": [
        {"fiscalDateEnding": "2024-06-30", "totalAssets": "3"},   # future -> must drop
        {"fiscalDateEnding": "2023-09-30", "totalAssets": "4"},   # past   -> must keep
    ],
})


@pytest.mark.unit
def test_fundamentals_look_ahead_filter_runs_on_json_string(monkeypatch):
    # #1115: the payload arrives as a JSON *string*; the old dict-only guard let
    # future-dated fiscal periods leak into historical runs.
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: _FUNDAMENTALS_JSON)
    out = avf.get_balance_sheet("AAPL", curr_date="2024-01-01")
    assert isinstance(out, str)  # callers still receive a str
    parsed = json.loads(out)
    assert [r["fiscalDateEnding"] for r in parsed["annualReports"]] == ["2023-12-31"]
    assert [r["fiscalDateEnding"] for r in parsed["quarterlyReports"]] == ["2023-09-30"]


@pytest.mark.unit
def test_fundamentals_no_curr_date_passes_through(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: _FUNDAMENTALS_JSON)
    assert avf.get_income_statement("AAPL") == _FUNDAMENTALS_JSON


@pytest.mark.unit
def test_fundamentals_non_json_body_unchanged(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    assert avf.get_cashflow("AAPL", curr_date="2024-01-01") == "not-json"


# ---------------------------------------------------------------------------
# Date trim (see the rationale on the unguarded trim in alpha_vantage_common,
# ported from upstream 2026-09-08)
# ---------------------------------------------------------------------------

_DAILY_CSV = (
    "timestamp,open,high,low,close,volume\n"
    "2024-05-13,1,1,1,1,10\n"   # after end_date -> must never be served
    "2024-05-10,1,1,1,1,10\n"
    "2024-05-09,1,1,1,1,10\n"
)


@pytest.mark.unit
def test_stock_data_is_trimmed_to_the_requested_window(monkeypatch):
    monkeypatch.setattr(avs, "_make_api_request", lambda *a, **k: _DAILY_CSV)
    out = avs.get_stock("IBM", "2024-05-09", "2024-05-10")
    assert "2024-05-10" in out and "2024-05-09" in out
    assert "2024-05-13" not in out, "bar after end_date leaked into the window"


@pytest.mark.unit
def test_unparseable_body_is_never_served_untrimmed(monkeypatch):
    """The trim used to swallow the failure and return the whole body, putting
    bars after end_date into a backtest. It must raise instead."""
    monkeypatch.setattr(avs, "_make_api_request",
                        lambda *a, **k: "timestamp,close\nnot-a-date,1\n")

    with pytest.raises(ValueError):
        avs.get_stock("IBM", "2024-05-09", "2024-05-10")


@pytest.mark.unit
def test_empty_body_still_passes_through(monkeypatch):
    monkeypatch.setattr(avs, "_make_api_request", lambda *a, **k: "")
    assert avs.get_stock("IBM", "2024-05-09", "2024-05-10") == ""


def _earnings_payload(reports):
    return json.dumps({"symbol": "AAPL", "quarterlyEarnings": reports})


@pytest.mark.unit
def test_get_recent_negative_earnings_surprise_returns_fraction(monkeypatch):
    # TODOS.md #87: screener.py's Alpha Vantage fallback for a negative
    # earnings surprise within the lookback window.
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda fn, params: _earnings_payload([{"reportedDate": recent, "surprisePercentage": "-6.5"}]),
    )
    assert avf.get_recent_negative_earnings_surprise("AAPL") == pytest.approx(-0.065)


@pytest.mark.unit
def test_get_recent_negative_earnings_surprise_positive_surprise_is_none(monkeypatch):
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda fn, params: _earnings_payload([{"reportedDate": recent, "surprisePercentage": "3.2"}]),
    )
    assert avf.get_recent_negative_earnings_surprise("AAPL") is None


@pytest.mark.unit
def test_get_recent_negative_earnings_surprise_outside_lookback_is_none(monkeypatch):
    from datetime import datetime, timedelta, timezone
    stale = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda fn, params: _earnings_payload([{"reportedDate": stale, "surprisePercentage": "-30.0"}]),
    )
    assert avf.get_recent_negative_earnings_surprise("AAPL", lookback_days=30) is None


@pytest.mark.unit
def test_get_recent_negative_earnings_surprise_skips_future_dated_report(monkeypatch):
    # quarterlyEarnings can carry a not-yet-reported entry first; the
    # fallback must skip forward to the actual most recent past report.
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d")
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    monkeypatch.setattr(
        avf, "_make_api_request",
        lambda fn, params: _earnings_payload([
            {"reportedDate": future, "surprisePercentage": None},
            {"reportedDate": recent, "surprisePercentage": "-1.1"},
        ]),
    )
    assert avf.get_recent_negative_earnings_surprise("AAPL") == pytest.approx(-0.011)


@pytest.mark.unit
def test_get_recent_negative_earnings_surprise_no_data_is_none(monkeypatch):
    monkeypatch.setattr(avf, "_make_api_request", lambda fn, params: "not-json")
    assert avf.get_recent_negative_earnings_surprise("BOGUS") is None


@pytest.mark.unit
def test_get_earnings_calendar_passes_through_and_scopes_horizon(monkeypatch):
    # TODOS.md #89: forward-looking next report date, distinct from the
    # already-reported EARNINGS data above.
    captured = {}

    def _fake_request(fn, params):
        captured["fn"] = fn
        captured["params"] = params
        return "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\r\nAAPL,APPLE INC,2026-10-29,2026-09-30,1.98,USD,\r\n"

    monkeypatch.setattr(avf, "_make_api_request", _fake_request)
    result = avf.get_earnings_calendar("AAPL", horizon="6month")
    assert "2026-10-29" in result
    assert captured["fn"] == "EARNINGS_CALENDAR"
    assert captured["params"] == {"symbol": "AAPL", "horizon": "6month"}
