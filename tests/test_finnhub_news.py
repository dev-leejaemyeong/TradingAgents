"""Finnhub company-news vendor (2026-10-12): output shape, look-ahead guard,
and the vendor errors that let news_data's chain fall through to yfinance."""
from datetime import datetime, timezone

import pytest
import requests

import tradingagents.dataflows.finnhub_news as fh
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorNotEntitledError,
    VendorRateLimitError,
)


def _epoch(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


@pytest.fixture
def api(monkeypatch):
    state = {"payload": [], "status": 200, "calls": []}
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    monkeypatch.setattr(fh, "FINNHUB_MIN_INTERVAL_SECONDS", 0.0)

    def fake_get(url, params=None, timeout=None):
        state["calls"].append(params)
        return _Resp(state["payload"], state["status"])

    monkeypatch.setattr(fh.requests, "get", fake_get)
    return state


@pytest.mark.unit
def test_formats_newest_first_with_summary_and_source(api):
    api["payload"] = [
        {"headline": "Old", "summary": "s-old", "source": "A", "url": "u1", "datetime": _epoch("2026-10-02 10:00")},
        {"headline": "New", "summary": "s-new", "source": "B", "url": "u2", "datetime": _epoch("2026-10-08 10:00")},
    ]
    out = fh.get_news_finnhub("lyft", "2026-10-01", "2026-10-09")
    assert out.startswith("## lyft News, from 2026-10-01 to 2026-10-09")
    assert out.index("### New (source: B)") < out.index("### Old (source: A)")
    assert "s-new" in out and "Link: u2" in out
    assert api["calls"][0]["symbol"] == "LYFT"


@pytest.mark.unit
def test_article_after_the_window_is_dropped(api):
    api["payload"] = [
        {"headline": "Future", "summary": "", "datetime": _epoch("2026-10-11 00:00")},
        {"headline": "Inside", "summary": "", "datetime": _epoch("2026-10-09 23:59")},
    ]
    out = fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")
    assert "Inside" in out and "Future" not in out


@pytest.mark.unit
def test_only_out_of_window_articles_is_a_no_news_message(api):
    api["payload"] = [{"headline": "Future", "summary": "", "datetime": _epoch("2026-10-20 00:00")}]
    assert fh.get_news_finnhub("X", "2026-10-01", "2026-10-09").startswith("No news found for X")


@pytest.mark.unit
def test_article_limit_is_respected(api):
    set_config({"news_article_limit": 2})
    try:
        api["payload"] = [{"headline": f"H{i}", "datetime": _epoch(f"2026-10-0{i} 10:00")} for i in range(1, 6)]
        assert fh.get_news_finnhub("X", "2026-10-01", "2026-10-09").count("### ") == 2
    finally:
        set_config({"news_article_limit": 20})


@pytest.mark.unit
@pytest.mark.parametrize("status,error", [(429, VendorRateLimitError), (401, VendorNotEntitledError),
                                          (403, VendorNotEntitledError)])
def test_http_status_maps_to_router_errors(api, status, error):
    api["status"] = status
    with pytest.raises(error):
        fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")


@pytest.mark.unit
def test_empty_result_is_no_market_data(api):
    with pytest.raises(NoMarketDataError):
        fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")


@pytest.mark.unit
def test_missing_key_is_not_configured_and_makes_no_call(api, monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY")
    with pytest.raises(VendorNotConfiguredError):
        fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")
    assert api["calls"] == []


@pytest.mark.unit
def test_error_payload_raises(api):
    api["payload"] = {"error": "Invalid API key"}
    with pytest.raises(ValueError, match="unexpected Finnhub response"):
        fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")


@pytest.mark.unit
def test_calls_are_spaced_by_the_rate_limit(api, monkeypatch):
    sleeps = []
    monkeypatch.setattr(fh, "FINNHUB_MIN_INTERVAL_SECONDS", 1.1)
    monkeypatch.setattr(fh.time, "sleep", lambda s: sleeps.append(s))
    api["payload"] = [{"headline": "H", "datetime": _epoch("2026-10-05 10:00")}]
    for _ in range(3):
        fh.get_news_finnhub("X", "2026-10-01", "2026-10-09")
    assert len(sleeps) >= 2 and all(0 < s <= 1.1 for s in sleeps)


@pytest.mark.unit
def test_empty_finnhub_falls_through_to_yfinance(monkeypatch):
    set_config({"data_vendors": {"news_data": "finnhub,yfinance"}})

    def empty(*a, **k):
        raise NoMarketDataError("X")

    monkeypatch.setitem(interface.VENDOR_METHODS["get_news"], "finnhub", empty)
    monkeypatch.setitem(interface.VENDOR_METHODS["get_news"], "yfinance", lambda *a, **k: "YF_NEWS")
    assert interface.route_to_vendor("get_news", "X", "2026-10-01", "2026-10-09") == "YF_NEWS"


@pytest.mark.unit
def test_finnhub_answer_is_used_when_it_has_news(monkeypatch):
    set_config({"data_vendors": {"news_data": "finnhub,yfinance"}})

    def must_not_call(*a, **k):
        raise AssertionError("fallback must not be called")

    monkeypatch.setitem(interface.VENDOR_METHODS["get_news"], "finnhub", lambda *a, **k: "FH_NEWS")
    monkeypatch.setitem(interface.VENDOR_METHODS["get_news"], "yfinance", must_not_call)
    assert interface.route_to_vendor("get_news", "X", "2026-10-01", "2026-10-09") == "FH_NEWS"
