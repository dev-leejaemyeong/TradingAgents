"""get_insider_transactions must never return a ticker's entire since-listing
history unfiltered -- confirmed 2026-09-09 that the Alpha Vantage endpoint has
no server-side date/limit support and can return hundreds of thousands of
tokens for a heavily-traded, long-listed company (AAPL: 7,148 records back to
2004, ~645K tokens unfiltered; CRWD: 493,969 tokens), which exceeds the
model's context window. Both vendor paths must bound to a recent window and
must not trust the vendor's row ordering when truncating to `limit`.
"""
import json

import pandas as pd
import pytest

import tradingagents.dataflows.alpha_vantage_news as avnews
import tradingagents.dataflows.y_finance as yfin


def _tx(date, executive="DOE, JANE"):
    return {
        "transaction_date": date,
        "ticker": "AAPL",
        "executive": executive,
        "executive_title": "Director",
        "security_type": "Common Stock",
        "acquisition_or_disposal": "D",
        "shares": "100.0",
        "share_price": "50.0",
    }


class TestAlphaVantageInsiderTransactions:
    @pytest.mark.unit
    def test_filters_out_transactions_older_than_lookback_window(self, monkeypatch):
        raw = json.dumps({"data": [_tx("2026-09-01"), _tx("2020-01-01"), _tx("2004-03-24")]})
        monkeypatch.setattr(avnews, "_make_api_request", lambda *a, **k: raw)
        out = json.loads(avnews.get_insider_transactions("AAPL", "2026-09-09", look_back_days=90))
        dates = [tx["transaction_date"] for tx in out["data"]]
        assert dates == ["2026-09-01"]

    @pytest.mark.unit
    def test_caps_at_limit_even_within_the_lookback_window(self, monkeypatch):
        raw = json.dumps({"data": [_tx(f"2026-08-{d:02d}") for d in range(1, 21)]})  # 20 recent records
        monkeypatch.setattr(avnews, "_make_api_request", lambda *a, **k: raw)
        out = json.loads(avnews.get_insider_transactions("AAPL", "2026-09-09", look_back_days=90, limit=5))
        assert len(out["data"]) == 5

    @pytest.mark.unit
    def test_does_not_trust_vendor_ordering_when_truncating(self, monkeypatch):
        # Deliberately scrambled order (oldest first, out of order) -- the
        # limit must still keep the most RECENT rows, not whatever happens to
        # be first in the vendor's response.
        raw = json.dumps({"data": [_tx("2026-08-01"), _tx("2026-08-15"), _tx("2026-08-05")]})
        monkeypatch.setattr(avnews, "_make_api_request", lambda *a, **k: raw)
        out = json.loads(avnews.get_insider_transactions("AAPL", "2026-09-09", look_back_days=90, limit=2))
        dates = [tx["transaction_date"] for tx in out["data"]]
        assert dates == ["2026-08-15", "2026-08-05"]

    @pytest.mark.unit
    def test_omitted_params_fall_back_to_config_defaults(self, monkeypatch):
        raw = json.dumps({"data": [_tx("2026-09-01"), _tx("2026-01-01")]})
        monkeypatch.setattr(avnews, "_make_api_request", lambda *a, **k: raw)
        monkeypatch.setattr(
            avnews,
            "get_config",
            lambda: {"insider_transactions_lookback_days": 30, "insider_transactions_limit": 50},
        )
        out = json.loads(avnews.get_insider_transactions("AAPL", "2026-09-09"))
        dates = [tx["transaction_date"] for tx in out["data"]]
        assert dates == ["2026-09-01"]  # 2026-01-01 is outside the 30-day config default

    @pytest.mark.unit
    def test_empty_history_returns_empty_data_list(self, monkeypatch):
        monkeypatch.setattr(avnews, "_make_api_request", lambda *a, **k: json.dumps({"data": []}))
        out = json.loads(avnews.get_insider_transactions("RDDT", "2026-09-09"))
        assert out["data"] == []


class TestYFinanceInsiderTransactions:
    def _fake_ticker(self, monkeypatch, df):
        class _FakeTicker:
            def __init__(self, *_args, **_kwargs):
                self.insider_transactions = df

        monkeypatch.setattr(yfin.yf, "Ticker", _FakeTicker)
        monkeypatch.setattr(yfin, "yf_retry", lambda fn: fn())
        monkeypatch.setattr(yfin, "normalize_symbol", lambda t: t)

    @pytest.mark.unit
    def test_filters_by_lookback_window_and_caps_at_limit(self, monkeypatch):
        df = pd.DataFrame({
            "Start Date": ["2026-09-01", "2026-08-20", "2020-01-01"],
            "Shares": [100, 200, 300],
        })
        self._fake_ticker(monkeypatch, df)
        out = yfin.get_insider_transactions("AAPL", "2026-09-09", look_back_days=90, limit=1)
        assert "2026-09-01" in out
        assert "2026-08-20" not in out  # trimmed by limit=1, most recent kept
        assert "2020-01-01" not in out  # trimmed by lookback window

    @pytest.mark.unit
    def test_empty_after_lookback_filter_reports_plainly(self, monkeypatch):
        df = pd.DataFrame({"Start Date": ["2004-03-24"], "Shares": [100]})
        self._fake_ticker(monkeypatch, df)
        out = yfin.get_insider_transactions("AAPL", "2026-09-09", look_back_days=90)
        assert "No insider transactions in the last 90 days" in out

    @pytest.mark.unit
    def test_no_data_at_all_reports_plainly(self, monkeypatch):
        self._fake_ticker(monkeypatch, pd.DataFrame())
        out = yfin.get_insider_transactions("ZZZZ", "2026-09-09")
        assert "No insider transactions reported" in out


class TestInsiderTransactionsToolWiring:
    @pytest.mark.unit
    def test_tool_passes_curr_date_and_overrides_through_to_vendor_routing(self, monkeypatch):
        import tradingagents.agents.utils.news_data_tools as tools_mod

        captured = {}

        def fake_route(method, *args, **kwargs):
            captured["method"] = method
            captured["args"] = args
            return "ok"

        monkeypatch.setattr(tools_mod, "route_to_vendor", fake_route)
        tools_mod.get_insider_transactions.func("AAPL", "2026-09-09", look_back_days=30, limit=10)
        assert captured["method"] == "get_insider_transactions"
        assert captured["args"] == ("AAPL", "2026-09-09", 30, 10)
