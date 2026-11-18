"""StockTwits fetch: only the past-7-days messages feed the sentiment ratio.

The stream returns the latest 30 messages whatever their age. Measured live on
2026-11-18 (59 tickers): median 7% of those 30 were within 7 days, 13/59 tickers
had none, and the oldest message's median age was 77 days -- so the Bullish/Bearish
ratio was a months-old average presented as current retail sentiment (TODOS #274).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tradingagents.dataflows import stocktwits

NOW = datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)


def _msg(age_days: float, sentiment: str | None = "Bullish", *, raw_date: str | None = None) -> dict:
    created = raw_date if raw_date is not None else (NOW - timedelta(days=age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = {"created_at": created, "body": f"post {age_days}", "user": {"username": "u"}}
    if sentiment:
        m["entities"] = {"sentiment": {"basic": sentiment}}
    return m


def _resp(messages):
    payload = json.dumps({"messages": messages}).encode("utf-8")

    class _R:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return payload
    return _R()


def _fetch(messages, **kw):
    with patch.object(stocktwits, "urlopen", return_value=_resp(messages)):
        return stocktwits.fetch_stocktwits_messages("NVDA", now=NOW, **kw)


@pytest.mark.unit
class TestSevenDayWindow:
    def test_all_recent_messages_are_used_and_counted(self):
        out = _fetch([_msg(0.5), _msg(1, "Bearish"), _msg(3, None)])
        assert "3 of 3" in out
        assert "Bullish: 1 (33%)" in out and "Bearish: 1 (33%)" in out and "Unlabeled: 1" in out

    def test_old_messages_are_dropped_from_the_ratio(self):
        # 2 recent (1 bull, 1 bear) + 3 months-old bullish: the ratio must be 50/50, not 80/20.
        out = _fetch([_msg(1, "Bullish"), _msg(2, "Bearish"),
                      _msg(90, "Bullish"), _msg(95, "Bullish"), _msg(120, "Bullish")])
        assert "2 of 5" in out
        assert "Bullish: 1 (50%)" in out and "Bearish: 1 (50%)" in out
        assert "post 90" not in out and "post 120" not in out

    def test_boundary_exactly_seven_days_is_in_one_second_older_is_out(self):
        inside = _msg(0, raw_date=(NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        outside = _msg(0, raw_date=(NOW - timedelta(days=7, seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        out = _fetch([inside, outside])
        assert "1 of 2" in out

    def test_nothing_recent_gives_a_placeholder_without_a_ratio(self):
        out = _fetch([_msg(13, "Bullish"), _msg(40, "Bullish"), _msg(133, "Bearish")])
        assert out.startswith("<no StockTwits messages for $NVDA in the past 7 days")
        assert "13 days old" in out                       # the newest of the fetched
        assert "no bullish/bearish ratio" in out
        assert "Bullish:" not in out and "Bearish:" not in out   # no ratio line at all

    def test_unreadable_dates_are_not_trusted_as_recent(self):
        out = _fetch([_msg(1, "Bullish"), _msg(0, raw_date="garbage"), _msg(0, raw_date="")])
        assert "1 of 3" in out and "Bullish: 1 (100%)" in out

    def test_all_dates_unreadable(self):
        out = _fetch([_msg(0, raw_date="garbage"), _msg(0, raw_date="n/a")])
        assert out.startswith("<no StockTwits messages for $NVDA in the past 7 days")
        assert "has a readable date" in out and "none of the 2" in out

    def test_naive_timestamp_is_treated_as_utc(self):
        naive = (NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        assert "1 of 1" in _fetch([_msg(0, raw_date=naive)])

    def test_slightly_future_timestamp_counts_as_recent(self):
        future = (NOW + timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert "1 of 1" in _fetch([_msg(0, raw_date=future)])

    def test_limit_still_caps_what_is_fetched_before_the_window(self):
        out = _fetch([_msg(0.1) for _ in range(40)], limit=30)
        assert "30 of 30" in out

    def test_empty_stream_keeps_the_existing_message(self):
        assert _fetch([]) == "<no StockTwits messages found for $NVDA>"

    def test_custom_window(self):
        out = _fetch([_msg(10), _msg(20)], window_days=14)
        assert "1 of 2" in out
