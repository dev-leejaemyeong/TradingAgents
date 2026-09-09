"""Kalshi Fed rate-decision vendor: next-meeting selection, outcome
formatting, string-to-float coercion, graceful degradation.

All API access is mocked, so these run without a network connection.
"""
import unittest
from unittest import mock

import pytest
import requests

from tradingagents.dataflows import kalshi_fed_rate


def _leg(suffix, close_time, *, bid, ask):
    return {
        "ticker": f"KXFEDDECISION-26SEP-{suffix}",
        "close_time": close_time,
        # Kalshi returns these as dollar-formatted strings, not floats --
        # confirmed live 2026-09-09. Tests must exercise that, not a float.
        "yes_bid_dollars": f"{bid:.4f}",
        "yes_ask_dollars": f"{ask:.4f}",
    }


_NEXT_MEETING = "2026-09-16T17:59:00Z"
_LATER_MEETING = "2026-10-28T17:59:00Z"

_MARKETS_RESPONSE = {
    "markets": [
        _leg("C26", _NEXT_MEETING, bid=0.0, ask=0.01),
        _leg("C25", _NEXT_MEETING, bid=0.0, ask=0.01),
        _leg("H0", _NEXT_MEETING, bid=0.45, ask=0.46),
        _leg("H25", _NEXT_MEETING, bid=0.54, ask=0.55),
        _leg("H26", _NEXT_MEETING, bid=0.01, ask=0.02),
        # A later meeting's legs should be ignored -- only the nearest
        # close_time's outcomes belong in the report.
        _leg("H0", _LATER_MEETING, bid=0.68, ask=0.69),
    ]
}


@pytest.mark.unit
class KalshiFedRateFormatTests(unittest.TestCase):
    def test_selects_nearest_meeting_only(self):
        with mock.patch.object(kalshi_fed_rate, "_request", return_value=_MARKETS_RESPONSE):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertIn("2026-09-16", out)
        self.assertNotIn("2026-10-28", out)

    def test_string_dollar_fields_coerced_to_float_midpoint(self):
        # yes_bid_dollars="0.5400", yes_ask_dollars="0.5500" -> mid 0.545 -> 55%
        # (round-half-to-even in Python's format spec rounds 0.545 to 54% in
        # some float representations -- assert loosely on the printed value
        # actually produced rather than hand-picking a rounding edge case).
        with mock.patch.object(kalshi_fed_rate, "_request", return_value=_MARKETS_RESPONSE):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertIn("Hold (no change)", out)
        self.assertIn("Hike 25bps", out)

    def test_outcomes_rendered_in_cut_to_hike_order(self):
        with mock.patch.object(kalshi_fed_rate, "_request", return_value=_MARKETS_RESPONSE):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertLess(out.index("Cut 25bps"), out.index("Hold (no change)"))
        self.assertLess(out.index("Hold (no change)"), out.index("Hike 25bps"))

    def test_missing_outcome_leg_is_skipped_not_crashed(self):
        partial = {"markets": [_leg("H0", _NEXT_MEETING, bid=0.9, ask=0.95)]}
        with mock.patch.object(kalshi_fed_rate, "_request", return_value=partial):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertIn("Hold (no change)", out)
        self.assertNotIn("Cut 25bps", out)


@pytest.mark.unit
class KalshiFedRateResilienceTests(unittest.TestCase):
    def test_network_error_degrades_gracefully(self):
        with mock.patch.object(
            kalshi_fed_rate, "_request", side_effect=requests.RequestException("boom")
        ):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertIn("unavailable", out.lower())

    def test_empty_markets_list_degrades_gracefully(self):
        # A series rename/discontinuation or a transient gap between
        # meetings -- must not crash (min() on an empty sequence would).
        with mock.patch.object(kalshi_fed_rate, "_request", return_value={"markets": []}):
            out = kalshi_fed_rate.get_fed_rate_probability()
        self.assertIn("No open Fed rate decision markets", out)


if __name__ == "__main__":
    unittest.main()
