"""get_prediction_markets tool wrapper: recession/Fed-rate keyword detection
and verified-source supplementing (TODOS #114).

All vendor/FRED/Kalshi calls are mocked, so these run without a network
connection.
"""
import unittest
from unittest import mock

import pytest

from tradingagents.agents.utils import prediction_markets_tools as pmt


@pytest.mark.unit
class SupplementDetectionTests(unittest.TestCase):
    def test_recession_topic_prepends_fred_supplement(self):
        with (
            mock.patch.object(pmt, "route_to_vendor", return_value="MANIFOLD_TEXT"),
            mock.patch.object(pmt, "get_macro_data", return_value="FRED_RECESSION_TEXT") as fred_call,
        ):
            out = pmt.get_prediction_markets.func("recession 2026")
        self.assertIn("FRED_RECESSION_TEXT", out)
        self.assertIn("MANIFOLD_TEXT", out)
        self.assertLess(out.index("FRED_RECESSION_TEXT"), out.index("MANIFOLD_TEXT"))
        fred_call.assert_called_once()
        self.assertEqual(fred_call.call_args.args[0], "recession_probability")

    def test_fed_rate_topic_prepends_kalshi_supplement(self):
        with (
            mock.patch.object(pmt, "route_to_vendor", return_value="MANIFOLD_TEXT"),
            mock.patch.object(pmt, "get_fed_rate_probability", return_value="KALSHI_FED_TEXT") as kalshi_call,
        ):
            out = pmt.get_prediction_markets.func("Fed rate cut 2026")
        self.assertIn("KALSHI_FED_TEXT", out)
        self.assertIn("MANIFOLD_TEXT", out)
        self.assertLess(out.index("KALSHI_FED_TEXT"), out.index("MANIFOLD_TEXT"))
        kalshi_call.assert_called_once()

    def test_unrelated_topic_has_no_supplement(self):
        with (
            mock.patch.object(pmt, "route_to_vendor", return_value="MANIFOLD_TEXT"),
            mock.patch.object(pmt, "get_macro_data") as fred_call,
            mock.patch.object(pmt, "get_fed_rate_probability") as kalshi_call,
        ):
            out = pmt.get_prediction_markets.func("Taiwan China conflict")
        self.assertEqual(out, "MANIFOLD_TEXT")
        fred_call.assert_not_called()
        kalshi_call.assert_not_called()

    def test_recession_keyword_matches_case_insensitively_and_as_whole_word(self):
        # Must match "Recession" (different case) but not fire on unrelated
        # substrings that happen to embed similar letters.
        with (
            mock.patch.object(pmt, "route_to_vendor", return_value="V"),
            mock.patch.object(pmt, "get_macro_data", return_value="F") as fred_call,
        ):
            pmt.get_prediction_markets.func("US Recession Risk 2026")
        fred_call.assert_called_once()

    def test_fed_rate_regex_matches_common_real_world_phrasings(self):
        for topic in ["Fed rate cut 2026", "Fed rate cut", "FOMC rate hike", "rate cut"]:
            with self.subTest(topic=topic):
                self.assertIsNotNone(pmt._FED_RATE_RE.search(topic))

    def test_recession_supplement_failure_falls_back_to_vendor_only(self):
        # A supplement is best-effort -- if FRED itself errors, the main
        # tool call must still return the vendor's own result, not raise.
        with (
            mock.patch.object(pmt, "route_to_vendor", return_value="MANIFOLD_TEXT"),
            mock.patch.object(pmt, "get_macro_data", side_effect=RuntimeError("boom")),
        ):
            out = pmt.get_prediction_markets.func("recession 2026")
        self.assertEqual(out, "MANIFOLD_TEXT")


if __name__ == "__main__":
    unittest.main()
