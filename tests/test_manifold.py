"""Manifold Markets prediction-market vendor: forward-looking filtering,
volume ranking, formatting, graceful degradation, and router integration.

All API access is mocked, so these run without a network connection.
"""
import copy
import unittest
from datetime import datetime, timezone
from unittest import mock

import pytest
import requests

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface, manifold
from tradingagents.dataflows.config import set_config


def _ms(iso_str):
    return int(datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _market(question, prob, *, volume, close_iso, resolved=False, bettors=100):
    return {
        "question": question,
        "probability": prob,
        "volume": volume,
        "closeTime": _ms(close_iso),
        "isResolved": resolved,
        "outcomeType": "BINARY",
        "uniqueBettorCount": bettors,
    }


# A mix: a high-volume open market, a resolved one, a past-closeTime one, and
# a lower-volume open one. Far-future/far-past dates keep the test independent
# of the real clock.
_SEARCH = [
    _market("Open big?", 0.76, volume=5_000_000, close_iso="2030-12-31"),
    _market("Resolved already?", 1.0, volume=9_000_000, close_iso="2030-12-31", resolved=True),
    _market("Past event?", 0.5, volume=8_000_000, close_iso="2020-01-01"),
    _market("Open small?", 0.30, volume=1_000, close_iso="2030-06-30"),
]


@pytest.mark.unit
class ManifoldFilterTests(unittest.TestCase):
    def test_resolved_and_past_markets_are_excluded(self):
        with mock.patch.object(manifold, "_request", return_value=_SEARCH):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertIn("Open big?", out)
        self.assertIn("Open small?", out)
        self.assertNotIn("Resolved already?", out)  # isResolved
        self.assertNotIn("Past event?", out)          # closeTime in the past

    def test_ranked_by_volume(self):
        with mock.patch.object(manifold, "_request", return_value=_SEARCH):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertLess(out.index("Open big?"), out.index("Open small?"))

    def test_limit_caps_results(self):
        with mock.patch.object(manifold, "_request", return_value=_SEARCH):
            out = manifold.get_prediction_markets("anything", limit=1)
        self.assertIn("Open big?", out)
        self.assertNotIn("Open small?", out)


@pytest.mark.unit
class ManifoldFormatTests(unittest.TestCase):
    def test_probability_and_volume_render(self):
        with mock.patch.object(manifold, "_request", return_value=_SEARCH):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertIn("Yes 76%", out)
        self.assertIn("M$5,000,000 volume", out)
        self.assertIn("resolves 2030-12-31", out)

    def test_volume_labelled_as_mana_not_dollars(self):
        with mock.patch.object(manifold, "_request", return_value=_SEARCH):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertIn("play-money currency", out)
        # Bare "(${amount}" (Polymarket-style, real USD) must not appear --
        # only "(M${amount}" (Manifold's play-money mana) should.
        self.assertNotIn("($5,000,000 volume", out)

    def test_low_liquidity_market_flagged(self):
        thin = [_market("Thin market?", 0.5, volume=100, close_iso="2030-06-30", bettors=3)]
        with mock.patch.object(manifold, "_request", return_value=thin):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertIn("low liquidity", out)

    def test_deep_liquidity_market_not_flagged(self):
        deep = [_market("Deep market?", 0.5, volume=100_000, close_iso="2030-06-30", bettors=500)]
        with mock.patch.object(manifold, "_request", return_value=deep):
            out = manifold.get_prediction_markets("anything", limit=10)
        self.assertNotIn("low liquidity", out)

    def test_no_matches_reports_clearly(self):
        with mock.patch.object(manifold, "_request", return_value=[]):
            out = manifold.get_prediction_markets("obscure ticker", limit=6)
        self.assertIn("No open prediction markets", out)


@pytest.mark.unit
class ManifoldResilienceTests(unittest.TestCase):
    def test_network_error_degrades_gracefully(self):
        # An external-service hiccup must not raise into the analyst.
        with mock.patch.object(
            manifold, "_request", side_effect=requests.RequestException("boom")
        ):
            out = manifold.get_prediction_markets("Fed rate cut")
        self.assertIn("unavailable", out.lower())
        self.assertIn("Fed rate cut", out)


@pytest.mark.unit
class ManifoldRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_default_config_routes_to_manifold(self):
        self.assertEqual(
            default_config.DEFAULT_CONFIG["data_vendors"]["prediction_markets"],
            "manifold",
        )

    def test_category_routes_to_manifold(self):
        self.assertEqual(
            interface.get_category_for_method("get_prediction_markets"),
            "prediction_markets",
        )
        set_config({"data_vendors": {"prediction_markets": "manifold"}})
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_prediction_markets": {"manifold": lambda *a, **k: "MANIFOLD_OK"}},
            clear=False,
        ):
            out = interface.route_to_vendor("get_prediction_markets", "fed", 5)
        self.assertEqual(out, "MANIFOLD_OK")


if __name__ == "__main__":
    unittest.main()
