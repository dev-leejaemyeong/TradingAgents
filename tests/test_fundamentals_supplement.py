"""yfinance fundamentals carry the Alpha Vantage fields yfinance has no
equivalent for (2026-09-21).

Measured side by side first (DESIGN.md "AV vs yfinance 벤더 비교"): the
indicator values are identical to four decimals, so indicators stay on
yfinance -- but Alpha Vantage's OVERVIEW carries roughly a dozen things
yfinance's `info` simply does not have, including which quarter the figures
are from and the analyst consensus. Those are merged in here rather than
switching the vendor, which would have traded away yfinance's own unique
fields (free cash flow, current ratio, forward EPS).

The supplement must never be able to break the report it is decorating.
"""
from unittest import mock

import pytest

from tradingagents.dataflows import y_finance


@pytest.mark.unit
class TestAlphaVantageSupplement:
    def _overview(self, **overrides):
        base = {
            "LatestQuarter": "2026-05-31",
            "AnalystTargetPrice": "1513.11",
            "AnalystRatingStrongBuy": "9",
            "AnalystRatingBuy": "35",
            "AnalystRatingHold": "4",
            "AnalystRatingSell": "0",
            "AnalystRatingStrongSell": "0",
            "QuarterlyEarningsGrowthYOY": "13.69",
            "QuarterlyRevenueGrowthYOY": "3.457",
            "EVToEBITDA": "16.51",
            "EVToRevenue": "12.49",
            "SharesFloat": "1125542000",
            "PercentInsiders": "0.238",
            "PercentInstitutions": "79.899",
            # yfinance already reports these -- they must not be duplicated.
            "ReturnOnEquityTTM": "0.666",
            "MarketCapitalization": "1147237564000",
            "Description": "long boilerplate that has no place in a report",
        }
        base.update(overrides)
        return base

    def _patch_av(self, payload):
        import json

        return mock.patch(
            "tradingagents.dataflows.alpha_vantage_fundamentals.get_fundamentals",
            return_value=json.dumps(payload) if payload is not None else "not json",
        )

    def test_adds_the_fields_yfinance_does_not_have(self):
        with self._patch_av(self._overview()):
            lines = y_finance._alpha_vantage_supplement("MU")
        text = "\n".join(lines)
        assert "Latest Quarter: 2026-05-31" in text
        assert "Analyst Target Price: 1513.11" in text
        assert "Analyst Ratings (Hold): 4" in text
        assert "Quarterly Revenue Growth YoY: 3.457" in text
        assert "EV to EBITDA: 16.51" in text
        assert "Percent Institutions: 79.899" in text

    def test_does_not_repeat_what_yfinance_already_reports(self):
        # The overlapping numbers agree between vendors (ROE 0.666 vs 0.66638),
        # so a second spelling of one number is pure token cost and invites the
        # model to read two facts where there is one.
        with self._patch_av(self._overview()):
            text = "\n".join(y_finance._alpha_vantage_supplement("MU"))
        assert "0.666" not in text
        assert "1147237564000" not in text
        assert "boilerplate" not in text

    def test_drops_alpha_vantages_placeholder_values(self):
        # OVERVIEW writes "-" or "None" where it has nothing; those must not
        # reach the report looking like data.
        payload = self._overview(AnalystTargetPrice="-", EVToEBITDA="None", SharesFloat="")
        with self._patch_av(payload):
            text = "\n".join(y_finance._alpha_vantage_supplement("MU"))
        assert "Analyst Target Price" not in text
        assert "EV to EBITDA" not in text
        assert "Shares Float" not in text
        assert "Latest Quarter: 2026-05-31" in text  # the rest still comes through

    def test_a_broken_alpha_vantage_is_silent(self):
        with mock.patch(
            "tradingagents.dataflows.alpha_vantage_fundamentals.get_fundamentals",
            side_effect=RuntimeError("rate limited"),
        ):
            assert y_finance._alpha_vantage_supplement("MU") == []

    def test_non_json_from_alpha_vantage_is_silent(self):
        with self._patch_av(None):
            assert y_finance._alpha_vantage_supplement("MU") == []

    def test_report_survives_a_dead_alpha_vantage(self):
        # The whole point of the silent failure: losing a secondary vendor must
        # cost the supplement, not the fundamentals report.
        info = {"longName": "Micron", "trailingPE": 22.9, "freeCashflow": 7639499776}
        with mock.patch.object(y_finance.yf, "Ticker") as ticker, mock.patch.object(
            y_finance, "yf_retry", lambda fn: fn()
        ), mock.patch(
            "tradingagents.dataflows.alpha_vantage_fundamentals.get_fundamentals",
            side_effect=RuntimeError("down"),
        ):
            ticker.return_value.info = info
            out = y_finance.get_fundamentals("MU", "2026-09-18")
        assert "PE Ratio (TTM): 22.9" in out
        assert "Free Cash Flow: 7639499776" in out
        assert "Analyst Target Price" not in out
