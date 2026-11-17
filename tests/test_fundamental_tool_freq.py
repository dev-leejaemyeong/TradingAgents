"""freq on the statement tools is limited to annual/quarterly (upstream bf7f4b5)."""
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tradingagents.agents.utils import fundamental_data_tools as fdt

TOOLS = [fdt.get_balance_sheet, fdt.get_cashflow, fdt.get_income_statement]


@pytest.mark.unit
@pytest.mark.parametrize("tool", TOOLS)
def test_schema_exposes_only_annual_and_quarterly(tool):
    enum = tool.args["freq"]["enum"]
    assert sorted(enum) == ["annual", "quarterly"]


@pytest.mark.unit
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("freq", ["annual", "quarterly"])
def test_valid_freq_reaches_vendor(tool, freq):
    with patch.object(fdt, "route_to_vendor", return_value="ok") as rv:
        assert tool.invoke({"ticker": "AAPL", "freq": freq, "curr_date": "2026-11-17"}) == "ok"
    assert rv.call_args.args[2] == freq


@pytest.mark.unit
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("freq", ["TTM", "semiannual", ""])
def test_other_freq_is_rejected_before_vendor(tool, freq):
    with patch.object(fdt, "route_to_vendor", return_value="ok") as rv:
        with pytest.raises(ValidationError):
            tool.invoke({"ticker": "AAPL", "freq": freq, "curr_date": "2026-11-17"})
    rv.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("tool", TOOLS)
def test_default_is_quarterly(tool):
    with patch.object(fdt, "route_to_vendor", return_value="ok") as rv:
        tool.invoke({"ticker": "AAPL"})
    assert rv.call_args.args[2] == "quarterly"
