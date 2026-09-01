"""Bridge to the top-level ``broker_reports/`` package (pure sys.path
insertion, same pattern ``backtest.py`` uses for ``screener``/``market_regime``
-- see ``broker_reports/broker_reports.py``'s docstring for the full design
and why this data deliberately lives outside the ``TradingAgents`` submodule).

Named differently from that module (``broker_report_data``, not
``broker_reports``) only to avoid a same-name-different-module ambiguity for
readers -- the sys.path insertion itself would resolve correctly either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

# dataflows/ -> tradingagents/ -> TradingAgents/ -> repo root
_BROKER_REPORTS_DIR = str(Path(__file__).resolve().parents[3] / "broker_reports")
if _BROKER_REPORTS_DIR not in sys.path:
    sys.path.insert(0, _BROKER_REPORTS_DIR)
import broker_reports  # noqa: E402


def get_broker_macro_note(curr_date: str) -> str:
    """This week's Yuanta macro note as short markdown bullets, or a
    NO_DATA_AVAILABLE sentinel if none is on file / the latest one is stale.
    Spliced directly into news_analyst's cached system-prompt block (NOT
    exposed as an LLM-callable @tool) -- see news_analyst.py's ``run_date``
    comment: the same content applies to every ticker debated in a given
    run, so it must ride the existing cache breakpoint rather than being
    fetched (and re-billed) per-ticker via a tool call.
    """
    return broker_reports.format_macro_note(curr_date)
