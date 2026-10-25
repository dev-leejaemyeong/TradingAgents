"""observation_only / past_context_override on propagate() (TODOS #173/A0-2a).

A same-input immediate re-run (orchestrator's Sell-rerun consistency check)
must not touch the memory log or overwrite the original run's state-log file,
and must see the exact past_context the original run used. Builds a bare
TradingAgentsGraph instance (same pattern as test_checkpoint_resume.py's
TestCheckpointSignature) with the LLM-backed collaborators stubbed out, so
these tests exercise only trading_graph.py's own control flow.
"""

import tempfile
import unittest

from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _final_state(decision="Sell"):
    return {
        "final_trade_decision": decision,
        "company_of_interest": "TEST",
        "trade_date": "2026-09-24",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "",
        },
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "", "neutral_history": "",
            "history": "", "judge_decision": "",
        },
        "investment_plan": "",
    }


class _FakeMemoryLog:
    def __init__(self):
        self.get_past_context_calls = []
        self.get_pending_entries_calls = 0
        self.store_decision_calls = []

    def get_past_context(self, ticker, n_same=5, n_cross=3):  # TODOS #168/M2: the call site passes counts
        self.get_past_context_calls.append(ticker)
        return "MEMORY_CONTEXT"

    def get_pending_entries(self):
        self.get_pending_entries_calls += 1
        return []

    def store_decision(self, **kwargs):
        self.store_decision_calls.append(kwargs)


class _FakeGraph:
    """Stands in for self.graph (the compiled LangGraph)."""

    def __init__(self, final_state):
        self._final_state = final_state
        self.invoke_calls = 0
        self.seen_past_context = None

    def invoke(self, init_state, **kwargs):
        self.invoke_calls += 1
        self.seen_past_context = init_state.get("past_context")
        return dict(self._final_state)


class _FakeSignalProcessor:
    def process_signal(self, full_signal):
        return full_signal  # already "Sell"/"Hold"/... in these tests


def _bare_graph(final_state, tmpdir) -> tuple[TradingAgentsGraph, _FakeMemoryLog, _FakeGraph]:
    g = object.__new__(TradingAgentsGraph)
    g.debug = False
    g.config = {"checkpoint_enabled": False, "results_dir": tmpdir}
    g.callbacks = []
    g.propagator = Propagator()
    g.signal_processor = _FakeSignalProcessor()
    g.log_states_dict = {}
    g.curr_state = None
    g.ticker = None
    g._checkpointer_ctx = None
    memory_log = _FakeMemoryLog()
    g.memory_log = memory_log
    fake_graph = _FakeGraph(final_state)
    g.graph = fake_graph
    # resolve_instrument_context normally does a real yfinance lookup --
    # stub it so these tests don't hit the network.
    g.resolve_instrument_context = lambda ticker, asset_type="stock": ""
    return g, memory_log, fake_graph


class TestObservationOnly(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_default_run_touches_memory_log_and_state_log(self):
        """Regression lock: unchanged behavior when neither new kwarg is used."""
        g, memory_log, fake_graph = _bare_graph(_final_state(), self.tmpdir)

        g.propagate("TEST", "2026-09-24", extra_context="EXTRA")

        self.assertEqual(memory_log.get_pending_entries_calls, 1)
        self.assertEqual(memory_log.get_past_context_calls, ["TEST"])
        self.assertEqual(len(memory_log.store_decision_calls), 1)
        self.assertEqual(fake_graph.seen_past_context, "EXTRA\n\nMEMORY_CONTEXT")
        self.assertIn("2026-09-24", g.log_states_dict)

    def test_observation_only_skips_all_three_side_effects(self):
        g, memory_log, fake_graph = _bare_graph(_final_state(), self.tmpdir)

        final_state, signal = g.propagate(
            "TEST", "2026-09-24", extra_context="EXTRA", observation_only=True,
        )

        self.assertEqual(signal, "Sell")
        self.assertEqual(memory_log.get_pending_entries_calls, 0)  # _resolve_pending_entries
        self.assertEqual(memory_log.store_decision_calls, [])
        self.assertEqual(g.log_states_dict, {})  # _log_state never wrote

    def test_past_context_override_bypasses_get_past_context(self):
        g, memory_log, fake_graph = _bare_graph(_final_state(), self.tmpdir)

        g.propagate(
            "TEST", "2026-09-24",
            observation_only=True,
            past_context_override="ORIGINAL PAST CONTEXT",
        )

        self.assertEqual(memory_log.get_past_context_calls, [])
        self.assertEqual(fake_graph.seen_past_context, "ORIGINAL PAST CONTEXT")

    def test_past_context_override_used_verbatim_even_with_extra_context(self):
        """extra_context is already folded into the caller-supplied override --
        it must not be concatenated a second time."""
        g, memory_log, fake_graph = _bare_graph(_final_state(), self.tmpdir)

        g.propagate(
            "TEST", "2026-09-24",
            extra_context="SHOULD NOT APPEAR TWICE",
            observation_only=True,
            past_context_override="ORIGINAL PAST CONTEXT",
        )

        self.assertEqual(fake_graph.seen_past_context, "ORIGINAL PAST CONTEXT")


if __name__ == "__main__":
    unittest.main()
