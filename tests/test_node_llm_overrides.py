"""Per-node LLM overrides: a single graph node can run on a different
vendor/model than the shared quick/deep tiers (needed to mix vendors per
role, e.g. Bull and Bear on different vendors to reduce correlated bias).

Absent ``node_llm_overrides`` entries, behavior is unchanged: every node
falls back to the shared quick/deep client exactly as before this existed.
"""

from __future__ import annotations

import pytest

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _bare_graph(config, callbacks=None):
    g = object.__new__(TradingAgentsGraph)
    g.config = config
    g.callbacks = callbacks or []
    return g


@pytest.mark.unit
class TestGetProviderKwargsWithOverrides:
    """_get_provider_kwargs(provider, overrides) resolves per-node, falling
    back to global config for keys the override doesn't specify."""

    def test_no_args_matches_original_behavior(self):
        graph = _bare_graph({"llm_provider": "anthropic", "anthropic_effort": "high"})
        assert graph._get_provider_kwargs() == {"effort": "high"}

    def test_override_provider_switches_which_knob_applies(self):
        graph = _bare_graph({"llm_provider": "anthropic", "anthropic_effort": "high"})
        kwargs = graph._get_provider_kwargs("openai", {"openai_reasoning_effort": "medium"})
        assert kwargs == {"reasoning_effort": "medium"}

    def test_override_missing_key_falls_back_to_global_config(self):
        graph = _bare_graph({"llm_provider": "openai", "temperature": 0.2})
        kwargs = graph._get_provider_kwargs("openai", {"openai_reasoning_effort": "low"})
        assert kwargs == {"reasoning_effort": "low", "temperature": 0.2}

    def test_override_key_takes_precedence_over_global(self):
        graph = _bare_graph({"llm_provider": "openai", "temperature": 0.2})
        kwargs = graph._get_provider_kwargs("openai", {"temperature": 0.9})
        assert kwargs["temperature"] == 0.9


@pytest.mark.unit
class TestBuildNodeLlms:
    def test_empty_overrides_returns_empty_dict(self):
        graph = _bare_graph({"llm_provider": "openai", "node_llm_overrides": {}})
        assert graph._build_node_llms() == {}

    def test_missing_key_returns_empty_dict(self):
        graph = _bare_graph({"llm_provider": "openai"})
        assert graph._build_node_llms() == {}

    def test_override_produces_entry_only_for_that_node(self):
        graph = _bare_graph({
            "llm_provider": "anthropic",
            "backend_url": None,
            "node_llm_overrides": {
                "bear_researcher": {"llm_provider": "openai", "model": "gpt-5.4"},
            },
        })
        node_llms = graph._build_node_llms()
        assert set(node_llms) == {"bear_researcher"}
        assert node_llms["bear_researcher"] is not None

    def test_identical_overrides_share_one_client(self, monkeypatch):
        calls = []
        import tradingagents.graph.trading_graph as trading_graph_module

        def fake_create_llm_client(**kwargs):
            calls.append(kwargs)
            class _FakeClient:
                def get_llm(self_inner):
                    return object()
            return _FakeClient()

        monkeypatch.setattr(trading_graph_module, "create_llm_client", fake_create_llm_client)

        graph = _bare_graph({
            "llm_provider": "anthropic",
            "backend_url": None,
            "node_llm_overrides": {
                "aggressive_analyst": {"llm_provider": "openai", "model": "o3"},
                "portfolio_manager": {"llm_provider": "openai", "model": "o3"},
                "bear_researcher": {"llm_provider": "openai", "model": "gpt-5.4"},
            },
        })
        node_llms = graph._build_node_llms()

        # Two distinct (provider, model) pairs among three overrides -> two
        # underlying client constructions, but three node entries.
        assert len(calls) == 2
        assert len(node_llms) == 3
        assert node_llms["aggressive_analyst"] is node_llms["portfolio_manager"]
        assert node_llms["bear_researcher"] is not node_llms["aggressive_analyst"]

    def test_missing_model_key_fails_loudly(self):
        graph = _bare_graph({
            "llm_provider": "anthropic",
            "backend_url": None,
            "node_llm_overrides": {
                "bull_researcher": {"llm_provider": "openai"},  # no "model"
            },
        })
        with pytest.raises(KeyError):
            graph._build_node_llms()


@pytest.mark.unit
class TestGraphSetupLlmFor:
    """GraphSetup._llm_for falls back to the shared quick/deep client for any
    node absent from node_llms, and returns the override otherwise."""

    def _setup(self, node_llms=None):
        return GraphSetup(
            quick_thinking_llm="QUICK",
            deep_thinking_llm="DEEP",
            tool_nodes={},
            conditional_logic=None,
            node_llms=node_llms,
        )

    def test_no_node_llms_falls_back_to_default(self):
        setup = self._setup()
        assert setup._llm_for("bull_researcher", "QUICK") == "QUICK"
        assert setup._llm_for("portfolio_manager", "DEEP") == "DEEP"

    def test_override_present_is_used_instead_of_default(self):
        setup = self._setup(node_llms={"bear_researcher": "GPT_5_4"})
        assert setup._llm_for("bear_researcher", "QUICK") == "GPT_5_4"
        # Unrelated node still falls back.
        assert setup._llm_for("bull_researcher", "QUICK") == "QUICK"
