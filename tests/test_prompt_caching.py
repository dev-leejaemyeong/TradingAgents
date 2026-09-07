"""Tests for the prompt-caching helpers (tradingagents.agents.utils.prompt_caching).

Anthropic needs an explicit cache_control breakpoint to cache anything;
other providers cache automatically on a shared prefix and must never
receive a cache_control key (unrecognized-field risk). cached_blocks()
is the single place that decision is made, so it's tested directly here
rather than only indirectly through each agent node.
"""

from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.utils.prompt_caching import (
    cached_blocks,
    mark_last_message_cacheable,
    supports_prompt_caching,
)


def _anthropic_llm():
    """A real ChatAnthropic instance — construction doesn't touch the network,
    so a placeholder key is enough for isinstance-based provider detection."""
    return ChatAnthropic(model="claude-haiku-4-5", api_key="test-key-not-real")


@pytest.mark.unit
class TestSupportsPromptCaching:
    def test_true_for_chat_anthropic(self):
        assert supports_prompt_caching(_anthropic_llm()) is True

    def test_false_for_other_providers(self):
        assert supports_prompt_caching(MagicMock()) is False


@pytest.mark.unit
class TestCachedBlocks:
    def test_non_anthropic_concatenates_plain_string(self):
        result = cached_blocks(MagicMock(), ("stable ", True), ("variable", False))
        assert result == "stable variable"

    def test_non_anthropic_never_gets_cache_control_key(self):
        """A plain string can't carry a cache_control key at all — this is the
        actual safety property: non-Anthropic providers must never see one."""
        result = cached_blocks(MagicMock(), ("a", True), ("b", True))
        assert isinstance(result, str)

    def test_anthropic_marks_only_cacheable_segments(self):
        llm = _anthropic_llm()
        result = cached_blocks(llm, ("stable", True), ("also stable", True), ("variable", False))
        assert result == [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "also stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "variable"},
        ]

    def test_anthropic_drops_empty_segments(self):
        llm = _anthropic_llm()
        result = cached_blocks(llm, ("stable", True), ("", False), ("tail", False))
        assert result == [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "tail"},
        ]

    def test_anthropic_all_uncached_segments_still_concatenate_in_order(self):
        llm = _anthropic_llm()
        result = cached_blocks(llm, ("first", False), ("second", False))
        assert [b["text"] for b in result] == ["first", "second"]
        assert all("cache_control" not in b for b in result)


@pytest.mark.unit
class TestMarkLastMessageCacheable:
    """TODOS.md #103: caches a node's own growing tool-call-loop history
    (LangGraph analyst <-> ToolNode), independent of cached_blocks()'s static
    role/date segments."""

    def test_empty_list_is_a_no_op(self):
        assert mark_last_message_cacheable(_anthropic_llm(), []) == []

    def test_non_anthropic_returns_messages_unchanged(self):
        messages = [HumanMessage(content="hi")]
        result = mark_last_message_cacheable(MagicMock(), messages)
        assert result is messages

    def test_marks_only_the_last_message(self):
        messages = [
            HumanMessage(content="placeholder"),
            AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": "call_1"}]),
            ToolMessage(content="tool result text", tool_call_id="call_1"),
        ]
        result = mark_last_message_cacheable(_anthropic_llm(), messages)

        assert result[0] is messages[0]
        assert result[1] is messages[1]
        assert result[2] is not messages[2]
        assert result[2].content == [
            {"type": "text", "text": "tool result text", "cache_control": {"type": "ephemeral"}}
        ]
        assert result[2].tool_call_id == "call_1"

    def test_original_messages_list_untouched(self):
        messages = [ToolMessage(content="result", tool_call_id="call_1")]
        mark_last_message_cacheable(_anthropic_llm(), messages)
        assert messages[0].content == "result"

    def test_empty_last_message_content_is_left_alone(self):
        messages = [HumanMessage(content="prior"), AIMessage(content="")]
        result = mark_last_message_cacheable(_anthropic_llm(), messages)
        assert result[-1].content == ""

    def test_non_string_last_content_is_left_alone(self):
        messages = [ToolMessage(content=[{"type": "text", "text": "already blocks"}], tool_call_id="x")]
        result = mark_last_message_cacheable(_anthropic_llm(), messages)
        assert result[0] is messages[0]
