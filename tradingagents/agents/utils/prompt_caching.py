"""Prompt-caching helpers.

Anthropic requires an explicit ``cache_control`` breakpoint on a content
block before it will cache anything — unlike OpenAI, which caches
automatically once a request shares an identical prefix (>=1024 tokens) with
a recent one. Either way, two things have to be true for a cache hit:

1. Stable content (identical across many calls) must come BEFORE variable
   content (ticker, date, growing debate history) in the prompt. Both
   caching mechanisms match on a shared *prefix* from the start of the
   message — content is not reordered or fuzzy-matched.
2. On Anthropic, each stable segment that should be cached independently
   needs its own ``cache_control`` breakpoint (max 4 per request).

New prompts in this codebase should build their content with
``cached_blocks()`` below rather than a single interpolated string, ordering
segments from most-stable (reused across an entire day's batch — e.g. static
role instructions) to least-stable (reused only within one ticker's run —
e.g. that ticker's analyst reports, re-sent on every debate turn) to fully
variable (grows every call — e.g. debate history, the latest counter-
argument). Mark a segment cacheable only if it will actually be resent
unchanged at least once more; a breakpoint on content that never repeats is
pure overhead (and consumes one of Anthropic's 4-per-request slots) for
nothing.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

_EPHEMERAL = {"type": "ephemeral"}


def supports_prompt_caching(llm) -> bool:
    """Whether ``llm`` needs (and accepts) explicit ``cache_control`` breakpoints.

    True only for the direct Anthropic integration. OpenAI-family and other
    providers cache automatically on a shared prefix with no special
    content-block markup — sending them a ``cache_control`` key would be at
    best a no-op and at worst an unrecognized-field error, so callers must
    gate on this before adding the key rather than setting it unconditionally.
    """
    return isinstance(llm, ChatAnthropic)


def cached_blocks(llm, *segments: tuple[str, bool]) -> str | list[dict]:
    """Build message ``content`` from ``(text, should_cache)`` segments,
    ordered most-stable first.

    On Anthropic, each segment marked ``should_cache=True`` gets its own
    ``cache_control`` breakpoint; every segment is still concatenated in
    order regardless of provider. On non-Anthropic providers this just joins
    the text — segment order (stable-first) is what lets that provider's
    automatic caching find a match; no markup is needed or applied.

    Empty segments are dropped so an unused optional section (e.g. no prior
    debate history yet) doesn't leave a stray empty content block.
    """
    if not supports_prompt_caching(llm):
        return "".join(text for text, _should_cache in segments)

    blocks: list[dict] = []
    for text, should_cache in segments:
        if not text:
            continue
        block: dict = {"type": "text", "text": text}
        if should_cache:
            block["cache_control"] = dict(_EPHEMERAL)
        blocks.append(block)
    return blocks
