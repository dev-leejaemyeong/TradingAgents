"""Okapi BM25 over a handful of documents (TODOS #168/M2, 2026-10-23).

Restores the similarity ranking the upstream per-agent memory used (the
``rank_bm25`` package) without adding a dependency: the corpus here is at most
a few hundred resolved decisions, so a plain implementation is enough, and a
package missing from the scheduled task's Python would fail silently where
this cannot.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def top_k(query: str, documents: list[str], k: int, *, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """``[(document index, BM25 score)]`` for the ``k`` best-matching documents,
    best first. Documents scoring 0 (no query term in common) are never
    returned, so an empty or unrelated corpus yields ``[]`` rather than an
    arbitrary pick. IDF is the non-negative ``log(1 + (N - n + .5)/(n + .5))``
    form, so a term present in every document still counts a little."""
    query_terms = set(tokenize(query))
    if not query_terms or not documents or k <= 0:
        return []
    tokenized = [tokenize(d) for d in documents]
    n_docs = len(tokenized)
    avg_len = sum(len(t) for t in tokenized) / n_docs or 1.0
    doc_freq = Counter(term for toks in tokenized for term in set(toks) if term in query_terms)
    scored = []
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        freqs = Counter(toks)
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            n = doc_freq[term]
            idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]
