"""Tests for the RSS-first Reddit fetcher (one combined r/a+b+c request, upstream
241638d), its single 60s 429 retry (#1295), the opt-in JSON path's degradation
(#862), and chunked-transfer error handling (#1024)."""

from __future__ import annotations

import http.client
import io
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>NVDA earnings beat, stock pops</title>
    <published>2026-05-20T14:30:00+00:00</published>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Great &lt;b&gt;quarter&lt;/b&gt; for NVDA&amp;#39;s datacenter unit.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>Is NVDA overvalued?</title>
    <published>2026-05-19T09:00:00Z</published>
    <content type="html">&lt;p&gt;Forward P/E discussion&lt;/p&gt;</content>
  </entry>
</feed>
"""


_OPEN_ERRORS: list[HTTPError] = []


def _http_error(code, msg, headers):
    """An HTTPError that the autouse fixture below closes after the test: urllib
    wraps even a BytesIO fp in a tempfile wrapper whose GC-time cleanup raises a
    ResourceWarning (an error under ``-W error``)."""
    err = HTTPError("url", code, msg, headers, io.BytesIO())
    _OPEN_ERRORS.append(err)
    return err


@pytest.fixture(autouse=True)
def _rss_enabled(monkeypatch):
    """The module ships disabled (Reddit ended RSS on 2026-11-13); these tests
    exercise the fetcher for the day it is switched back on."""
    monkeypatch.setattr(reddit, "_RSS_DISCONTINUED", False)


@pytest.fixture(autouse=True)
def _close_http_errors():
    yield
    while _OPEN_ERRORS:
        _OPEN_ERRORS.pop().close()


def _combined_atom(entries):
    """An Atom feed whose entries are (title, subreddit-or-None) pairs."""
    body = ""
    for i, (title, sub) in enumerate(entries):
        cat = f'<category term="{sub}" label="r/{sub}"/>' if sub else ""
        body += (
            f"<entry><title>{title}</title>{cat}"
            f"<published>2026-05-{20 - i % 9:02d}T09:00:00Z</published>"
            f'<content type="html">&lt;p&gt;body {i}&lt;/p&gt;</content></entry>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'
    ).encode("utf-8")


def _resp(read_fn):
    """A minimal context-manager response whose read() runs ``read_fn``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner, amt=None):
            return read_fn()
    return _Resp()


def _atom_resp():
    return _resp(lambda: _SAMPLE_ATOM.encode("utf-8"))


def _raise(exc):
    def _r():
        raise exc
    return _resp(_r)


@pytest.mark.unit
class TestIsoToTimestamp:
    def test_parses_offset_and_z(self):
        assert reddit._iso_to_timestamp("2026-05-20T14:30:00+00:00") > 0
        assert reddit._iso_to_timestamp("2026-05-19T09:00:00Z") > 0

    def test_none_and_garbage_return_none(self):
        assert reddit._iso_to_timestamp(None) is None
        assert reddit._iso_to_timestamp("not-a-date") is None


@pytest.mark.unit
class TestStripHtml:
    def test_extracts_between_sc_markers_and_unescapes(self):
        raw = "<!-- SC_OFF --><div class=\"md\"><p>Great <b>quarter</b> &amp; more</p></div><!-- SC_ON -->"
        assert reddit._strip_html(raw) == "Great quarter & more"

    def test_empty(self):
        assert reddit._strip_html("") == ""


@pytest.mark.unit
class TestRssParsing:
    def test_parses_atom_entries(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", limit=5, timeout=5.0)
        assert len(posts) == 2
        assert posts[0]["title"] == "NVDA earnings beat, stock pops"
        assert posts[0]["source"] == "rss"
        assert posts[0]["score"] is None
        assert posts[0]["num_comments"] is None
        assert posts[0]["created_utc"] > 0
        assert "datacenter unit" in posts[0]["selftext"]

    def test_entry_category_term_becomes_the_subreddit(self):
        feed = _combined_atom([("a", "stocks"), ("b", "wallstreetbets")])
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: feed)):
            posts = reddit._fetch_subreddit_rss("NVDA", "wallstreetbets+stocks", 100, 5.0)
        assert [p["subreddit"] for p in posts] == ["stocks", "wallstreetbets"]

    def test_untagged_entry_of_a_single_subreddit_request_belongs_to_it(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert {p["subreddit"] for p in posts} == {"stocks"}

    def test_untagged_entry_of_a_combined_request_stays_unattributed(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks+investing", 5, 5.0)
        assert {p["subreddit"] for p in posts} == {""}

    def test_malformed_xml_returns_none_not_empty_list(self):
        # None (not []) marks this a failed fetch, distinct from a request
        # that succeeded and genuinely found zero posts.
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: b"<<not xml>>")):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None


@pytest.mark.unit
class TestFetchSubredditIsRssFirst:
    """The default per-subreddit fetch goes straight to RSS — it must not hit
    the WAF-blocked JSON endpoint, which only burned rate-limit budget."""

    def test_delegates_to_rss_without_touching_json(self):
        sentinel = [{"title": "x", "source": "rss", "score": None,
                     "num_comments": None, "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=sentinel) as rss, \
             patch.object(reddit, "urlopen",
                          side_effect=AssertionError("JSON endpoint must not be called")):
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out is sentinel


@pytest.mark.unit
class TestJsonPathFallsBackToRss:
    """The opt-in JSON path still degrades to RSS on a 403 (kept for #862)."""

    def test_403_triggers_rss(self):
        err = _http_error(403, "Blocked", {})
        rss_posts = [{"title": "x", "source": "rss", "score": None,
                      "num_comments": None, "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "urlopen", side_effect=err), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            out = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"


@pytest.mark.unit
class TestRss429Backoff:
    def test_429_then_success_retries_once(self):
        err = _http_error(429, "Too Many Requests", {})
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # original + one retry
        slept.assert_called_once()         # backed off before retrying
        assert len(posts) == 2

    def test_429_twice_gives_up_after_a_single_retry(self):
        # 2026-11-17 (upstream #1295): at ~1 request/min a retry before ~60s still
        # 429s, so a second quick retry is a wasted request that also spends the
        # limit. One retry after a long wait; then give up.
        err = _http_error(429, "Too Many Requests", {})
        with patch.object(reddit, "urlopen", side_effect=[err, err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # original + ONE retry
        assert slept.call_count == 1
        assert posts is None               # not [] -- this was a failure, not confirmed silence

    def test_retry_after_header_is_honoured(self):
        err = _http_error(429, "Too Many Requests", {"Retry-After": "12"})
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(12.0)

    def test_retry_after_zero_is_honoured_not_treated_as_absent(self):
        # A valid "Retry-After: 0" means retry at once; the old `val else None`
        # treated the string "0" as falsy and waited the 5s headerless fallback.
        err = _http_error(429, "Too Many Requests", {"Retry-After": "0"})
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(0.0)

    def test_headerless_429_backoff_is_jittered_around_sixty_seconds(self):
        err = _http_error(429, "Too Many Requests", {})
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        (wait,), _ = slept.call_args
        assert 48.0 <= wait <= 72.0

    # A malformed numeric Retry-After ("-1", "nan") used to be returned as the
    # wait; the real time.sleep() raises ValueError for both, and that raise
    # happens inside `except HTTPError` so it escaped the fetch. (upstream #1601)
    @staticmethod
    def _strict_sleep(seconds):
        if seconds != seconds or seconds < 0:   # same contract as time.sleep
            raise ValueError("sleep length must be a non-negative number")

    @pytest.mark.parametrize("raw", ["-1", "-0.5", "nan", "NaN"])
    def test_invalid_retry_after_uses_the_jittered_fallback(self, raw):
        err = _http_error(429, "Too Many Requests", {"Retry-After": raw})
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep", side_effect=self._strict_sleep) as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # retried, not crashed
        assert len(posts) == 2
        (wait,), _ = slept.call_args
        assert 48.0 <= wait <= 72.0        # headerless fallback

    @pytest.mark.parametrize("raw, expected", [
        ("0", 0.0), ("12", 12.0), ("100", 60.0), ("inf", 60.0),   # valid / capped at 60s
        ("-1", None), ("-0.5", None), ("nan", None), ("abc", None), ("", None),
    ])
    def test_retry_after_seconds_parsing(self, raw, expected):
        err = _http_error(429, "Too Many Requests", {"Retry-After": raw})
        try:
            assert reddit._retry_after_seconds(err) == expected
        finally:
            err.close()   # fp=None makes HTTPError open a temp file; avoid ResourceWarning


@pytest.mark.unit
class TestChunkedTransferErrorsHandled:
    """IncompleteRead/RemoteDisconnected come from http.client and are NOT
    OSErrors, so they were previously uncaught and crashed the pipeline (#1024)."""

    def test_rss_incomplete_read_returns_none(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None

    def test_json_incomplete_read_falls_back_to_rss(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()


@pytest.mark.unit
class TestFeedReadIsBounded:
    """An oversized response must not be streamed fully into memory before
    parsing -- it degrades through the same failure path as a malformed feed."""

    def test_oversized_rss_body_returns_none_not_a_crash(self):
        oversized = _resp(lambda: b"x" * (reddit._MAX_FEED_BYTES + 1))
        with patch.object(reddit, "urlopen", return_value=oversized):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None

    def test_oversized_json_body_falls_back_to_rss(self):
        oversized = _resp(lambda: b"x" * (reddit._MAX_FEED_BYTES + 1))
        with patch.object(reddit, "urlopen", return_value=oversized), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()

    def test_feed_within_the_cap_is_read_normally(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert len(posts) == 2


@pytest.mark.unit
class TestFormatterHandlesRssPosts:
    def test_rss_posts_omit_fake_counts_and_note_source(self):
        rss_posts = [{
            "title": "NVDA pops", "score": None, "num_comments": None,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "great quarter", "source": "rss", "subreddit": "stocks",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=rss_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "via RSS feed" in out
        assert "↑" not in out  # no fake score arrow
        assert "NVDA pops" in out
        assert "great quarter" in out

    def test_json_posts_still_show_counts(self):
        json_posts = [{
            "title": "NVDA pops", "score": 1234, "num_comments": 56,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "", "subreddit": "stocks",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=json_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "1234↑" in out
        assert "56c" in out
        assert "via RSS" not in out


@pytest.mark.unit
class TestCombinedFeed:
    """All subreddits go out as ONE r/a+b+c request (upstream 241638d): anonymous
    RSS allows ~1 request/min per IP, so one request per subreddit spent the
    budget three times per ticker."""

    _SUBS = ("wallstreetbets", "stocks", "investing")

    def _run(self, feed_bytes, **kw):
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: feed_bytes)) as op, \
             patch.object(reddit.time, "sleep", side_effect=AssertionError("no sleep expected")):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=self._SUBS, **kw)
        return out, op

    def test_one_request_covers_every_subreddit(self):
        feed = _combined_atom([("w1", "wallstreetbets"), ("s1", "stocks"), ("i1", "investing")])
        out, op = self._run(feed)
        assert op.call_count == 1
        url = op.call_args[0][0].full_url
        assert "/r/wallstreetbets+stocks+investing/search.rss" in url
        assert "limit=100" in url
        for t in ("w1", "s1", "i1"):
            assert t in out

    def test_posts_are_grouped_back_per_subreddit_in_requested_order(self):
        feed = _combined_atom([("i1", "investing"), ("w1", "wallstreetbets"), ("s1", "stocks")])
        out, _ = self._run(feed)
        assert out.index("r/wallstreetbets") < out.index("r/stocks") < out.index("r/investing")
        # each title sits under its own subreddit's block
        blocks = {b.splitlines()[0].split()[0]: b for b in out.split("\n\n")}
        assert "w1" in blocks["r/wallstreetbets"] and "w1" not in blocks["r/stocks"]
        assert "i1" in blocks["r/investing"]

    def test_limit_per_sub_applies_per_subreddit_newest_first(self):
        feed = _combined_atom([(f"w{i}", "wallstreetbets") for i in range(8)]
                              + [("s0", "stocks")])
        out, _ = self._run(feed, limit_per_sub=3)
        assert "w0" in out and "w2" in out and "w3" not in out   # newest 3 only
        assert "s0" in out                                        # not crowded out

    def test_subreddit_without_posts_is_reported_as_none_found(self):
        feed = _combined_atom([("w1", "wallstreetbets")])
        out, _ = self._run(feed)
        assert "r/stocks: <no posts found mentioning NVDA" in out
        assert "r/investing: <no posts found mentioning NVDA" in out

    def test_unknown_subreddit_tag_is_shown_not_dropped(self):
        feed = _combined_atom([("w1", "wallstreetbets"), ("x1", "SomeOtherSub"), ("u1", None)])
        out, _ = self._run(feed)
        assert "x1" in out and "r/SomeOtherSub" in out    # tagged outside our list
        assert "u1" in out and "r/unknown" in out         # no tag at all

    def test_full_page_does_not_claim_a_quiet_subreddit_has_no_posts(self):
        # 100 hits all from one subreddit: the others may just be beyond the page.
        feed = _combined_atom([(f"w{i}", "wallstreetbets") for i in range(100)])
        out, _ = self._run(feed)
        assert "r/stocks: <not among the newest 100 matches" in out
        assert "r/stocks: <no posts found" not in out

    def test_429_retries_once_after_a_long_wait_then_succeeds(self):
        err = _http_error(429, "Too Many Requests", {})
        feed = _combined_atom([("w1", "wallstreetbets")])
        with patch.object(reddit, "urlopen",
                          side_effect=[err, _resp(lambda: feed)]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            out = reddit.fetch_reddit_posts("NVDA", subreddits=self._SUBS)
        assert op.call_count == 2 and slept.call_count == 1
        assert "w1" in out


@pytest.mark.unit
class TestFetchFailureNotConfusedWithSilence:
    """2026-07-29 finding: a rate-limited/failed fetch must never render as
    "no posts found" -- the sentiment analyst read that phrasing as confirmed
    market silence on a real run and folded it into a bearish signal, when
    the truth was just a failed HTTP request."""

    def test_failed_fetch_is_labeled_unavailable_not_no_posts(self):
        with patch.object(reddit, "_fetch_subreddit", return_value=None):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks", "investing"))
        assert "data unavailable" in out
        assert "NOT evidence of an absence" in out
        assert "no posts found" not in out
        # The blanket "confirmed silence" fallback must not fire on a failure.
        assert "<no Reddit posts found mentioning" not in out

    def test_http_429_exhausted_renders_as_unavailable_end_to_end(self):
        err = _http_error(429, "Too Many Requests", {})
        with patch.object(reddit, "urlopen", side_effect=[err, err]), \
             patch.object(reddit.time, "sleep"):
            out = reddit.fetch_reddit_posts("NVDA")
        assert "data unavailable" in out
        assert "no posts found" not in out

    def test_http_403_renders_as_unavailable_with_a_single_request(self):
        err = _http_error(403, "Blocked", {})
        with patch.object(reddit, "urlopen", side_effect=err) as op, \
             patch.object(reddit.time, "sleep", side_effect=AssertionError("403 is not retried")):
            out = reddit.fetch_reddit_posts("NVDA")
        assert op.call_count == 1
        assert "data unavailable" in out

    def test_genuinely_empty_result_still_uses_confirmed_silence_message(self):
        # Preserves the old behavior for the case this message is actually true.
        with patch.object(reddit, "_fetch_subreddit", return_value=[]):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks", "investing"))
        assert "<no Reddit posts found mentioning NVDA" in out
        assert "data unavailable" not in out


@pytest.mark.unit
class TestCryptoSearchTerm:
    """A crypto pair (BTC-USD) barely matches Reddit text; search the base (#1113)."""

    def _captured_ticker(self, ticker):
        seen = {}

        def fake_fetch(t, sub, limit, timeout):
            seen["ticker"] = t
            return []

        with patch.object(reddit, "_fetch_subreddit", side_effect=fake_fetch):
            reddit.fetch_reddit_posts(ticker, subreddits=("stocks",))
        return seen["ticker"]

    def test_crypto_pair_searches_base(self):
        assert self._captured_ticker("BTC-USD") == "BTC"

    def test_equity_passes_through(self):
        assert self._captured_ticker("NVDA") == "NVDA"


@pytest.mark.unit
class TestDiscontinuedSwitch:
    """Reddit ended RSS on 2026-11-13: the shipped default makes no request."""

    @pytest.fixture(autouse=True)
    def _rss_enabled(self):   # override the module fixture: test the real default
        yield

    def test_shipped_default_is_discontinued(self):
        assert reddit._RSS_DISCONTINUED is True

    def test_makes_no_request_and_never_sleeps(self):
        with patch.object(reddit, "urlopen", side_effect=AssertionError("no request expected")), \
             patch.object(reddit.time, "sleep", side_effect=AssertionError("no wait expected")):
            out = reddit.fetch_reddit_posts("NVDA")
        assert "discontinued" in out and "permanently offline" in out
        assert "NOT evidence of an absence of NVDA discussion" in out

    def test_placeholder_is_not_read_as_silence(self):
        out = reddit.fetch_reddit_posts("NVDA")
        assert "no posts found" not in out and "no Reddit posts" not in out

    def test_crypto_pair_uses_the_base_symbol(self):
        assert "absence of BTC discussion" in reddit.fetch_reddit_posts("BTC-USD")

    def test_switching_it_off_restores_the_fetch(self, monkeypatch):
        monkeypatch.setattr(reddit, "_RSS_DISCONTINUED", False)
        with patch.object(reddit, "urlopen", return_value=_atom_resp()) as op:
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert op.call_count == 1 and "NVDA earnings beat" in out
