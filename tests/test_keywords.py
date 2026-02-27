"""Tests for keyword extraction and stopword filtering."""

from twitch_monitor.analytics import ChatAnalytics


class TestKeywordExtraction:
    def test_top_keywords_basic(self):
        a = ChatAnalytics(window_seconds=60, top_n_keywords=5)
        now = 1000.0
        a.add_message("pogchamp pogchamp pogchamp", timestamp=now)
        a.add_message("hype hype pogchamp", timestamp=now)
        a.add_message("gg gg hype", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        words = [w for w, _ in keywords]
        assert words[0] == "pogchamp"
        assert "hype" in words
        assert "gg" in words or len(words) <= 5

    def test_stopwords_are_filtered(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("this is the best stream", timestamp=now)
        a.add_message("the stream is so good", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        keyword_words = {w for w, _ in keywords}
        assert "the" not in keyword_words
        assert "this" not in keyword_words
        assert "is" not in keyword_words

    def test_short_tokens_are_filtered(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("a I am ok go do it me", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        keyword_words = {w for w, _ in keywords}
        for w in keyword_words:
            assert len(w) > 2

    def test_punctuation_is_stripped(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("amazing! awesome! amazing!", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        words = [w for w, _ in keywords]
        assert "amazing" in words
        assert "amazing!" not in words

    def test_empty_buffer_returns_empty_keywords(self):
        a = ChatAnalytics(window_seconds=60)
        keywords = a.compute_top_keywords(now=1000.0)
        assert keywords == []

    def test_keyword_count_limit(self):
        a = ChatAnalytics(window_seconds=60, top_n_keywords=3)
        now = 1000.0
        for word in ["alpha", "bravo", "charlie", "delta", "echo"]:
            a.add_message(f"{word} {word}", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        assert len(keywords) <= 3

    def test_case_insensitivity(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("PogChamp POGCHAMP pogchamp", timestamp=now)

        keywords = a.compute_top_keywords(now=now)
        words = [w for w, _ in keywords]
        assert "pogchamp" in words
        counts = {w: c for w, c in keywords}
        assert counts["pogchamp"] == 3
