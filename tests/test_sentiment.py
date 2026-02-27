"""Tests for sentiment aggregation."""

from twitch_monitor.analytics import ChatAnalytics


class TestSentiment:
    def test_positive_messages(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("This is absolutely wonderful and amazing!", timestamp=now)
        a.add_message("I love this stream so much!", timestamp=now)
        a.add_message("Great play, you are the best!", timestamp=now)

        s = a.compute_sentiment(now=now)
        assert s["avg_compound"] > 0
        assert s["positive_count"] >= 2
        assert s["total"] == 3

    def test_negative_messages(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("This is terrible and awful", timestamp=now)
        a.add_message("Worst stream ever, so bad", timestamp=now)
        a.add_message("I hate this, disgusting", timestamp=now)

        s = a.compute_sentiment(now=now)
        assert s["avg_compound"] < 0
        assert s["negative_count"] >= 2
        assert s["total"] == 3

    def test_mixed_sentiment(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("This is absolutely amazing!", timestamp=now)
        a.add_message("This is the worst thing ever", timestamp=now)

        s = a.compute_sentiment(now=now)
        assert s["positive_count"] >= 1
        assert s["negative_count"] >= 1
        assert s["total"] == 2

    def test_empty_buffer_sentiment(self):
        a = ChatAnalytics(window_seconds=60)
        s = a.compute_sentiment(now=1000.0)
        assert s["avg_compound"] == 0.0
        assert s["positive_count"] == 0
        assert s["neutral_count"] == 0
        assert s["negative_count"] == 0
        assert s["total"] == 0

    def test_sentiment_counts_sum_to_total(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        messages = [
            "I love this",
            "meh okay",
            "terrible play",
            "nice shot!",
            "whatever",
        ]
        for msg in messages:
            a.add_message(msg, timestamp=now)

        s = a.compute_sentiment(now=now)
        assert s["positive_count"] + s["neutral_count"] + s["negative_count"] == s["total"]
        assert s["total"] == 5
