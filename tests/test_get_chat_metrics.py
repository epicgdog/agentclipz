"""Tests for the unified get_chat_metrics function."""

from twitch_monitor.analytics import ChatAnalytics


class TestGetChatMetrics:
    def test_returns_all_expected_keys(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("hello world", timestamp=now)

        metrics = a.get_chat_metrics(now=now)

        assert "window_seconds" in metrics
        assert "messages_in_window" in metrics
        assert "messages_per_second" in metrics
        assert "top_keywords" in metrics

    def test_metrics_internal_consistency(self):
        a = ChatAnalytics(window_seconds=30)
        now = 1000.0
        for i in range(15):
            a.add_message(f"message {i}", timestamp=now - i)

        metrics = a.get_chat_metrics(now=now)

        assert metrics["window_seconds"] == 30
        assert metrics["messages_in_window"] == 15
        assert metrics["messages_per_second"] == round(15 / 30, 2)

    def test_empty_metrics(self):
        a = ChatAnalytics(window_seconds=60)
        metrics = a.get_chat_metrics(now=1000.0)

        assert metrics["messages_in_window"] == 0
        assert metrics["messages_per_second"] == 0.0
        assert metrics["top_keywords"] == []

    def test_keywords_are_list_of_tuples(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("pogchamp pogchamp hype", timestamp=now)

        metrics = a.get_chat_metrics(now=now)
        keywords = metrics["top_keywords"]

        assert isinstance(keywords, list)
        for item in keywords:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], int)

    def test_metrics_json_serializable(self):
        """Verify the metrics dict can be JSON-serialized for API use."""
        import json

        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        a.add_message("test message", timestamp=now)

        metrics = a.get_chat_metrics(now=now)
        serialized = json.dumps(metrics)
        deserialized = json.loads(serialized)
        assert deserialized["messages_in_window"] == 1
