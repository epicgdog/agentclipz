"""Tests for the message buffer, pruning, and chat speed computation."""

import time

from twitch_monitor.analytics import ChatAnalytics


class TestBufferPruning:
    def test_messages_within_window_are_kept(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        for i in range(5):
            a.add_message(f"msg {i}", timestamp=now - i)

        metrics = a.get_chat_metrics(now=now)
        assert metrics["messages_in_window"] == 5

    def test_old_messages_are_pruned(self):
        a = ChatAnalytics(window_seconds=10)
        now = 1000.0
        a.add_message("old", timestamp=now - 20)
        a.add_message("also old", timestamp=now - 15)
        a.add_message("fresh", timestamp=now - 3)

        metrics = a.get_chat_metrics(now=now)
        assert metrics["messages_in_window"] == 1

    def test_empty_buffer_returns_zero_speed(self):
        a = ChatAnalytics(window_seconds=60)
        metrics = a.get_chat_metrics(now=1000.0)
        assert metrics["messages_in_window"] == 0
        assert metrics["messages_per_second"] == 0.0

    def test_boundary_message_is_pruned(self):
        """A message strictly older than the window is pruned; one at the
        exact cutoff edge (< is strict) is retained by the implementation."""
        a = ChatAnalytics(window_seconds=10)
        now = 1000.0
        a.add_message("just past cutoff", timestamp=now - 10.01)
        metrics = a.get_chat_metrics(now=now)
        assert metrics["messages_in_window"] == 0

    def test_message_just_inside_window_is_kept(self):
        a = ChatAnalytics(window_seconds=10)
        now = 1000.0
        a.add_message("just inside", timestamp=now - 9.99)
        metrics = a.get_chat_metrics(now=now)
        assert metrics["messages_in_window"] == 1


class TestChatSpeed:
    def test_known_speed(self):
        a = ChatAnalytics(window_seconds=60)
        now = 1000.0
        for i in range(30):
            a.add_message("test", timestamp=now - i)

        speed = a.compute_chat_speed(now=now)
        assert speed["messages_in_window"] == 30
        assert speed["messages_per_second"] == 0.5

    def test_speed_after_pruning(self):
        a = ChatAnalytics(window_seconds=10)
        now = 1000.0
        for i in range(20, 0, -1):
            a.add_message("msg", timestamp=now - i)

        speed = a.compute_chat_speed(now=now)
        assert speed["messages_in_window"] == 10
        assert speed["messages_per_second"] == 1.0
