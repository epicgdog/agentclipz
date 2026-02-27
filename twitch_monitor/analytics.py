"""Real-time chat analytics: speed, keywords, and sentiment over a rolling window."""

from __future__ import annotations

import re
import string
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SECONDS = 60
DEFAULT_TOP_N_KEYWORDS = 10
SENTIMENT_POS_THRESHOLD = 0.05
SENTIMENT_NEG_THRESHOLD = -0.05

# ---------------------------------------------------------------------------
# Lightweight stopwords (covers the most common English filler words)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "this", "that",
    "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "no", "so", "up", "out", "about",
    "just", "than", "then", "too", "very", "also", "into", "over",
    "such", "only", "own", "same", "some", "other", "all", "each",
    "every", "both", "few", "more", "most", "any", "how", "what",
    "which", "who", "whom", "when", "where", "why", "here", "there",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "am", "im", "dont",
    "get", "got", "like", "lol", "oh", "yeah", "yes", "ok", "okay",
}

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


# ---------------------------------------------------------------------------
# Buffer entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _MessageEntry:
    timestamp: float
    text: str
    compound_score: float


# ---------------------------------------------------------------------------
# ChatAnalytics
# ---------------------------------------------------------------------------

class ChatAnalytics:
    """Accumulates chat messages in a rolling window and computes live metrics."""

    def __init__(
        self,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        top_n_keywords: int = DEFAULT_TOP_N_KEYWORDS,
    ) -> None:
        self.window_seconds = window_seconds
        self.top_n_keywords = top_n_keywords
        self._buffer: deque[_MessageEntry] = deque()
        self._sentiment = SentimentIntensityAnalyzer()

    # -- ingestion ---------------------------------------------------------

    def add_message(self, text: str, timestamp: float | None = None) -> None:
        """Record a new chat message and its sentiment score."""
        ts = timestamp if timestamp is not None else time.time()
        compound = self._sentiment.polarity_scores(text)["compound"]
        self._buffer.append(_MessageEntry(timestamp=ts, text=text, compound_score=compound))

    # -- pruning -----------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    # -- chat speed --------------------------------------------------------

    def compute_chat_speed(self, now: float | None = None) -> dict[str, float]:
        now = now or time.time()
        self._prune(now)
        count = len(self._buffer)
        mps = count / self.window_seconds if self.window_seconds else 0.0
        return {
            "messages_in_window": count,
            "messages_per_second": round(mps, 2),
            "messages_per_minute": round(mps * 60, 2),
        }

    # -- keyword extraction ------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = _PUNCT_RE.sub(" ", text.lower())
        return [tok for tok in text.split() if len(tok) > 2 and tok not in STOPWORDS]

    def compute_top_keywords(
        self, n: int | None = None, now: float | None = None
    ) -> list[tuple[str, int]]:
        now = now or time.time()
        self._prune(now)
        n = n or self.top_n_keywords
        counter: Counter[str] = Counter()
        for entry in self._buffer:
            counter.update(self._tokenize(entry.text))
        return counter.most_common(n)

    # -- sentiment ---------------------------------------------------------

    def compute_sentiment(self, now: float | None = None) -> dict[str, Any]:
        now = now or time.time()
        self._prune(now)

        if not self._buffer:
            return {
                "avg_compound": 0.0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
                "total": 0,
            }

        pos = neu = neg = 0
        total_compound = 0.0
        for entry in self._buffer:
            total_compound += entry.compound_score
            if entry.compound_score >= SENTIMENT_POS_THRESHOLD:
                pos += 1
            elif entry.compound_score <= SENTIMENT_NEG_THRESHOLD:
                neg += 1
            else:
                neu += 1

        count = len(self._buffer)
        return {
            "avg_compound": round(total_compound / count, 4),
            "positive_count": pos,
            "neutral_count": neu,
            "negative_count": neg,
            "total": count,
        }

    # -- unified metrics ---------------------------------------------------

    def get_chat_metrics(self, now: float | None = None) -> dict[str, Any]:
        """Return a single snapshot of all chat metrics for the current window."""
        now = now or time.time()
        self._prune(now)
        return {
            "window_seconds": self.window_seconds,
            **self.compute_chat_speed(now),
            "top_keywords": self.compute_top_keywords(now=now),
            "sentiment": self.compute_sentiment(now),
        }
