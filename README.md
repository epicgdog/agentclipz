# Twitch Chat Monitor

Real-time Twitch chat analytics: chat speed, top keywords, and sentiment — powered by TwitchIO, VADER sentiment analysis, and a Rich terminal dashboard.

## Quick start

### 1. Get a Twitch OAuth token

1. Go to <https://dev.twitch.tv/console/apps> and register an application (or use an existing one).
2. Generate an OAuth token at <https://twitchapps.com/tmi/> (or use the Twitch CLI).
   The token looks like `oauth:abc123...`.

### 2. Set environment variables

```bash
export TWITCH_TOKEN="oauth:your_token_here"
export TWITCH_CHANNEL="channel_name"
```

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 4. Run

There are three ways to launch the monitor:

```bash
# Rich terminal dashboard (recommended)
python -m twitch_monitor.interface_cli

# Plain-text console output
python run.py
# or
python -m twitch_monitor
```

The **dashboard** mode uses Rich to render a live-updating terminal UI with panels for chat speed, top keywords, and sentiment. The **plain-text** mode prints a metrics snapshot to stdout every 10 seconds.

## Terminal dashboard

The Rich dashboard refreshes every 3 seconds and shows three panels:

- **Chat Speed** — messages per second, messages per minute, total messages in the rolling window, and a visual activity bar.
- **Top Keywords** — the most frequent non-stopword terms in the window, with bar charts.
- **Sentiment** — average VADER compound score, mood label, and a breakdown of positive / neutral / negative message counts with percentages.

Press `Ctrl+C` to exit.

## Example plain-text output

```
============================================================
  Speed:     42.0 msg/min (0.70 msg/s, 42 in window)
  Keywords:  pogchamp(8), hype(5), gg(4), clip(3), letsgo(2)
  Sentiment: avg=+0.214  +18 / ~15 / -9
============================================================
```

## Configuration

All knobs can be set in code when constructing `ChatMonitorBot`:

| Parameter          | Default | Description                              |
| ------------------ | ------- | ---------------------------------------- |
| `window_seconds`   | 60      | Rolling window length for all metrics    |
| `top_n_keywords`   | 10      | Number of top keywords to track          |
| `report_interval`  | 10      | Seconds between console metric dumps     |

```python
from twitch_monitor import ChatMonitorBot

bot = ChatMonitorBot(
    token="oauth:...",
    channel="some_channel",
    window_seconds=120,
    top_n_keywords=15,
    report_interval=5,
)
bot.run()
```

## Using analytics programmatically

You can use `ChatAnalytics` on its own (e.g., to integrate with a web dashboard):

```python
from twitch_monitor import ChatAnalytics

analytics = ChatAnalytics(window_seconds=60)

analytics.add_message("PogChamp this is amazing!")
analytics.add_message("terrible play FailFish")

metrics = analytics.get_chat_metrics()
print(metrics)
```

`get_chat_metrics()` returns a dict:

```python
{
    "window_seconds": 60,
    "messages_in_window": 2,
    "messages_per_second": 0.03,
    "messages_per_minute": 2.0,
    "top_keywords": [("pogchamp", 1), ("amazing", 1), ("terrible", 1), ...],
    "sentiment": {
        "avg_compound": 0.1234,
        "positive_count": 1,
        "neutral_count": 0,
        "negative_count": 1,
        "total": 2,
    },
}
```

The output is fully JSON-serializable, making it easy to expose via HTTP, WebSocket, or log to a file.

## Running tests

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest tests/ -v
```

Tests cover:
- **Buffer & pruning** — messages expire correctly after the window elapses.
- **Chat speed** — MPS/MPM calculations for known inputs.
- **Keywords** — stopword removal, punctuation stripping, case insensitivity, count limits.
- **Sentiment** — positive/negative/mixed aggregation, empty-buffer edge case.
- **Unified metrics** — all keys present, internal consistency, JSON serializability.

## Interpreting sentiment

The compound score ranges from **-1.0** (most negative) to **+1.0** (most positive):

| Range              | Label    |
| ------------------ | -------- |
| >= +0.05           | Positive |
| -0.05 to +0.05    | Neutral  |
| <= -0.05           | Negative |

## Manual end-to-end testing

1. Set `TWITCH_TOKEN` and `TWITCH_CHANNEL` in your environment.
2. Run `python -m twitch_monitor.interface_cli`.
3. Observe the dashboard updating as chat messages arrive.
4. Verify that:
   - **Speed** metrics increase when chat activity spikes.
   - **Top keywords** reflect actual emotes and repeated terms.
   - **Sentiment** moves positive during hype and negative during frustration.
5. Press `Ctrl+C` to stop.

## Extending

- **HTTP/WebSocket endpoint**: Wrap `get_chat_metrics()` in a FastAPI or Flask route to feed a live dashboard.
- **OBS overlay**: Serve the same metrics as a small HTML page and add it as a browser source in OBS.
- **Persistent logging**: Write metrics snapshots to a file or database for historical analysis.
- **Custom NLP**: Swap out the keyword extraction or sentiment engine in `analytics.py` for something heavier (spaCy, transformers, etc.).

## Project structure

```
agentclipz/
├── requirements.txt
├── run.py                       # convenience launcher (plain-text mode)
├── README.md
├── tests/
│   ├── __init__.py
│   ├── test_analytics_buffer.py # buffer pruning & chat speed tests
│   ├── test_keywords.py         # keyword extraction tests
│   ├── test_sentiment.py        # sentiment aggregation tests
│   └── test_get_chat_metrics.py # unified metrics tests
└── twitch_monitor/
    ├── __init__.py
    ├── __main__.py              # python -m twitch_monitor (plain-text)
    ├── analytics.py             # rolling buffer, speed, keywords, sentiment
    ├── bot.py                   # TwitchIO bot + periodic reporting
    └── interface_cli.py         # Rich terminal dashboard
```
