# agentclipz 🎮🚀

**agentclipz** is an autonomous AI agent that monitors live Twitch streams, identifies hype moments, extracts clips, analyzes emotions, burns subtitles, and automatically publishes them to social media.

Turn your stream highlights into vertical content (Instagram Reels / TikTok) without lifting a finger.

## 🌟 Features

- **Real-time Stream Capturing**: Uses `streamlink` and `ffmpeg` to maintain a rolling buffer of your live HLS stream.
- **Hype-Based Triggering**: Monitors chat "messages per second" (MPS). When the chat goes wild, the agent triggers a recording.
- **Emotion Analysis**: Powered by **Modulate API**, the agent analyzes your voice to generate a transcript and detect emotional peaks (Amused, Excited, Happy, etc.).
- **Automatic Subtitling**: Generates and burns stylized subtitles directly into your clips from the AI transcript.
- **AI Copywriting**: Integrates with **Reka AI** to "watch" the clip (via transcript context) and write engaging, emoji-filled social media captions.
- **Auto-Publishing**: Automatically posts the subtitled highlight as an Instagram Reel.
- **Orphan Cleanup**: Robust process management that kills lingering background processes to keep your system clean.

---

## 🛠️ Setup

### 1. Requirements

- **Python 3.10+**
- **FFmpeg**: Must be installed and reachable in your system PATH.
- **Twitch Account**: For monitoring chat and stream data.
- **API Keys**:
    - **Modulate**: For emotion analysis and transcription.
    - **Reka**: For AI caption generation.

### 2. Installation

```bash
# Clone the repository
# git clone https://github.com/epicgdog/agentclipz
# cd agentclipz

# Setup virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory:

```env
# Twitch Config
TWITCH_TOKEN="oauth:your_token_here"
TWITCH_CHANNEL="your_channel_name"
TWITCH_STREAM_URL="https://twitch.tv/your_channel"

# AI Config
MODULATE_KEY="your_modulate_api_key"
REKA_API_KEY="your_reka_api_key"
REKA_MODEL="reka-core"

# Instagram Config (Producer)
IG_USERNAME="your_ig_username"
IG_PASSWORD="your_ig_password"

# Pipeline Settings
CLIP_START_MPS=0.5          # Chat speed to start recording
CLIP_STOP_MPS=0.2           # Chat speed to stop recording
BUFFER_DURATION_S=60        # Seconds of history to keep
MIN_RECORD_S=15             # Minimum clip length
MAX_RECORD_S=60             # Maximum clip length
CLIP_OUTPUT_DIR="./clips"
```

---

## 🚀 Usage

The simplest way to run the entire pipeline (monitor + trigger + analyzer + publisher) is the **Live Dashboard**:

```bash
python -m twitch_monitor.interface_cli
```

The dashboard shows:
- **Chat Speed (MPS)**: Live activity tracking.
- **Top Keywords**: What chat is talking about right now.
- **Bot Status**: Connection and recording state.
- **Last Clip Info**: Emotions detected in the most recent highlight.

---

## 📂 Project Structure

- `twitch_monitor/`: Core logic for stream capturing and chat polling.
- `twitch_monitor/clip_trigger.py`: The "brain" that detects peaks and manages recording.
- `twitch_monitor/clipper.py`: Integration with Modulate for emotion/subtitles.
- `publisher/main.py`: Integration with Reka AI and Instagram publishing.
- `highlight_clipper.py`: Logic for finding the best 30s window within a raw clip.
- `clips/`: Directory where raw videos, subtitles, and AI reports are saved.

---

## 🔒 Security & Performance

- **Safety First**: Orphaned FFmpeg and Streamlink processes are automatically identified and killed on startup and shutdown.
- **Event Loop aware**: Heavy video processing runs in background executors to prevent the chat bot or terminal UI from freezing.
- **Buffer Cleansing**: Chat and video buffers are reset after successful triggers to prevent duplicate clips from the same event.
