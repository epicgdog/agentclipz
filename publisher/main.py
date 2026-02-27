import os
import sys
from dotenv import load_dotenv
from instagrapi import Client

# Load environment variables from .env
load_dotenv()

def login_instagram():
    """Authenticate with Instagram using credentials from .env."""
    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")
    
    if not username or not password:
        print("Error: IG_USERNAME and IG_PASSWORD must be set in the .env file.")
        sys.exit(1)

    print(f"Logging into Instagram as {username}...")
    cl = Client()
    
    try:
        # Optional: instagrapi supports saving/loading sessions to avoid frequent logins
        # but for simplicity, we login directly here.
        cl.login(username, password)
        print("Login successful!")
        return cl
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)


def post_photo(cl: Client, image_path: str, caption: str = ""):
    """Post a local image to Instagram feed."""
    print(f"Publishing photo: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return
    
    try:
        media = cl.photo_upload(image_path, caption)
        print(f"Photo posted successfully! Media ID: {media.pk}")
    except Exception as e:
        print(f"Failed to post photo: {e}")


def post_clip(cl: Client, video_path: str, caption: str = ""):
    """Post a local video to Instagram as a Reel."""
    print(f"Publishing video/clip as Reel: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' does not exist.")
        return
    
    try:
        media = cl.clip_upload(video_path, caption)
        print(f"Clip posted successfully! Media ID: {media.pk}")
    except Exception as e:
        print(f"Failed to post clip: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python main.py photo <path_to_image> [caption]")
        print("  python main.py clip <path_to_video> [caption]")
        print("\nExamples:")
        print("  python main.py photo my_picture.jpg \"Check this out!\"")
        print("  python main.py clip my_video.mp4 \"Epic highlight 🔥\"")
        sys.exit(1)

    action = sys.argv[1].lower()
    media_path = sys.argv[2]
    # Join remaining args as caption, otherwise use a default
    caption = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else "Autoposted by AgentClipz"

    # Authenticate
    ig_client = login_instagram()

    # Publish
    if action == "photo":
        post_photo(ig_client, media_path, caption)
    elif action == "clip":
        post_clip(ig_client, media_path, caption)
    else:
        print(f"Unknown command: '{action}'. Use 'photo' or 'clip'.")
