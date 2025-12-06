#!/usr/bin/env python3
"""
YouTube Study Notes Generator
=============================
Converts YouTube videos into structured study notes using AI.

Supported Providers:
  - Google Gemini 2.5 Flash (FREE - 1M token context)
  - Groq Llama 3.3 70B (FREE - 128K token context)
  - Z.AI GLM-4.6 (Paid - Coding Plan)

Usage:
    python app.py                           # Interactive mode
    python app.py "https://youtube.com/..." # Direct URL mode
"""

import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Callable
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Load environment variables
load_dotenv(override=True)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FOLDER = "YouTubeNotes"
SYSTEM_PROMPT_FILE = "gpt-inst.md"
REQUEST_TIMEOUT = 300  # 5 minutes
MAX_OUTPUT_TOKENS = 8192

# Provider configurations
PROVIDERS = {
    "gemini": {
        "name": "Google Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "context": "1M tokens",
        "context_tokens": 1_000_000,
        "free": True,
    },
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "context": "128K tokens",
        "context_tokens": 128_000,
        "rate_limit_tpm": 12_000,  # Free tier TPM limit
        "free": True,
    },
    "zai": {
        "name": "Z.AI GLM-4.6",
        "model": "glm-4.6",
        "env_key": "ZAI_API_KEY",
        "context": "32K tokens",
        "context_tokens": 32_000,
        "free": False,
    },
}


def estimate_tokens(word_count: int) -> int:
    """Estimate token count from word count (roughly 1.3 tokens per word for English)."""
    return int(word_count * 1.3)


# ============================================================
# UTILITIES
# ============================================================

class ProgressIndicator:
    """Animated spinner for long-running operations."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing"):
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _animate(self):
        idx = 0
        while self._running:
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stdout.write(f"\r   {frame} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)


def parse_api_error(response: requests.Response, provider: str) -> str:
    """Extract error message from API response."""
    try:
        err_json = response.json()
        return err_json.get("error", {}).get("message", response.text)
    except Exception:
        return response.text


# ============================================================
# PROVIDER SELECTION
# ============================================================

def get_available_providers() -> Dict[str, dict]:
    """Return providers that have API keys configured."""
    return {
        key: config
        for key, config in PROVIDERS.items()
        if os.getenv(config["env_key"])
    }


def select_provider_with_stats(word_count: int) -> str:
    """Display provider menu with token usage stats and return selected provider key."""
    estimated_tokens = estimate_tokens(word_count)
    
    print("\n" + "=" * 60)
    print("  🤖 Select AI Provider")
    print("=" * 60)
    print(f"\n  📊 Transcript: ~{word_count:,} words (~{estimated_tokens:,} tokens)")
    print("-" * 60)

    provider_keys = list(PROVIDERS.keys())
    available = get_available_providers()
    
    # Find best recommendation (available provider with lowest usage that fits and doesn't exceed rate limit)
    best_provider = None
    best_usage = float('inf')
    for key, config in PROVIDERS.items():
        if key in available:
            usage = (estimated_tokens / config["context_tokens"]) * 100
            rate_limit = config.get("rate_limit_tpm")
            exceeds_rate_limit = rate_limit and estimated_tokens > rate_limit
            if usage < 80 and usage < best_usage and not exceeds_rate_limit:
                best_usage = usage
                best_provider = key

    for idx, (key, config) in enumerate(PROVIDERS.items(), 1):
        is_available = key in available
        usage_pct = (estimated_tokens / config["context_tokens"]) * 100
        rate_limit = config.get("rate_limit_tpm")
        exceeds_rate_limit = rate_limit and estimated_tokens > rate_limit
        
        # Status indicators
        if not is_available:
            status = "❌ (no API key)"
            recommend = ""
        elif exceeds_rate_limit:
            status = f"⚠️  Exceeds {rate_limit // 1000}K TPM rate limit"
            recommend = ""
        elif usage_pct > 80:
            status = f"⚠️  Usage: {usage_pct:.0f}% (may truncate)"
            recommend = ""
        else:
            status = f"✅ Usage: {usage_pct:.1f}%"
            recommend = " ⭐ Recommended" if key == best_provider else ""
        
        tag = " [FREE]" if config["free"] else " [PAID]"
        print(f"\n  {idx}. {config['name']}{tag}{recommend}")
        print(f"     Context: {config['context']} | {status}")

    print("\n" + "-" * 60)

    if not available:
        print("\n❌ No API keys configured!")
        print("   Add at least one key to your .env file:")
        print("   - GEMINI_API_KEY  (free: https://aistudio.google.com)")
        print("   - GROQ_API_KEY    (free: https://console.groq.com)")
        print("   - ZAI_API_KEY     (paid: https://z.ai)")
        sys.exit(1)

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(PROVIDERS)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(provider_keys):
                selected = provider_keys[idx]
                if selected in available:
                    return selected
                print(f"   ❌ {PROVIDERS[selected]['name']} not configured.")
            else:
                print("   Invalid choice.")
        except ValueError:
            print("   Please enter a number.")


# ============================================================
# LLM PROVIDERS
# ============================================================

def call_gemini(system_prompt: str, user_message: str) -> str:
    """Call Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": MAX_OUTPUT_TOKENS},
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API Error {response.status_code}: {parse_api_error(response, 'gemini')}")

    result = response.json()
    if "candidates" not in result or not result["candidates"]:
        raise Exception(f"Unexpected Gemini response: {result}")

    return result["candidates"][0]["content"]["parts"][0]["text"]


def call_groq(system_prompt: str, user_message: str) -> str:
    """Call Groq API."""
    api_key = os.getenv("GROQ_API_KEY")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise Exception(f"Groq API Error {response.status_code}: {parse_api_error(response, 'groq')}")

    result = response.json()
    if "choices" not in result or not result["choices"]:
        raise Exception(f"Unexpected Groq response: {result}")

    if result["choices"][0].get("finish_reason") == "length":
        print("⚠️  Warning: Response was truncated.")

    return result["choices"][0]["message"]["content"]


def call_zai(system_prompt: str, user_message: str) -> str:
    """Call Z.AI GLM-4.6 API with streaming."""
    api_key = os.getenv("ZAI_API_KEY")

    response = requests.post(
        "https://api.z.ai/api/coding/paas/v4/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        },
        json={
            "model": "glm-4.6",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": True,  # Enable streaming
        },
        timeout=REQUEST_TIMEOUT,
        stream=True,  # Enable chunked transfer
    )

    if response.status_code != 200:
        raise Exception(f"Z.AI API Error {response.status_code}: {parse_api_error(response, 'zai')}")

    # Parse SSE stream and collect content
    full_content = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                text = delta["content"]
                full_content.append(text)
                print(text, end="", flush=True)  # Real-time output

    print()  # Newline after streaming completes
    return "".join(full_content)


# Provider dispatch table
PROVIDER_FUNCTIONS: Dict[str, Callable[[str, str], str]] = {
    "gemini": call_gemini,
    "groq": call_groq,
    "zai": call_zai,
}


def generate_notes(provider: str, system_prompt: str, transcript: str, video_title: str, video_id: str, chapters: list = None) -> str:
    """Generate study notes using the selected provider."""
    
    # Build chapters section if available
    chapters_info = ""
    if chapters:
        chapters_info = "\n\nYouTube Chapters (use these as Key Moments):\n"
        for ch in chapters:
            chapters_info += f"- [{ch['time']}] ({ch['seconds']}s) {ch['title']}\n"
        chapters_info += "\nNote: Chapters are provided. Use these for the KEY MOMENTS section instead of generating your own.\n"
    
    user_message = (
        f"Create study notes from this YouTube video transcript.\n\n"
        f"Video Title: {video_title}\n"
        f"Video ID: {video_id}"
        f"{chapters_info}\n\n"
        f"Transcript:\n{transcript}"
    )

    config = PROVIDERS[provider]
    print(f"🤖 Generating notes with {config['name']}...")

    if provider == "zai":
        # Z.AI streams output directly, no spinner needed
        print()  # Blank line before streamed output
        return PROVIDER_FUNCTIONS[provider](system_prompt, user_message)
    else:
        with ProgressIndicator("Generating notes (this may take 1-3 minutes)..."):
            return PROVIDER_FUNCTIONS[provider](system_prompt, user_message)


# ============================================================
# YOUTUBE FUNCTIONS
# ============================================================

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    # Short URL format
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]

    # Standard formats
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        for prefix in ("/embed/", "/v/", "/shorts/"):
            if parsed.path.startswith(prefix):
                return parsed.path.split("/")[2]

    # Fallback: regex for video ID pattern
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&?/]|$)", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract video ID from: {url}")


def format_duration(seconds: int) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    if seconds is None:
        return "Unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_video_info(video_id: str) -> Dict[str, any]:
    """Fetch video metadata including chapters using yt-dlp."""
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            
            # Extract chapters if available
            chapters = []
            if info.get("chapters"):
                for ch in info["chapters"]:
                    start_seconds = int(ch["start_time"])
                    minutes, seconds = divmod(start_seconds, 60)
                    chapters.append({
                        "time": f"{minutes}:{seconds:02d}",
                        "seconds": start_seconds,
                        "title": ch["title"],
                    })
            
            return {
                "title": info.get("title", video_id),
                "channel": info.get("channel", info.get("uploader", "Unknown")),
                "duration": format_duration(info.get("duration")),
                "chapters": chapters,
            }
    except Exception as e:
        print(f"   Warning: Could not fetch video info ({e})")
        return {"title": video_id, "channel": "Unknown", "duration": "Unknown", "chapters": []}


def fetch_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube with fallback to any available language."""
    api = YouTubeTranscriptApi()

    # Try English first
    try:
        data = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        # Fallback: try any available transcript
        try:
            transcripts = api.list(video_id)
            for t in transcripts:
                try:
                    data = t.translate("en").fetch() if t.is_translatable else t.fetch()
                    break
                except Exception:
                    continue
            else:
                raise NoTranscriptFound(video_id, ["en"], "No usable transcript")
        except Exception as e:
            raise NoTranscriptFound(video_id, ["en"], str(e))

    # Extract text from transcript data
    if hasattr(data, "snippets"):
        text = " ".join(s.text for s in data.snippets)
    elif isinstance(data, list):
        text = " ".join(entry.get("text", "") for entry in data)
    else:
        text = " ".join(str(entry) for entry in data)

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# TRANSCRIPT CACHING
# ============================================================

TRANSCRIPTS_FOLDER = "transcripts"


def get_transcript_cache_path(video_id: str) -> str:
    """Get path to cached transcript file."""
    cache_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER, TRANSCRIPTS_FOLDER)
    return os.path.join(cache_dir, f"{video_id}.txt")


def load_cached_transcript(video_id: str) -> Optional[str]:
    """Load transcript from cache if it exists."""
    cache_path = get_transcript_cache_path(video_id)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_transcript_to_cache(video_id: str, transcript: str) -> str:
    """Save transcript to cache folder."""
    cache_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER, TRANSCRIPTS_FOLDER)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{video_id}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return cache_path


# ============================================================
# FILE OPERATIONS
# ============================================================

def get_script_dir() -> str:
    """Get directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def load_system_prompt() -> str:
    """Load system prompt from gpt-inst.md file."""
    path = os.path.join(get_script_dir(), SYSTEM_PROMPT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"System prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Create a filesystem-safe filename from title."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    safe = re.sub(r"[\s_]+", "_", safe).strip("_.")
    return safe[:max_length] or "untitled"


def find_existing_note(video_id: str, model_name: str) -> Optional[str]:
    """Find existing note file for the same video ID and model."""
    output_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER)
    if not os.path.exists(output_dir):
        return None

    # Match both video ID in content AND model name in filename
    video_pattern = f"https://www.youtube.com/watch?v={video_id}"
    for filename in os.listdir(output_dir):
        if filename.endswith(".md") and model_name in filename:
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    if video_pattern in f.read(500):
                        return filepath
            except Exception:
                continue
    return None


def calculate_read_time(word_count: int, wpm: int = 200) -> int:
    """Calculate estimated read time in minutes."""
    return max(1, round(word_count / wpm))


def save_notes(
    notes: str,
    video_title: str,
    video_id: str,
    provider: str,
    channel: str = "Unknown",
    duration: str = "Unknown",
    transcript_words: int = 0,
) -> str:
    """Save notes to YouTubeNotes folder. Overwrites if same video+model exists."""
    output_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    model_name = PROVIDERS[provider]["model"]

    # Check for existing note with same video ID and model
    existing = find_existing_note(video_id, model_name)
    if existing:
        filepath = existing
        print(f"   Overwriting: {os.path.basename(filepath)}")
    else:
        filename = f"{video_id}_{sanitize_filename(video_title)}_{model_name}.md"
        filepath = os.path.join(output_dir, filename)

    # Calculate read time based on notes word count
    notes_word_count = len(notes.split())
    read_time = calculate_read_time(notes_word_count)

    # Create header with enriched metadata
    header = (
        f"<!-- \n"
        f"Source: https://www.youtube.com/watch?v={video_id}\n"
        f"Channel: {channel}\n"
        f"Duration: {duration}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Provider: {PROVIDERS[provider]['name']}\n"
        f"Transcript Words: ~{transcript_words:,}\n"
        f"-->\n\n"
    )

    # Prepend read time estimate to notes
    read_time_line = f"**Estimated Read Time:** {read_time} min\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + read_time_line + notes)

    return filepath


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  📚 YouTube Study Notes Generator")
    print("=" * 60)

    # Get URL first
    url = sys.argv[1] if len(sys.argv) > 1 else input("\n🎬 Enter YouTube URL: ").strip()
    if not url:
        print("❌ No URL provided.")
        sys.exit(1)

    if len(sys.argv) > 1:
        print(f"\n🎬 YouTube URL: {url}")

    try:
        # Process video
        print("\n📎 Extracting video ID...")
        video_id = extract_video_id(url)
        print(f"   Video ID: {video_id}")

        print("📝 Fetching video info...")
        video_info = get_video_info(video_id)
        print(f"   Title: {video_info['title']}")
        print(f"   Channel: {video_info['channel']}")
        print(f"   Duration: {video_info['duration']}")
        if video_info.get("chapters"):
            print(f"   Chapters: {len(video_info['chapters'])} found ✓")
        else:
            print(f"   Chapters: None (LLM will generate key moments)")

        # Check transcript cache first
        print("📜 Fetching transcript...")
        transcript = load_cached_transcript(video_id)
        if transcript:
            print(f"   Loaded from cache")
        else:
            transcript = fetch_transcript(video_id)
            save_transcript_to_cache(video_id, transcript)
            print(f"   Fetched and cached")
        transcript_words = len(transcript.split())
        print(f"   Transcript: {transcript_words:,} words")

        # Now select provider with stats
        provider = select_provider_with_stats(transcript_words)
        print(f"\n✅ Selected: {PROVIDERS[provider]['name']}")

        print("\n📋 Loading system prompt...")
        system_prompt = load_system_prompt()
        print(f"   Loaded: {SYSTEM_PROMPT_FILE}")

        # Generate and save
        notes = generate_notes(
            provider,
            system_prompt,
            transcript,
            video_info["title"],
            video_id,
            chapters=video_info.get("chapters", []),
        )

        print("💾 Saving notes...")
        filepath = save_notes(
            notes=notes,
            video_title=video_info["title"],
            video_id=video_id,
            provider=provider,
            channel=video_info["channel"],
            duration=video_info["duration"],
            transcript_words=transcript_words,
        )

        print("\n" + "=" * 50)
        print("✅ Success!")
        print(f"   {filepath}")
        print("=" * 50 + "\n")

    except TranscriptsDisabled:
        print("\n❌ Transcripts are disabled for this video.")
        sys.exit(1)
    except NoTranscriptFound:
        print("\n❌ No transcript found for this video.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
