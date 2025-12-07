#!/usr/bin/env python3
"""
YouTube Study Notes Generator
=============================
Converts YouTube videos into structured study notes using AI.

Supported Providers:
  - Google Gemini 2.5 Flash (FREE - 1M token context)
  - Groq Llama 3.3 70B (FREE - 128K token context)
  - Z.AI GLM-4.6 (Paid - Coding Plan)
  - OpenRouter (Amazon Nova 2 Lite) (FREE - 32K token context)

Usage:
    python app.py                                    # Interactive mode
    python app.py "https://youtube.com/..."          # Direct URL, interactive prompt selection
    python app.py "URL" --prompt study-notes         # Use specific prompt
    python app.py --prompt quick-summary "URL"       # Argument order flexible
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Callable, List
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Load environment variables
load_dotenv(override=True)

# Import provider configurations from external file
from providers import PROVIDERS


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class TokenLimitExceeded(Exception):
    """Raised when a request exceeds TPM (tokens per minute) limits."""
    def __init__(self, limit: int, requested: int, message: str):
        self.limit = limit
        self.requested = requested
        super().__init__(message)

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FOLDER = "YouTubeNotes"
PROMPTS_FOLDER = "prompts"
DEFAULT_PROMPT = "study-notes"
REQUEST_TIMEOUT = 300  # 5 minutes
MAX_OUTPUT_TOKENS = 8192


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


def parse_tpm_error(error_message: str) -> dict | None:
    """
    Parse TPM error to extract limit and requested tokens.
    
    Example error:
    "Request too large... Limit 12000, Requested 20322..."
    
    Returns:
        {"limit": 12000, "requested": 20322} or None if not a TPM error
    """
    match = re.search(r'Limit\s+(\d+),\s*Requested\s+(\d+)', error_message, re.IGNORECASE)
    if match:
        return {"limit": int(match.group(1)), "requested": int(match.group(2))}
    return None


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
# PROMPT SELECTION
# ============================================================

def get_available_prompts() -> List[str]:
    """Scan prompts folder and return list of available prompt names."""
    prompts_dir = os.path.join(get_script_dir(), PROMPTS_FOLDER)
    if not os.path.exists(prompts_dir):
        return []
    
    prompts = []
    for f in sorted(os.listdir(prompts_dir)):
        if f.endswith(".md"):
            prompts.append(f[:-3])  # Remove .md extension
    return prompts


def get_prompt_description(prompt_name: str) -> str:
    """Get first line of prompt file as description."""
    path = os.path.join(get_script_dir(), PROMPTS_FOLDER, f"{prompt_name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            # Remove markdown heading markers
            return first_line.lstrip("#").strip()[:60]
    except Exception:
        return "Custom prompt template"


def select_prompt() -> str:
    """Display prompt menu and return selected prompt name."""
    prompts = get_available_prompts()
    
    if not prompts:
        print("\n❌ No prompts found in prompts/ folder!")
        sys.exit(1)
    
    if len(prompts) == 1:
        return prompts[0]
    
    print("\n" + "=" * 60)
    print("  📝 Select Note Format")
    print("=" * 60)
    
    for idx, prompt in enumerate(prompts, 1):
        is_default = prompt == DEFAULT_PROMPT
        default_tag = " [DEFAULT]" if is_default else ""
        description = get_prompt_description(prompt)
        print(f"\n  {idx}. {prompt}{default_tag}")
        print(f"     {description}")
    
    print("\n" + "-" * 60)
    
    # Find default index
    default_idx = prompts.index(DEFAULT_PROMPT) + 1 if DEFAULT_PROMPT in prompts else 1
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(prompts)}) or press Enter for default [{default_idx}]: ").strip()
            
            if not choice:
                return prompts[default_idx - 1]
            
            idx = int(choice) - 1
            if 0 <= idx < len(prompts):
                return prompts[idx]
            print("   Invalid choice.")
        except ValueError:
            print("   Please enter a number.")


# ============================================================
# LLM PROVIDERS
# ============================================================

def call_openai_compatible(system_prompt: str, user_message: str, config: dict) -> str:
    """
    Generic handler for all OpenAI-compatible APIs.
    Works with: Groq, OpenRouter, Together, DeepSeek, Fireworks, etc.
    """
    api_key = os.getenv(config["env_key"])

    response = requests.post(
        config["api_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        timeout=REQUEST_TIMEOUT,
    )

    # Check for TPM limit error (413)
    if response.status_code == 413:
        error_text = parse_api_error(response, config['name'])
        tpm_info = parse_tpm_error(error_text)
        
        if tpm_info:
            raise TokenLimitExceeded(
                limit=tpm_info["limit"],
                requested=tpm_info["requested"],
                message=error_text
            )
        # If we can't parse the TPM info, raise a generic exception
        raise Exception(f"{config['name']} API Error 413: {error_text}")

    if response.status_code != 200:
        raise Exception(f"{config['name']} API Error {response.status_code}: {parse_api_error(response, config['name'])}")

    result = response.json()
    if "choices" not in result or not result["choices"]:
        raise Exception(f"Unexpected {config['name']} response: {result}")

    if result["choices"][0].get("finish_reason") == "length":
        print("⚠️  Warning: Response was truncated.")

    return result["choices"][0]["message"]["content"]


def call_gemini(system_prompt: str, user_message: str, config: dict) -> str:
    """Call Google Gemini API."""
    api_key = os.getenv(config["env_key"])
    url = config["api_url"].format(model=config["model"], api_key=api_key)

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


def call_zai(system_prompt: str, user_message: str, config: dict) -> str:
    """Call Z.AI GLM-4.6 API with streaming."""
    api_key = os.getenv(config["env_key"])

    response = requests.post(
        config["api_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        },
        json={
            "model": config["model"],
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


def get_provider_function(provider: str) -> Callable[[str, str], str]:
    """Get the appropriate function for a provider based on its api_type."""
    config = PROVIDERS[provider]
    api_type = config.get("api_type", "openai")

    if api_type == "gemini":
        return lambda system, user: call_gemini(system, user, config)
    elif api_type == "zai":
        return lambda system, user: call_zai(system, user, config)
    else:  # Default to OpenAI-compatible
        return lambda system, user: call_openai_compatible(system, user, config)


# ============================================================
# CHUNKED PROCESSING FOR LARGE TRANSCRIPTS
# ============================================================

def generate_notes_chunked(
    provider: str,
    system_prompt: str, 
    transcript: str,
    video_title: str,
    video_id: str,
    chapters: list = None,
    tpm_limit: int = 12000,
    requested_tokens: int = 0
) -> str:
    """
    Generate notes using map-reduce for large transcripts that exceed TPM limits.
    
    Process:
    1. Split transcript into chunks that fit within TPM limits
    2. Summarize each chunk (MAP phase)
    3. Combine summaries (REDUCE phase)
    4. Generate final notes from combined summary
    """
    config = PROVIDERS[provider]
    provider_fn = get_provider_function(provider)
    
    # Calculate chunk size based on actual token limits
    # Leave room for prompts (~500 tokens) and output (~1000 tokens)
    safe_input_tokens = int(tpm_limit * 0.7)  # Use 70% of limit for safety
    chunk_char_size = safe_input_tokens * 4  # ~4 chars per token
    
    print(f"\n📦 Chunked Processing Mode")
    print(f"   TPM Limit: {tpm_limit:,} | Requested: {requested_tokens:,}")
    print(f"   Chunk size: ~{safe_input_tokens:,} tokens")
    
    # Split transcript into chunks with overlap for context
    chunks = []
    words = transcript.split()
    current_chunk = []
    current_size = 0
    overlap_words = 50  # Keep last 50 words for context overlap
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_char_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last N words for overlap
            current_chunk = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else []
            current_size = sum(len(w) + 1 for w in current_chunk)
        current_chunk.append(word)
        current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    print(f"   Split into {len(chunks)} chunks")
    
    # ========== MAP PHASE: Summarize each chunk ==========
    summaries = []
    
    for i, chunk in enumerate(chunks, 1):
        # Wait between requests to respect TPM (60 seconds per window)
        if i > 1:
            # Calculate wait time based on how many chunks fit in TPM
            chunks_per_minute = max(1, tpm_limit // safe_input_tokens)
            wait_time = max(5, 65 // chunks_per_minute)  # At least 5s, ensure rate limit reset
            print(f"   ⏳ Waiting {wait_time}s for rate limit...")
            time.sleep(wait_time)
        
        print(f"   [{i}/{len(chunks)}] Summarizing chunk...")
        
        chunk_system = """You summarize portions of YouTube video transcripts.
Capture: key points, examples, statistics, quotes, and actionable advice.
Be detailed but concise (200-400 words). Preserve important technical details."""
        
        chunk_prompt = f"""Summarize this portion ({i}/{len(chunks)}) of a YouTube video transcript.

Video: {video_title}

Transcript portion:
---
{chunk}
---

Provide a detailed summary of this section."""
        
        try:
            with ProgressIndicator(f"Processing chunk {i}/{len(chunks)}..."):
                summary = provider_fn(chunk_system, chunk_prompt)
            summaries.append(f"[Part {i}/{len(chunks)}]\n{summary}")
        except TokenLimitExceeded:
            # If still too large, use a shorter chunk
            print(f"   ⚠️  Chunk {i} still too large, using abbreviated version...")
            abbreviated = " ".join(chunk.split()[:500])  # First 500 words only
            with ProgressIndicator(f"Processing abbreviated chunk {i}/{len(chunks)}..."):
                summary = provider_fn(chunk_system, f"Summarize briefly:\n{abbreviated}")
            summaries.append(f"[Part {i}/{len(chunks)} - abbreviated]\n{summary}")
    
    # ========== REDUCE PHASE: Combine summaries ==========
    print(f"\n   🔗 Combining {len(summaries)} summaries...")
    
    # Wait for rate limit reset before combining
    print(f"   ⏳ Waiting 65s for rate limit reset...")
    time.sleep(65)
    
    combined = "\n\n".join(summaries)
    
    # Check if combined summaries need further reduction
    combined_tokens = len(combined) // 4
    if combined_tokens > safe_input_tokens:
        print(f"   📝 Combined summaries too large ({combined_tokens:,} tokens), condensing...")
        
        condense_system = "You combine partial summaries into a unified, coherent summary. Remove redundancies while preserving all unique insights."
        condense_prompt = f"""Combine these partial summaries of "{video_title}" into one coherent summary.
Remove redundancies, maintain flow, preserve all unique insights.

{combined}"""
        
        with ProgressIndicator("Condensing summaries..."):
            combined = provider_fn(condense_system, condense_prompt)
        
        # Wait again before final generation
        print(f"   ⏳ Waiting 65s for rate limit reset...")
        time.sleep(65)
    
    # ========== FINAL PASS: Generate notes from summary ==========
    print(f"\n   📝 Generating final study notes...")
    
    chapters_info = ""
    if chapters:
        chapters_info = "\n\nYouTube Chapters (use these as Key Moments):\n"
        for ch in chapters:
            chapters_info += f"- [{ch['time']}] ({ch['seconds']}s) {ch['title']}\n"
        chapters_info += "\nNote: Chapters are provided. Use these for the KEY MOMENTS section instead of generating your own.\n"
    
    final_message = f"""Create study notes from this YouTube video.

Video Title: {video_title}
Video ID: {video_id}
{chapters_info}

IMPORTANT: The following is a CONDENSED SUMMARY of the full transcript (the original was too long).
Generate comprehensive study notes based on this summary.

Summary of Transcript:
---
{combined}
---"""

    with ProgressIndicator("Generating final notes..."):
        return provider_fn(system_prompt, final_message)


def generate_notes(provider: str, system_prompt: str, transcript: str, video_title: str, video_id: str, chapters: list = None) -> str:
    """Generate study notes using the selected provider with automatic chunking on TPM errors."""
    
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

    # Get the appropriate function based on api_type
    provider_fn = get_provider_function(provider)

    try:
        if config.get("api_type") == "zai":
            # Z.AI streams output directly, no spinner needed
            print()  # Blank line before streamed output
            return provider_fn(system_prompt, user_message)
        else:
            with ProgressIndicator("Generating notes (this may take 1-3 minutes)..."):
                return provider_fn(system_prompt, user_message)
                
    except TokenLimitExceeded as e:
        # Handle by switching to chunked processing
        print(f"\n⚠️  Token limit exceeded!")
        print(f"   Limit: {e.limit:,} | Requested: {e.requested:,}")
        print(f"   Switching to chunked processing mode...")
        
        return generate_notes_chunked(
            provider=provider,
            system_prompt=system_prompt,
            transcript=transcript,
            video_title=video_title,
            video_id=video_id,
            chapters=chapters,
            tpm_limit=e.limit,
            requested_tokens=e.requested
        )


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


def load_system_prompt(prompt_name: str = DEFAULT_PROMPT) -> str:
    """Load system prompt from specified file in prompts folder."""
    path = os.path.join(get_script_dir(), PROMPTS_FOLDER, f"{prompt_name}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"System prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_filename(title: str, max_length: int = 80) -> str:
    """Create a filesystem-safe filename from title."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title)
    safe = re.sub(r"[\s_]+", "_", safe).strip("_.")
    return safe[:max_length] or "untitled"


def find_existing_note(video_id: str, prompt_name: str, model_name: str) -> Optional[str]:
    """Find existing note file for the same video ID, prompt, and model."""
    output_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER)
    if not os.path.exists(output_dir):
        return None

    # Match video ID in content AND both prompt and model name in filename
    video_pattern = f"https://www.youtube.com/watch?v={video_id}"
    for filename in os.listdir(output_dir):
        if filename.endswith(".md") and prompt_name in filename and model_name in filename:
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
    prompt_name: str,
    channel: str = "Unknown",
    duration: str = "Unknown",
    transcript_words: int = 0,
) -> str:
    """Save notes to YouTubeNotes folder. Overwrites if same video+prompt+model exists."""
    output_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Use nickname for shorter filenames
    nickname = PROVIDERS[provider]["nickname"]

    # Check for existing note with same video ID, prompt, and provider
    existing = find_existing_note(video_id, prompt_name, nickname)
    if existing:
        filepath = existing
        print(f"   Overwriting: {os.path.basename(filepath)}")
    else:
        # Use shorter title (40 chars) for compact filenames
        # Format: {video_id}_{title}_{prompt}_{provider}.md
        filename = f"{video_id}_{sanitize_filename(video_title, max_length=40)}_{prompt_name}_{nickname}.md"
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
        f"Prompt: {prompt_name}\n"
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YouTube Study Notes Generator - Convert YouTube videos into structured study notes using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                    # Interactive mode
  python app.py "https://youtube.com/..."          # Direct URL, interactive prompt selection
  python app.py "URL" --prompt study-notes         # Use specific prompt
  python app.py --prompt quick-summary "URL"       # Argument order flexible
        """
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL"
    )
    parser.add_argument(
        "--prompt", "-p",
        help=f"Prompt template to use (default: {DEFAULT_PROMPT}). Available prompts are in the prompts/ folder."
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  📚 YouTube Study Notes Generator")
    print("=" * 60)

    # Get URL
    url = args.url if args.url else input("\n🎬 Enter YouTube URL: ").strip()
    if not url:
        print("❌ No URL provided.")
        sys.exit(1)

    if args.url:
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

        # Select prompt (CLI argument or interactive)
        if args.prompt:
            prompt_name = args.prompt
            available_prompts = get_available_prompts()
            if prompt_name not in available_prompts:
                print(f"\n❌ Prompt '{prompt_name}' not found.")
                print(f"   Available prompts: {', '.join(available_prompts)}")
                sys.exit(1)
            print(f"\n📝 Using prompt: {prompt_name}")
        else:
            prompt_name = select_prompt()
            print(f"\n✅ Selected prompt: {prompt_name}")

        # Select provider with stats
        provider = select_provider_with_stats(transcript_words)
        print(f"\n✅ Selected provider: {PROVIDERS[provider]['name']}")

        print("\n📋 Loading system prompt...")
        system_prompt = load_system_prompt(prompt_name)
        print(f"   Loaded: {PROMPTS_FOLDER}/{prompt_name}.md")

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
            prompt_name=prompt_name,
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
