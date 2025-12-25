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
from notion_client import Client as NotionClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# Load environment variables
load_dotenv(override=True)

# Import provider configurations from external file
from providers import PROVIDERS

# Import shared transcript utilities
from transcript_utils import (
    extract_video_id,
    get_video_metadata as get_video_info,
    download_transcript as fetch_transcript,
    get_script_dir,
    load_cached_transcript,
    save_transcript_to_cache,
    estimate_tokens,
    format_duration,
)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class TokenLimitExceeded(Exception):
    """Raised when a request exceeds TPM (tokens per minute) limits."""
    def __init__(self, limit: int, requested: int, message: str):
        self.limit = limit
        self.requested = requested
        super().__init__(message)


class RestartException(Exception):
    """Raised when user wants to restart from beginning."""
    pass

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FOLDER = "YouTubeNotes"
PROMPTS_FOLDER = "prompts"
DEFAULT_PROMPT = "youtube-summary"
REQUEST_TIMEOUT = 300  # 5 minutes
MAX_OUTPUT_TOKENS = 8192


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
        if config.get("env_key") and os.getenv(config["env_key"])
    }


def select_provider_with_stats(word_count: int) -> str:
    """Display provider menu with token usage stats and return selected provider key."""
    estimated_tokens = estimate_tokens(word_count)

    print("\n" + "=" * 60)
    print("  🤖 Select AI Provider")
    print("=" * 60)
    print(f"\n  📊 Transcript: ~{word_count:,} words (~{estimated_tokens:,} tokens)")
    print("-" * 60)

    # Filter out providers that don't need API keys (like Cursor)
    api_providers = {
        key: config
        for key, config in PROVIDERS.items()
        if config.get("env_key")  # Only include providers that need API keys
    }

    provider_keys = list(api_providers.keys())
    available = get_available_providers()

    # Find best recommendation (available provider with lowest usage that fits and doesn't exceed rate limit)
    best_provider = None
    best_usage = float('inf')
    for key, config in api_providers.items():
        if key in available:
            usage = (estimated_tokens / config["context_tokens"]) * 100
            rate_limit = config.get("rate_limit_tpm")
            exceeds_rate_limit = rate_limit and estimated_tokens > rate_limit
            if usage < 80 and usage < best_usage and not exceeds_rate_limit:
                best_usage = usage
                best_provider = key

    for idx, (key, config) in enumerate(api_providers.items(), 1):
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
    print("\nOptions:")
    print("  [1-{}]: Select option".format(len(api_providers)))
    print("  [r]: Restart from beginning")

    if not available:
        print("\n❌ No API keys configured!")
        print("   Add at least one key to your .env file:")
        print("   - GEMINI_API_KEY  (free: https://aistudio.google.com)")
        print("   - GROQ_API_KEY    (free: https://console.groq.com)")
        print("   - ZAI_API_KEY     (paid: https://z.ai)")
        sys.exit(1)

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(api_providers)}): ").strip().lower()

            # Handle restart option
            if choice in ['r', 'restart']:
                raise RestartException()

            idx = int(choice) - 1
            if 0 <= idx < len(provider_keys):
                selected = provider_keys[idx]
                if selected in available:
                    return selected
                print(f"   ❌ {api_providers[selected]['name']} not configured.")
            else:
                print("   Invalid choice.")
        except ValueError:
            print("   Please enter a number or 'r' to restart.")
        except RestartException:
            # Re-raise restart exception
            raise


# ============================================================
# PROMPT SELECTION
# ============================================================

def get_available_prompts() -> List[str]:
    """Scan prompts folder and return list of available prompt names.
    
    Returns prompts with DEFAULT_PROMPT first, then rest sorted alphabetically.
    """
    prompts_dir = os.path.join(get_script_dir(), PROMPTS_FOLDER)
    if not os.path.exists(prompts_dir):
        return []
    
    prompts = []
    for f in sorted(os.listdir(prompts_dir)):
        if f.endswith(".md"):
            prompts.append(f[:-3])  # Remove .md extension
    
    # Put default prompt first, then rest alphabetically
    if DEFAULT_PROMPT in prompts:
        prompts.remove(DEFAULT_PROMPT)
        prompts = [DEFAULT_PROMPT] + prompts
    
    return prompts


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
        print(f"\n  {idx}. {prompt}{default_tag}")
    
    print("\n" + "-" * 60)
    print("\nOptions:")
    print("  [1-{}]: Select option".format(len(prompts)))
    print("  [r]: Restart from beginning")
    
    # Default is always first (index 1)
    default_idx = 1
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(prompts)}) or press Enter for default [{default_idx}]: ").strip().lower()
            
            if not choice:
                return prompts[default_idx - 1]
            
            # Handle restart option
            if choice in ['r', 'restart']:
                raise RestartException()
            
            idx = int(choice) - 1
            if 0 <= idx < len(prompts):
                return prompts[idx]
            print("   Invalid choice.")
        except ValueError:
            print("   Please enter a number or 'r' to restart.")
        except RestartException:
            # Re-raise restart exception
            raise


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
            # Handle both raw yt-dlp format and formatted chapters
            # Raw yt-dlp: {'start_time': float, 'end_time': float, 'title': str}
            # Formatted: {'time': str, 'seconds': int, 'title': str}
            if 'start_time' in ch:
                # Raw yt-dlp format - convert to expected format
                start_time = ch['start_time']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                time_str = f"{minutes}:{seconds:02d}"
                title = ch.get('title', 'Unknown Chapter')
                chapters_info += f"- [{time_str}] ({int(start_time)}s) {title}\n"
            elif 'time' in ch and 'seconds' in ch:
                # Already formatted
                chapters_info += f"- [{ch['time']}] ({ch['seconds']}s) {ch['title']}\n"
            else:
                # Fallback for unexpected formats
                title = ch.get('title', 'Unknown Chapter')
                chapters_info += f"- {title}\n"
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
            # Handle both raw yt-dlp format and formatted chapters
            # Raw yt-dlp: {'start_time': float, 'end_time': float, 'title': str}
            # Formatted: {'time': str, 'seconds': int, 'title': str}
            if 'start_time' in ch:
                # Raw yt-dlp format - convert to expected format
                start_time = ch['start_time']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                time_str = f"{minutes}:{seconds:02d}"
                title = ch.get('title', 'Unknown Chapter')
                chapters_info += f"- [{time_str}] ({int(start_time)}s) {title}\n"
            elif 'time' in ch and 'seconds' in ch:
                # Already formatted
                chapters_info += f"- [{ch['time']}] ({ch['seconds']}s) {ch['title']}\n"
            else:
                # Fallback for unexpected formats
                title = ch.get('title', 'Unknown Chapter')
                chapters_info += f"- {title}\n"
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
# FILE OPERATIONS
# ============================================================


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
# NOTION INTEGRATION
# ============================================================

def markdown_to_notion_blocks(markdown: str) -> list:
    """
    Convert markdown text to Notion block format.
    
    Handles:
    - Headings (# ## ###)
    - Bullet points (- )
    - Numbered lists (1. 2. etc)
    - Bold text (**text**)
    - Regular paragraphs
    
    Note: Notion API limits to 100 blocks per request.
    Note: Notion has a 2000 character limit per text block.
    """
    NOTION_TEXT_LIMIT = 2000
    blocks = []
    lines = markdown.split('\n')
    
    def split_long_text(text: str, limit: int = NOTION_TEXT_LIMIT) -> list:
        """Split text into chunks that fit within Notion's character limit."""
        if len(text) <= limit:
            return [text]
        
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            
            # Find a good break point (space, period, comma) near the limit
            break_point = limit
            for sep in ['. ', ', ', ' ', '.', ',']:
                idx = text.rfind(sep, 0, limit)
                if idx > limit // 2:  # Don't break too early
                    break_point = idx + len(sep)
                    break
            
            chunks.append(text[:break_point].strip())
            text = text[break_point:].strip()
        
        return chunks
    
    def parse_rich_text(text: str) -> list:
        """Parse inline formatting: bold (**text**) and links ([text](url))."""
        rich_text = []
        
        # Helper to parse links within a text segment
        def parse_links(segment: str, is_bold: bool = False) -> list:
            """Parse markdown links in a text segment, including double-bracket format [[text]](url)."""
            result = []
            # Pattern handles both [text](url) and [[text]](url) formats
            link_pattern = r'\[+([^\]]+)\]+\(([^)]+)\)'
            link_last_end = 0
            
            for link_match in re.finditer(link_pattern, segment):
                # Add text before the link
                if link_match.start() > link_last_end:
                    plain = segment[link_last_end:link_match.start()]
                    if plain:
                        item = {"type": "text", "text": {"content": plain}}
                        if is_bold:
                            item["annotations"] = {"bold": True}
                        result.append(item)
                
                # Add the link
                link_text = link_match.group(1)
                link_url = link_match.group(2)
                item = {"type": "text", "text": {"content": link_text, "link": {"url": link_url}}}
                if is_bold:
                    item["annotations"] = {"bold": True}
                result.append(item)
                link_last_end = link_match.end()
            
            # Add remaining text after last link
            if link_last_end < len(segment):
                remaining = segment[link_last_end:]
                if remaining:
                    item = {"type": "text", "text": {"content": remaining}}
                    if is_bold:
                        item["annotations"] = {"bold": True}
                    result.append(item)
            
            # If no links found, return the whole segment
            if not result:
                item = {"type": "text", "text": {"content": segment}}
                if is_bold:
                    item["annotations"] = {"bold": True}
                result.append(item)
            
            return result
        
        # First pass: find bold sections
        bold_pattern = r'\*\*(.+?)\*\*'
        last_end = 0
        
        for match in re.finditer(bold_pattern, text):
            # Add text before the bold (may contain links)
            if match.start() > last_end:
                plain = text[last_end:match.start()]
                if plain:
                    rich_text.extend(parse_links(plain, is_bold=False))
            
            # Add bold text (may contain links)
            bold_content = match.group(1)
            rich_text.extend(parse_links(bold_content, is_bold=True))
            last_end = match.end()
        
        # Add remaining text after last bold (may contain links)
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining:
                rich_text.extend(parse_links(remaining, is_bold=False))
        
        # If nothing was parsed, return plain text
        if not rich_text:
            rich_text = [{"type": "text", "text": {"content": text}}]
        
        return rich_text
    
    def add_block(block_type: str, content: str, key: str):
        """Add block(s), splitting if content exceeds Notion's limit."""
        chunks = split_long_text(content)
        for chunk in chunks:
            blocks.append({
                "object": "block",
                "type": block_type,
                key: {"rich_text": parse_rich_text(chunk)}
            })
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Heading 1
        if stripped.startswith('# '):
            # Truncate headings if too long (shouldn't happen often)
            heading_text = stripped[2:][:NOTION_TEXT_LIMIT]
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": parse_rich_text(heading_text)}
            })
        # Heading 2
        elif stripped.startswith('## '):
            heading_text = stripped[3:][:NOTION_TEXT_LIMIT]
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": parse_rich_text(heading_text)}
            })
        # Heading 3
        elif stripped.startswith('### '):
            heading_text = stripped[4:][:NOTION_TEXT_LIMIT]
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": parse_rich_text(heading_text)}
            })
        # Bullet point - split if too long
        elif stripped.startswith('- ') or stripped.startswith('* '):
            add_block("bulleted_list_item", stripped[2:], "bulleted_list_item")
        # Numbered list (matches "1. ", "2. ", etc.) - split if too long
        elif re.match(r'^\d+\.\s', stripped):
            content = re.sub(r'^\d+\.\s', '', stripped)
            add_block("numbered_list_item", content, "numbered_list_item")
        # Regular paragraph - split if too long
        else:
            add_block("paragraph", stripped, "paragraph")
    
    return blocks


def extract_title_from_notes(notes: str) -> str:
    """Extract the title from notes (first ### heading before Summary)."""
    lines = notes.split('\n')
    for line in lines:
        stripped = line.strip()
        # Look for ### heading that's not "Summary" or numbered sections
        if stripped.startswith('### ') and not stripped.startswith('### Summary') and not re.match(r'### \d+\.', stripped):
            return stripped[4:].strip()
    return ""


def extract_tags_from_notes(notes: str) -> list:
    """Extract AI-generated tags from notes. Handles multiple LLM formats."""
    patterns = [
        r'\*\*Tags?:\*\*\s*(.+?)(?:\n|$)',      # **Tags:** or **Tag:** (bold with colon)
        r'\*\*Tags?\*\*:\s*(.+?)(?:\n|$)',       # **Tags**: or **Tags**: (bold colon separated)
        r'Tags?:\s*(.+?)(?:\n|$)',               # Tags: or Tag: (no markdown)
        r'🏷\s*(.+?)(?:\n|$)',                   # Emoji tag format
    ]

    for pattern in patterns:
        match = re.search(pattern, notes, re.IGNORECASE)
        if match:
            tags_line = match.group(1)
            # Split by comma, pipe, or semicolon
            tags = re.split(r'[,|;]', tags_line)
            # Clean each tag
            tags = [tag.strip().strip('`\'"') for tag in tags]
            # Remove any empty tags and limit length
            tags = [tag for tag in tags if tag and len(tag) > 1]
            return tags[:10]

    return []


def generate_tags_from_content(notes: str, video_title: str, channel: str, prompt_name: str) -> list:
    """
    Get tags for the content. First tries to extract AI-generated tags,
    then falls back to auto-generation from metadata.
    """
    # First, try to extract AI-generated tags from notes
    tags = extract_tags_from_notes(notes)
    if tags:
        return tags
    
    # Fallback: auto-generate tags from metadata
    auto_tags = set()
    
    # 1. Add prompt type as a tag (convert "study-notes" to "StudyNotes")
    prompt_tag = prompt_name.replace("-", " ").title().replace(" ", "")
    auto_tags.add(prompt_tag)
    
    # 2. Add channel name (cleaned - remove special chars, limit length)
    channel_clean = re.sub(r'[^\w\s]', '', channel).strip()
    if channel_clean:
        channel_tag = channel_clean.replace(" ", "")[:25]
        auto_tags.add(channel_tag)
    
    # 3. Extract key words from video title (capitalized words, likely topics)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'is', 'it', 'this', 'that', 'how', 'what',
                  'why', 'when', 'where', 'who', 'i', 'you', 'we', 'my', 'your', 'our'}
    
    title_words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', video_title)
    for word in title_words:
        if word.lower() not in stop_words and len(word) > 2:
            auto_tags.add(word.replace(" ", ""))
            if len(auto_tags) >= 10:
                break
    
    return list(auto_tags)[:10]


def strip_title_and_tags_from_notes(notes: str) -> str:
    """Remove title heading and tags line from notes for Notion body (these go in page properties)."""
    lines = notes.split('\n')
    result_lines = []
    skip_next_empty = False
    found_title = False

    for line in lines:
        stripped = line.strip()

        # Skip the title line (first ### that's not Summary or numbered)
        # Handles both: "### Title" and "### **Title**" formats
        if not found_title and stripped.startswith('### ') and not stripped.startswith('### Summary') and not re.match(r'### \d+\.', stripped):
            found_title = True
            skip_next_empty = True
            continue

        # Skip the Tags line (handle various formats)
        if re.match(r'\*\*Tags?\*:\*', stripped) or re.match(r'Tags?:', stripped):
            skip_next_empty = True
            continue

        # Skip empty line after removed content
        if skip_next_empty and not stripped:
            skip_next_empty = False
            continue

        skip_next_empty = False
        result_lines.append(line)

    return '\n'.join(result_lines)


def publish_to_notion(
    notes: str,
    video_title: str,
    video_id: str,
    channel: str,
    duration: str,
    provider: str,
    prompt_name: str,
) -> str:
    """
    Publish notes to Notion database.
    
    Returns the Notion page URL.
    Handles Notion's 100-block limit by batching.
    Auto-generates tags from content (not AI-generated).
    """
    notion_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not notion_key or not database_id:
        raise ValueError("NOTION_API_KEY and NOTION_DATABASE_ID must be set in .env")
    
    notion = NotionClient(auth=notion_key)
    
    # Auto-generate tags from content
    tags = generate_tags_from_content(notes, video_title, channel, prompt_name)

    # Always use YouTube video title (never extract from LLM-generated notes)
    # This ensures consistency across all LLM providers
    page_title = video_title
    
    # Strip title from body content (title goes in page property)
    body_content = strip_title_and_tags_from_notes(notes)
    
    # Convert markdown to Notion blocks (without title/hashtags)
    all_blocks = markdown_to_notion_blocks(body_content)
    
    # Notion API limits: 100 blocks per request
    # Create page with first batch of blocks
    first_batch = all_blocks[:100]
    remaining_blocks = all_blocks[100:]
    
    # Build properties dict
    properties = {
        "Name": {"title": [{"text": {"content": page_title[:100]}}]},  # Title limit
        "YouTube URL": {"url": f"https://www.youtube.com/watch?v={video_id}"},
        "Channel": {"rich_text": [{"text": {"content": channel[:200]}}]},
        "Duration": {"rich_text": [{"text": {"content": duration}}]},
        "Provider": {"select": {"name": PROVIDERS[provider]["name"]}},
        "Prompt": {"select": {"name": prompt_name}},
        "Date Added": {"date": {"start": datetime.now().strftime("%Y-%m-%d")}},
    }
    
    # Add auto-generated Tags
    if tags:
        properties["Tags"] = {"multi_select": [{"name": tag} for tag in tags]}
    
    # Create the page with properties and initial content
    page = notion.pages.create(
        parent={"database_id": database_id},
        properties=properties,
        children=first_batch,
    )
    
    page_id = page["id"]
    
    # Append remaining blocks in batches of 100
    while remaining_blocks:
        batch = remaining_blocks[:100]
        remaining_blocks = remaining_blocks[100:]
        notion.blocks.children.append(block_id=page_id, children=batch)
    
    return page["url"]


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


def generate_notes_from_transcript(video_id: str, transcript: str, metadata: dict, prompt_name: str = None) -> tuple:
    """
    Generate notes from a pre-downloaded transcript.
    This function is called by main.py when transcript is already downloaded.

    Args:
        video_id: YouTube video ID
        transcript: Plain text transcript
        metadata: Video metadata dict (title, channel, duration, chapters)
        prompt_name: Optional prompt template name (if None, will prompt user)

    Returns:
        Tuple of (notes: str, provider: str, prompt_name: str)
    """
    # Prompt selection loop (allows going back from provider menu)
    provider_selected = False
    final_provider = None
    final_prompt_name = None

    while not provider_selected:
        try:
            # Select prompt if not provided
            if not prompt_name:
                prompt_name = select_prompt()
            else:
                available_prompts = get_available_prompts()
                if prompt_name not in available_prompts:
                    print(f"\n❌ Prompt '{prompt_name}' not found.")
                    print(f"   Available prompts: {', '.join(available_prompts)}")
                    sys.exit(1)

            # Select provider
            transcript_words = len(transcript.split())
            provider = select_provider_with_stats(transcript_words)

            # If we get here, both selections succeeded
            final_provider = provider
            final_prompt_name = prompt_name
            provider_selected = True

        except RestartException:
            # Restart prompt selection
            prompt_name = None
            continue

    # Load system prompt
    system_prompt = load_system_prompt(final_prompt_name)

    # Generate notes
    notes = generate_notes(
        final_provider,
        system_prompt,
        transcript,
        metadata.get('title', video_id),
        video_id,
        chapters=metadata.get('chapters', []),
    )

    return notes, final_provider, final_prompt_name


def save_notes(video_id: str, notes: str, metadata: dict, prompt_name: str = None, provider: str = None) -> str:
    """
    Save generated notes to file.

    Args:
        video_id: YouTube video ID
        notes: Generated notes content
        metadata: Video metadata dict
        prompt_name: Prompt template used
        provider: AI provider used

    Returns:
        Path to saved file
    """
    output_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    safe_title = re.sub(r'[^\w\s-]', '', metadata.get('title', video_id)).strip()
    safe_title = re.sub(r'[-\s]+', '-', safe_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prompt_name:
        filename = f"{timestamp}_{safe_title}_{prompt_name}.md"
    else:
        filename = f"{timestamp}_{safe_title}.md"

    filepath = os.path.join(output_dir, filename)

    # Add metadata header
    header_lines = []
    header_lines.append(f"# {metadata.get('title', 'Unknown')}")
    header_lines.append("")
    header_lines.append(f"**Source:** https://www.youtube.com/watch?v={video_id}")
    header_lines.append(f"**Channel:** {metadata.get('channel', 'Unknown')}")

    if metadata.get('duration'):
        duration = metadata['duration']
        if isinstance(duration, int):
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            secs = duration % 60
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {secs}s"
            else:
                duration_str = f"{secs}s"
            header_lines.append(f"**Duration:** {duration_str}")

    if provider:
        header_lines.append(f"**AI Provider:** {PROVIDERS[provider]['name']}")

    if prompt_name:
        header_lines.append(f"**Prompt Template:** {prompt_name}")

    header_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header_lines.append("")
    header_lines.append("---")
    header_lines.append("")

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(notes)

    return filepath


def main():
    """Main entry point."""
    args = parse_args()
    
    # Outer loop for restart capability
    while True:
        try:
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

                # Prompt selection loop (allows going back from provider menu)
                provider_selected = False
                while not provider_selected:
                    try:
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
                        
                        # Provider selection loop
                        while True:
                            try:
                                # Select provider with stats
                                provider = select_provider_with_stats(transcript_words)
                                print(f"\n✅ Selected provider: {PROVIDERS[provider]['name']}")
                                provider_selected = True
                                break  # Exit provider selection loop on success
                            except RestartException:
                                raise  # Re-raise to outer loop
                        
                    except RestartException:
                        raise  # Re-raise to outer loop

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

                # Publish to Notion if configured
                notion_url = None
                if os.getenv("NOTION_API_KEY") and os.getenv("NOTION_DATABASE_ID"):
                    print("📤 Publishing to Notion...")
                    try:
                        notion_url = publish_to_notion(
                            notes=notes,
                            video_title=video_info["title"],
                            video_id=video_id,
                            channel=video_info["channel"],
                            duration=video_info["duration"],
                            provider=provider,
                            prompt_name=prompt_name,
                        )
                        print(f"   ✅ Published to Notion")
                    except Exception as e:
                        print(f"   ⚠️  Notion publish failed: {e}")

                print("\n" + "=" * 50)
                print("✅ Success!")
                print(f"   📁 Local: {filepath}")
                if notion_url:
                    print(f"   🔗 Notion: {notion_url}")
                print("=" * 50 + "\n")
                
                # Successfully completed, exit outer loop
                break

            except TranscriptsDisabled:
                print("\n❌ Transcripts are disabled for this video.")
                sys.exit(1)
            except NoTranscriptFound:
                print("\n❌ No transcript found for this video.")
                sys.exit(1)
            except KeyboardInterrupt:
                print("\n\n⚠️  Cancelled by user.")
                sys.exit(130)
            except RestartException:
                # Restart from beginning (will continue outer loop)
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
                sys.exit(1)
        
        except RestartException:
            # Restart from beginning (will continue outer loop)
            continue


if __name__ == "__main__":
    main()
