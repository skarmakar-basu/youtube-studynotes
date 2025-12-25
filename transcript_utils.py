#!/usr/bin/env python3
"""
Shared transcript utilities for YouTube Study Notes Generator
Provides common transcript downloading and caching functions
"""

import os
import re
import sys
import requests
from typing import Optional, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FOLDER = "YouTubeNotes"
TRANSCRIPTS_FOLDER = "transcripts"


def get_script_dir() -> str:
    """Get directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


# ============================================================
# VIDEO ID EXTRACTION
# ============================================================

def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
        r'youtube\.com/embed/([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ============================================================
# VIDEO METADATA
# ============================================================

def get_video_metadata(video_id: str) -> Dict:
    """Get video metadata using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'channel': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'chapters': info.get('chapters', [])
            }
    except Exception as e:
        return {'title': 'Unknown', 'channel': 'Unknown', 'duration': 0, 'chapters': []}


# ============================================================
# TRANSCRIPT DOWNLOADING
# ============================================================

def download_subtitles_with_ytdlp(video_id: str) -> str:
    """
    Download subtitles in SRT format using yt-dlp.

    Args:
        video_id: YouTube video ID

    Returns:
        Path to downloaded SRT file

    Raises:
        Exception: If download fails for any reason
    """
    cache_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER, TRANSCRIPTS_FOLDER)
    os.makedirs(cache_dir, exist_ok=True)
    srt_path = os.path.join(cache_dir, f"{video_id}.srt")

    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        # Extract video info to get subtitle URLs
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Get available subtitles (prefer manual over automatic)
            subtitles = info.get("subtitles", {})
            automatic_captions = info.get("automatic_captions", {})

            # Combine both, prioritizing manual subtitles
            all_subtitles = {}
            all_subtitles.update(subtitles)
            all_subtitles.update(automatic_captions)

            if not all_subtitles:
                raise Exception("No subtitles available for this video")

            # Find English subtitle URL (prefer en, en-US, en-GB, then any)
            subtitle_url = None
            subtitle_lang = None

            for lang_code in ["en", "en-US", "en-GB"]:
                if lang_code in all_subtitles:
                    # Find SRT format if available
                    for sub_info in all_subtitles[lang_code]:
                        if sub_info.get("ext") == "srt":
                            subtitle_url = sub_info.get("url")
                            subtitle_lang = lang_code
                            break
                    if subtitle_url:
                        break

            # If no English found, try any language
            if not subtitle_url:
                for lang_code, lang_subs in all_subtitles.items():
                    for sub_info in lang_subs:
                        if sub_info.get("ext") == "srt":
                            subtitle_url = sub_info.get("url")
                            subtitle_lang = lang_code
                            break
                    if subtitle_url:
                        break

            if not subtitle_url:
                raise Exception("No SRT format subtitles available")

            # Download subtitle file directly
            response = requests.get(subtitle_url, timeout=30)
            response.raise_for_status()

            # Save to file
            with open(srt_path, "wb") as f:
                f.write(response.content)

            return srt_path

    except requests.RequestException as e:
        # Clean up partial files if any
        if os.path.exists(srt_path):
            try:
                os.remove(srt_path)
            except Exception:
                pass
        raise Exception(f"Failed to download subtitle file: {str(e)}")
    except yt_dlp.utils.DownloadError as e:
        # Clean up partial files if any
        if os.path.exists(srt_path):
            try:
                os.remove(srt_path)
            except Exception:
                pass
        raise Exception(f"yt-dlp extraction error: {str(e)}")
    except Exception as e:
        # Clean up partial files if any
        if os.path.exists(srt_path):
            try:
                os.remove(srt_path)
            except Exception:
                pass
        # Re-raise with more context
        error_msg = str(e)
        if "yt-dlp" not in error_msg.lower() and "subtitle" not in error_msg.lower():
            error_msg = f"yt-dlp subtitle download failed: {error_msg}"
        raise Exception(error_msg)


def parse_srt_to_text(srt_path: str) -> str:
    """
    Parse SRT file and extract plain text (remove timestamps and formatting).

    Args:
        srt_path: Path to SRT file

    Returns:
        Plain text string with all subtitle text joined
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_lines = []
    skip_next = False

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip sequence numbers (numeric only)
        if line.isdigit():
            skip_next = True
            continue

        # Skip timestamp lines (contain -->)
        if "-->" in line:
            skip_next = False
            continue

        # Collect text lines
        if not skip_next and line:
            text_lines.append(line)

    # Join all text with spaces and normalize whitespace
    text = " ".join(text_lines)
    return re.sub(r"\s+", " ", text).strip()


def download_transcript_with_api(video_id: str) -> str:
    """Download transcript using youtube-transcript-api (FALLBACK METHOD)"""
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

    # Convert to list format if needed
    if hasattr(data, "snippets"):
        transcript_list = [
            {"start": s.start, "duration": s.duration, "text": s.text}
            for s in data.snippets
        ]
    elif isinstance(data, list):
        transcript_list = data
    else:
        # Convert to list format
        transcript_list = []
        for entry in data:
            if isinstance(entry, dict):
                transcript_list.append(entry)
            else:
                transcript_list.append({
                    "start": getattr(entry, "start", 0),
                    "duration": getattr(entry, "duration", 0),
                    "text": getattr(entry, "text", str(entry))
                })

    # Extract plain text
    if isinstance(transcript_list, list):
        text = " ".join(entry.get("text", "") if isinstance(entry, dict) else getattr(entry, "text", str(entry)) for entry in transcript_list)
    else:
        text = " ".join(str(entry) for entry in transcript_list)

    plain_text = re.sub(r"\s+", " ", text).strip()
    return plain_text


def convert_transcript_to_srt(transcript_data: list, video_id: str) -> str:
    """
    Convert youtube-transcript-api format to SRT format.

    Args:
        transcript_data: List of dicts with 'start', 'duration', 'text' keys
        video_id: YouTube video ID (for reference)

    Returns:
        SRT content as string
    """
    srt_lines = []

    for idx, entry in enumerate(transcript_data, 1):
        # Get values, handling different data structures
        if isinstance(entry, dict):
            start = entry.get("start", 0)
            duration = entry.get("duration", 0)
            text = entry.get("text", "")
        elif hasattr(entry, "start") and hasattr(entry, "duration") and hasattr(entry, "text"):
            start = entry.start
            duration = entry.duration
            text = entry.text
        else:
            continue

        if not text:
            continue

        # Calculate end time
        end = start + duration

        # Convert seconds to HH:MM:SS,mmm format
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        start_time = format_timestamp(start)
        end_time = format_timestamp(end)

        # Format SRT entry
        srt_lines.append(str(idx))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

    return "\n".join(srt_lines)


def save_srt_file(video_id: str, srt_content: str) -> str:
    """
    Save SRT content to file.

    Args:
        video_id: YouTube video ID
        srt_content: SRT formatted content string

    Returns:
        Path to saved SRT file
    """
    cache_dir = os.path.join(get_script_dir(), OUTPUT_FOLDER, TRANSCRIPTS_FOLDER)
    os.makedirs(cache_dir, exist_ok=True)
    srt_path = os.path.join(cache_dir, f"{video_id}.srt")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return srt_path


# ============================================================
# TRANSCRIPT CACHING
# ============================================================

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
# UNIFIED TRANSCRIPT DOWNLOAD
# ============================================================

def download_transcript(video_id: str, show_output: bool = True) -> Optional[str]:
    """
    Download transcript using yt-dlp (PRIMARY) with youtube-transcript-api (FALLBACK).
    Implements caching to avoid re-downloading.

    Args:
        video_id: YouTube video ID
        show_output: Whether to show download progress messages

    Returns:
        Plain text transcript string, or None if both methods fail
    """
    # Check cache first
    cached_text = load_cached_transcript(video_id)
    if cached_text:
        if show_output:
            print("   Loaded from cache")
        return cached_text

    # PRIMARY METHOD: Try yt-dlp first
    try:
        srt_path = download_subtitles_with_ytdlp(video_id)
        plain_text = parse_srt_to_text(srt_path)

        # Save plain text to cache
        save_transcript_to_cache(video_id, plain_text)

        if show_output:
            print(f"   Downloaded with yt-dlp")
        return plain_text

    except Exception as ytdlp_error:
        # FALLBACK METHOD: Use youtube-transcript-api
        try:
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

            # Convert to list format if needed
            if hasattr(data, "snippets"):
                transcript_list = [
                    {"start": s.start, "duration": s.duration, "text": s.text}
                    for s in data.snippets
                ]
            elif isinstance(data, list):
                transcript_list = data
            else:
                # Convert to list format
                transcript_list = []
                for entry in data:
                    if isinstance(entry, dict):
                        transcript_list.append(entry)
                    else:
                        transcript_list.append({
                            "start": getattr(entry, "start", 0),
                            "duration": getattr(entry, "duration", 0),
                            "text": getattr(entry, "text", str(entry))
                        })

            # Convert to SRT format and save
            srt_content = convert_transcript_to_srt(transcript_list, video_id)
            save_srt_file(video_id, srt_content)

            # Extract plain text
            if isinstance(transcript_list, list):
                text = " ".join(entry.get("text", "") if isinstance(entry, dict) else getattr(entry, "text", str(entry)) for entry in transcript_list)
            else:
                text = " ".join(str(entry) for entry in transcript_list)

            plain_text = re.sub(r"\s+", " ", text).strip()

            # Save plain text to cache
            save_transcript_to_cache(video_id, plain_text)

            if show_output:
                print(f"   Downloaded with youtube-transcript-api (yt-dlp failed)")
            return plain_text

        except (NoTranscriptFound, TranscriptsDisabled) as e:
            if show_output:
                print(f"❌ Error fetching transcript: {e}")
            return None
        except Exception as api_error:
            if show_output:
                print(f"❌ Both methods failed:")
                print(f"   yt-dlp: {str(ytdlp_error)}")
                print(f"   youtube-transcript-api: {str(api_error)}")
            return None


# ============================================================
# TRANSCRIPT INFO
# ============================================================

def estimate_tokens(word_count: int) -> int:
    """Estimate token count from word count (roughly 1.3 tokens per word for English)."""
    return int(word_count * 1.3)


def get_transcript_info(transcript: str, metadata: Dict = None) -> Dict:
    """
    Get information about a transcript.

    Args:
        transcript: Plain text transcript
        metadata: Optional video metadata dict

    Returns:
        Dict with transcript statistics
    """
    word_count = len(transcript.split())
    char_count = len(transcript)
    token_estimate = estimate_tokens(word_count)

    info = {
        'word_count': word_count,
        'char_count': char_count,
        'token_estimate': token_estimate,
    }

    # Add metadata if provided
    if metadata:
        info['title'] = metadata.get('title', 'Unknown')
        info['channel'] = metadata.get('channel', 'Unknown')
        info['duration'] = metadata.get('duration', 0)

    return info


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS"""
    if seconds == 0:
        return "Unknown"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def display_transcript_info(info: Dict):
    """Display transcript information in a formatted way"""
    print("\n" + "=" * 60)
    print("  ✓ Transcript Downloaded Successfully")
    print("=" * 60)

    if 'title' in info:
        print(f"\n  📹 Video: {info['title']}")
        print(f"  📺 Channel: {info['channel']}")

        if info['duration'] > 0:
            duration_str = format_duration(info['duration'])
            print(f"  ⏱️  Duration: {duration_str}")

    print(f"\n  📊 Transcript Statistics:")
    print(f"     Words: {info['word_count']:,}")
    print(f"     Characters: {info['char_count']:,}")
    print(f"     Est. Tokens: {info['token_estimate']:,}")
    print("=" * 60)
