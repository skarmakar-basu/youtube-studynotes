#!/usr/bin/env python3
"""
YouTube Study Notes - Cursor Workflow
Downloads transcript and prepares it for Cursor's built-in LLM processing
"""

import os
import sys
import re
import json
import argparse
import requests
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp

def extract_video_id(url):
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

def get_video_metadata(video_id):
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

def parse_srt_to_text(srt_path):
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


def download_subtitles_with_ytdlp(video_id):
    """
    Download subtitles in SRT format using yt-dlp (PRIMARY METHOD).
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Plain text transcript string
        
    Raises:
        Exception: If download fails for any reason
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create transcripts directory
    transcript_dir = Path("YouTubeNotes/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    srt_path = transcript_dir / f"{video_id}.srt"
    
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
            
            # Parse SRT to plain text
            plain_text = parse_srt_to_text(str(srt_path))
            return plain_text
            
    except requests.RequestException as e:
        # Clean up partial files if any
        if srt_path.exists():
            try:
                srt_path.unlink()
            except Exception:
                pass
        raise Exception(f"Failed to download subtitle file: {str(e)}")
    except yt_dlp.utils.DownloadError as e:
        # Clean up partial files if any
        if srt_path.exists():
            try:
                srt_path.unlink()
            except Exception:
                pass
        raise Exception(f"yt-dlp extraction error: {str(e)}")
    except Exception as e:
        # Clean up partial files if any
        if srt_path.exists():
            try:
                srt_path.unlink()
            except Exception:
                pass
        # Re-raise with more context
        error_msg = str(e)
        if "yt-dlp" not in error_msg.lower() and "subtitle" not in error_msg.lower():
            error_msg = f"yt-dlp subtitle download failed: {error_msg}"
        raise Exception(error_msg)


def download_transcript_with_api(video_id):
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


def download_transcript(video_id):
    """
    Download transcript using yt-dlp (PRIMARY) with youtube-transcript-api (FALLBACK).
    
    Returns:
        Plain text transcript string, or None if both methods fail
    """
    # PRIMARY METHOD: Try yt-dlp first
    try:
        transcript = download_subtitles_with_ytdlp(video_id)
        print(f"   Downloaded with yt-dlp")
        return transcript
    except Exception as ytdlp_error:
        # FALLBACK METHOD: Use youtube-transcript-api
        try:
            transcript = download_transcript_with_api(video_id)
            print(f"   Downloaded with youtube-transcript-api (yt-dlp failed)")
            return transcript
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            print(f"❌ Error fetching transcript: {e}")
            return None
        except Exception as api_error:
            print(f"❌ Both methods failed:")
            print(f"   yt-dlp: {str(ytdlp_error)}")
            print(f"   youtube-transcript-api: {str(api_error)}")
            return None

def prepare_for_cursor(url):
    """Main workflow: Download transcript and prepare for Cursor"""
    
    print("🎬 YouTube Study Notes - Cursor Workflow")
    print("=" * 60)
    
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return
    
    print(f"📹 Video ID: {video_id}")
    
    # Get metadata
    print("📊 Fetching video metadata...")
    metadata = get_video_metadata(video_id)
    print(f"📝 Title: {metadata['title']}")
    print(f"👤 Channel: {metadata['channel']}")
    
    # Download transcript
    print("📥 Downloading transcript...")
    transcript = download_transcript(video_id)
    
    if not transcript:
        print("❌ Could not download transcript")
        return
    
    word_count = len(transcript.split())
    print(f"✅ Transcript downloaded: ~{word_count:,} words")
    
    # Save transcript
    transcript_dir = Path("YouTubeNotes/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_path = transcript_dir / f"{video_id}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Add transcript file path to CURSOR_TASK.md queue (append mode)
    staging_path = Path("CURSOR_TASK.md")
    
    # Append transcript path to queue file
    transcript_path_str = str(transcript_path)
    with open(staging_path, "a", encoding="utf-8") as f:
        f.write(f"{transcript_path_str}\n")
    
    # Count total videos in queue
    queue_count = 0
    if staging_path.exists():
        with open(staging_path, "r", encoding="utf-8") as f:
            queue_count = len([line.strip() for line in f if line.strip()])
    
    print("\n" + "=" * 60)
    print("✅ TRANSCRIPT ADDED TO QUEUE!")
    print("=" * 60)
    print(f"\n📄 Transcript saved: {transcript_path}")
    print(f"📋 Added to queue: CURSOR_TASK.md")
    print(f"📊 Total videos in queue: {queue_count}")
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Add more videos (optional): Run this script again with more URLs")
    print("   2. Process all videos: In Cursor Chat, say:")
    print("      'Complete the task in CURSOR_TASK.md'")
    print("\n   Cursor will process all videos sequentially until the queue is empty.")
    print("\n" + "=" * 60)


def prepare_for_cursor_with_transcript(url: str, transcript: str, metadata: dict):
    """
    Prepare transcript for Cursor workflow when transcript is already downloaded.
    This function is called by main.py.

    Args:
        url: YouTube video URL
        transcript: Plain text transcript (already downloaded)
        metadata: Video metadata dict
    """
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return

    # Save transcript
    transcript_dir = Path("YouTubeNotes/transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcript_dir / f"{video_id}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Add transcript file path to CURSOR_TASK.md queue (append mode)
    staging_path = Path("CURSOR_TASK.md")

    # Append transcript path to queue file
    transcript_path_str = str(transcript_path)
    with open(staging_path, "a", encoding="utf-8") as f:
        f.write(f"{transcript_path_str}\n")

    # Count total videos in queue
    queue_count = 0
    if staging_path.exists():
        with open(staging_path, "r", encoding="utf-8") as f:
            queue_count = len([line.strip() for line in f if line.strip()])

    print("\n" + "=" * 60)
    print("✅ TRANSCRIPT ADDED TO QUEUE!")
    print("=" * 60)
    print(f"\n📄 Transcript saved: {transcript_path}")
    print(f"📋 Added to queue: CURSOR_TASK.md")
    print(f"📊 Total videos in queue: {queue_count}")
    print(f"\n🎯 NEXT STEPS:")
    print("   1. Add more videos (optional): Run this script again with more URLs")
    print("   2. Process all videos: In Cursor Chat, say:")
    print("      'Complete the task in CURSOR_TASK.md'")
    print("\n   Cursor will process all videos sequentially until the queue is empty.")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Study Notes - Cursor Workflow: Download transcripts and queue for Cursor processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cursor_workflow.py "https://youtube.com/watch?v=VIDEO_ID"
  python cursor_workflow.py                                    # Interactive mode
  
The script downloads the transcript and adds it to CURSOR_TASK.md queue.
Then use Cursor to process all queued videos with: "Complete the task in CURSOR_TASK.md"
        """
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL"
    )
    
    args = parser.parse_args()
    
    if args.url:
        prepare_for_cursor(args.url)
    else:
        url = input("Enter YouTube URL: ").strip()
        if url:
            prepare_for_cursor(url)
        else:
            print("❌ No URL provided.")
            sys.exit(1)

