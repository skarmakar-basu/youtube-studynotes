#!/usr/bin/env python3
"""
YouTube Study Notes - Cursor Workflow
Downloads transcript and prepares it for Cursor's built-in LLM processing
"""

import os
import sys
import argparse
from pathlib import Path

# Import shared utilities
from transcript_utils import (
    extract_video_id,
    get_video_metadata,
    download_transcript,
)


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
