#!/usr/bin/env python3
"""
YouTube Study Notes Generator - Main Entry Point
===============================================
Unified entry point that downloads transcript first, then routes to appropriate workflow.
"""

import sys
import os
from transcript_utils import (
    extract_video_id,
    download_transcript,
    get_video_metadata,
    get_transcript_info,
    display_transcript_info
)


def get_workflow_choice(transcript_info: dict) -> str:
    """
    Display interactive menu and get user's workflow choice.

    Args:
        transcript_info: Dictionary with transcript statistics

    Returns:
        'api' or 'cursor'
    """
    print("\nSelect Workflow:")
    print("-" * 60)
    print("  [1] API Workflow (Gemini, Groq, OpenRouter)")
    print("      - Faster processing")
    print("      - Requires API key configuration")
    print("      - Real-time generation")
    print()
    print("  [2] Cursor Workflow")
    print("      - Uses Cursor's built-in LLM (no API cost)")
    print("      - Best for batch processing multiple videos")
    print("      - Supports very long transcripts")
    print("      - Requires Cursor IDE")
    print("-" * 60)

    while True:
        try:
            choice = input("\nEnter choice [1-2]: ").strip()

            if choice in ['1', 'api']:
                return 'api'
            elif choice in ['2', 'cursor']:
                return 'cursor'
            else:
                print("   Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\n\nCancelled.")
            sys.exit(0)


def run_api_workflow(video_id: str, transcript: str, metadata: dict):
    """Run the API workflow with pre-downloaded transcript"""
    print("\n🚀 Starting API Workflow...")
    print("-" * 60)

    # Import app module
    import app

    try:
        # Generate notes using transcript
        notes, provider, prompt_name = app.generate_notes_from_transcript(
            video_id=video_id,
            transcript=transcript,
            metadata=metadata
        )

        if not notes:
            print("\n❌ Failed to generate notes - no output received")
            sys.exit(1)

        # Save notes
        output_path = app.save_notes(video_id, notes, metadata, prompt_name, provider)
        print(f"\n✓ Notes saved to: {output_path}")

        # Ask about Notion
        publish_choice = input("\nPublish to Notion? [y/N]: ").strip().lower()
        if publish_choice in ['y', 'yes']:
            print("\n📤 Publishing to Notion...")
            try:
                # Format duration
                duration = metadata.get('duration', 0)
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
                else:
                    duration_str = str(duration)

                notion_url = app.publish_to_notion(
                    notes=notes,
                    video_title=metadata.get('title', 'Unknown'),
                    video_id=video_id,
                    channel=metadata.get('channel', 'Unknown'),
                    duration=duration_str,
                    provider=provider,
                    prompt_name=prompt_name,
                )
                print(f"✓ Published to: {notion_url}")
            except Exception as e:
                print(f"❌ Failed to publish to Notion: {e}")
                import traceback
                traceback.print_exc()

    except app.RestartException:
        print("\n⚠️  Restart requested - returning to main menu")
        # This will be caught by main() and the program will continue
    except Exception as e:
        print(f"\n❌ Error during API workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_cursor_workflow(video_id: str, transcript: str, metadata: dict):
    """Run the Cursor workflow with pre-downloaded transcript"""
    print("\n🚀 Starting Cursor Workflow...")
    print("-" * 60)

    # Import cursor_workflow module
    import cursor_workflow

    # Prepare for Cursor
    try:
        cursor_workflow.prepare_for_cursor_with_transcript(
            url=f"https://www.youtube.com/watch?v={video_id}",
            transcript=transcript,
            metadata=metadata
        )
        print("\n✓ Transcript added to Cursor queue")
        print("  Open Cursor IDE and run: /cursor")
    except Exception as e:
        print(f"\n❌ Error during Cursor workflow: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    print("\n🎬 YouTube Study Notes Generator")
    print("=" * 60)

    # Get YouTube URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        try:
            url = input("\nEnter YouTube URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nCancelled.")
            sys.exit(0)

    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        sys.exit(1)

    print(f"\n📹 Video ID: {video_id}")
    print("⏳ Downloading transcript...")

    # Download transcript
    transcript = download_transcript(video_id)
    if not transcript:
        print("\n❌ Failed to download transcript")
        print("   Please check:")
        print("   - The video has subtitles/captions available")
        print("   - Your internet connection")
        sys.exit(1)

    # Get video metadata
    metadata = get_video_metadata(video_id)

    # Get and display transcript info
    info = get_transcript_info(transcript, metadata)
    display_transcript_info(info)

    # Get workflow choice
    workflow = get_workflow_choice(info)

    # Run selected workflow
    if workflow == 'api':
        run_api_workflow(video_id, transcript, metadata)
    else:  # cursor
        run_cursor_workflow(video_id, transcript, metadata)

    print("\n✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
