# Unified Workflow Implementation

## Overview

The project has been refactored to download transcripts **once**, then route to the appropriate workflow through an interactive menu. This eliminates duplicate transcript downloading logic and improves the user experience.

## What Changed

### New Files Created

1. **`transcript_utils.py`** - Shared transcript utilities module
   - Common transcript downloading functions (yt-dlp + youtube-transcript-api)
   - Transcript caching
   - Video metadata extraction
   - Transcript information display

2. **`main.py`** - New unified entry point
   - Downloads transcript first
   - Shows transcript statistics
   - Interactive workflow selection menu
   - Routes to API or Cursor workflow

### Modified Files

1. **`app.py`** - Added new functions:
   - `generate_notes_from_transcript()` - Accepts pre-downloaded transcript
   - `save_notes()` - Save notes with metadata header

2. **`cursor_workflow.py`** - Added new function:
   - `prepare_for_cursor_with_transcript()` - Accepts pre-downloaded transcript

## How to Use

### Option 1: Using the Unified Entry Point (Recommended)

```bash
# Interactive mode (prompts for URL)
python main.py

# Provide URL directly
python main.py "https://youtube.com/watch?v=VIDEO_ID"
```

**Workflow:**
1. Enter YouTube URL
2. Transcript is downloaded automatically
3. View transcript statistics (words, characters, estimated tokens)
4. Select workflow from interactive menu:
   - **[1] API Workflow** - Fast processing, requires API key
   - **[2] Cursor Workflow** - Free, batch processing, best for long videos

### Option 2: Using Individual Workflows (Backward Compatible)

The original workflows still work as before:

```bash
# API workflow directly
python app.py "https://youtube.com/watch?v=VIDEO_ID"

# Cursor workflow directly
python cursor_workflow.py "https://youtube.com/watch?v=VIDEO_ID"
```

## Benefits

1. **No Duplicate Downloads** - Transcript downloaded once, cached for reuse
2. **Better User Experience** - See transcript info before choosing workflow
3. **Early Validation** - Know immediately if transcript is unavailable
4. **Unified Caching** - Both workflows benefit from shared cache
5. **Backward Compatible** - Existing workflows continue to work

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Get YouTube URL                                   │  │
│  │  2. Download transcript (transcript_utils.py)        │  │
│  │  3. Display transcript statistics                     │  │
│  │  4. Show interactive menu                             │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│                 ┌──────────────────────┐                    │
│                 │  User selects option │                    │
│                 └──────────────────────┘                    │
│                            │                                 │
│           ┌────────────────┴────────────────┐               │
│           ▼                                 ▼               │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │  API Workflow    │          │ Cursor Workflow  │        │
│  │  (app.py)        │          │(cursor_workflow) │        │
│  └──────────────────┘          └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
YouTube/
├── main.py                      # NEW: Unified entry point
├── transcript_utils.py          # NEW: Shared transcript utilities
├── app.py                       # Modified: Added generate_notes_from_transcript()
├── cursor_workflow.py           # Modified: Added prepare_for_cursor_with_transcript()
├── publish_to_notion.py         # Unchanged
├── providers.py                 # Unchanged
└── prompts/                     # Unchanged
```

## Example Session

```bash
$ python main.py "https://youtube.com/watch?v=dQw4w9WgXcQ"

🎬 YouTube Study Notes Generator
============================================================

📹 Video ID: dQw4w9WgXcQ
⏳ Downloading transcript...
   Downloaded with yt-dlp

============================================================
  ✓ Transcript Downloaded Successfully
============================================================

  📹 Video: Never Gonna Give You Up
  📺 Channel: Rick Astley
  ⏱️  Duration: 3m 33s

  📊 Transcript Statistics:
     Words: 412
     Characters: 2,456
     Est. Tokens: 535
============================================================

Select Workflow:
------------------------------------------------------------
  [1] API Workflow (Gemini, Groq, OpenRouter)
      - Faster processing
      - Requires API key configuration
      - Real-time generation

  [2] Cursor Workflow
      - Uses Cursor's built-in LLM (no API cost)
      - Best for batch processing multiple videos
      - Supports very long transcripts
      - Requires Cursor IDE
------------------------------------------------------------

Enter choice [1-2]: 1

🚀 Starting API Workflow...
------------------------------------------------------------
...
```

## Notes

- Transcript caching is enabled by default in `YouTubeNotes/transcripts/`
- Both workflows now use the same shared transcript utilities
- The unified flow is optional - you can still call workflows directly
- All existing functionality remains intact
