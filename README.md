# YouTube Study Notes Generator

Convert YouTube videos into structured, consultant-optimized study notes using AI.

---

## Features

- **Multi-provider support** — Choose between Google Gemini, Groq, or Z.AI
- **Automatic transcription** — Fetches YouTube's auto-generated captions
- **Custom note format** — Uses your `gpt-inst.md` template for consistent output
- **Smart overwriting** — Re-running on the same video updates the existing note
- **Transcript caching** — Transcripts are saved locally to avoid re-fetching
- **Progress indicator** — Visual feedback during generation
- **Token usage stats** — See context usage before selecting a provider

---

## Supported AI Providers

| Provider | Model | Context | Free Tier | Best For |
|----------|-------|---------|-----------|----------|
| **Google Gemini** | gemini-2.5-flash | 1M tokens | ✅ 15 req/min | Long videos, high quality |
| **Groq** | Llama 3.3 70B | 128K tokens | ✅ ~30 req/min | Fast results |
| **Z.AI** | GLM-4.6 | 32K tokens | ❌ Paid | Existing subscribers |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/skarmakar-basu/youtube-studynotes.git
cd youtube-studynotes
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env  # If example exists, or create manually
```

Add your API keys (you only need ONE provider):

```env
# Google Gemini (FREE) — https://aistudio.google.com
GEMINI_API_KEY=your_key_here

# Groq (FREE) — https://console.groq.com
GROQ_API_KEY=your_key_here

# Z.AI (Paid) — https://z.ai
ZAI_API_KEY=your_key_here
```

### 5. Run the App
```bash
python app.py
```

Or with a URL directly:
```bash
python app.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

---

## Usage

### Interactive Mode
```bash
python app.py
```
You'll be prompted for a YouTube URL, then shown provider options with token usage stats.

### Direct URL Mode
```bash
python app.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Provider Selection
```
==================================================
  🤖 Select AI Provider
==================================================

  📊 Transcript: ~5,432 words (~7,062 tokens)
------------------------------------------------------------

  1. Google Gemini 2.5 Flash [FREE] ⭐ Recommended
     Context: 1M tokens | ✅ Usage: 0.7%

  2. Groq (Llama 3.3 70B) [FREE]
     Context: 128K tokens | ✅ Usage: 5.5%

  3. Z.AI GLM-4.6 [PAID]
     Context: 32K tokens | ✅ Usage: 22.1%

Enter choice (1-3): 1
```

### Find Your Notes
Generated notes are saved to:
```
YouTubeNotes/<video_id>_<video_title>_<model>.md
```

---

## Getting Free API Keys

### Google Gemini (Recommended)
1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google
3. Click **Get API Key** → **Create API key**
4. Copy and add to `.env`

### Groq
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / Sign in
3. Go to **API Keys** → **Create API Key**
4. Copy and add to `.env`

---

## Project Structure

```
youtube-studynotes/
├── app.py              # Main application
├── gpt-inst.md         # Note format template (customizable)
├── requirements.txt    # Python dependencies
├── .env                # API keys (create this, not committed)
├── .gitignore          # Git ignore rules
├── README.md           # This guide
└── YouTubeNotes/       # Generated notes output
    ├── transcripts/    # Cached transcripts
    │   └── <video_id>.txt
    └── <video_id>_<title>_<model>.md
```

---

## Customizing Note Format

Edit `gpt-inst.md` to change how notes are structured. The AI follows this template when generating notes.

Current template sections:
1. **Title & Discovery Tags** — Clear title with hashtags
2. **The Hook** — Why this topic matters
3. **Core Concept** — The WHAT and WHY
4. **How It Works** — The mechanics and HOW
5. **Three Perspectives** — Real-world, technical, and pitfalls
6. **Practical Cheat Sheet** — Quick reference bullets
7. **Key Terms Glossary** — Important definitions
8. **Memory Anchors** — Summary, analogy, flashcards, deeper questions
9. **Key Moments** — Notable timestamps (optional)

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| **No API keys configured** | Add at least one key to `.env` |
| **Transcripts are disabled** | Video owner turned off captions, try another video |
| **No transcript found** | Video has no captions, try another video |
| **Response truncated** | Rare with Gemini's 1M context; try Gemini for long videos |
| **Timeout** | Long videos take 1-3 min; be patient or try Groq (faster) |
| **Module not found** | Ensure virtual environment is activated: `source venv/bin/activate` |
| **Permission denied** | Check file permissions in YouTubeNotes folder |

---

## Technical Details

- **Transcription**: `youtube-transcript-api` — Fetches YouTube's existing captions
- **Video metadata**: `yt-dlp` — Title, channel, duration extraction
- **API calls**: `requests` — Direct REST calls, no SDK dependencies
- **Configuration**: `python-dotenv` — Loads `.env` file
- **Python**: 3.8+ recommended

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

Personal use. API usage subject to respective provider terms.
