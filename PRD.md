# Product Requirements Document (PRD)
## YouTube Study Notes Generator

**Version:** 1.0
**Last Updated:** January 3, 2026
**Status:** Active Development
**Target Audience:** Open Source Community

---

## 1. Executive Summary

### 1.1 Product Vision

The YouTube Study Notes Generator is a **personal productivity tool** designed to help professionals and learners efficiently consume and retain knowledge from YouTube video content. By leveraging multiple AI providers, it transforms video transcripts into structured, searchable study notes that integrate seamlessly into personal knowledge management systems.

### 1.2 Problem Statement

YouTube has become a primary source of learning for professionals, with millions of educational videos across technical tutorials, conference talks, and industry insights. However, consuming this content efficiently presents challenges:

- **Time consumption**: Watching full-length videos is time-intensive
- **Information retention**: Passive watching leads to poor retention
- **Searchability**: Video content cannot be easily searched or referenced
- **Note-taking quality**: Manual note-taking is tedious and inconsistent
- **Knowledge organization**: Video insights are difficult to integrate into existing knowledge systems

### 1.3 Solution

An AI-powered CLI application that:
1. Extracts transcripts from YouTube videos automatically
2. Generates structured study notes using multiple LLM providers
3. Outputs notes in multiple formats for different use cases
4. Integrates with Notion for centralized knowledge management
5. Supports both API-based and Cursor IDE-based workflows

---

## 2. Target Users & Use Cases

### 2.1 Primary Personas

| Persona | Description | Goals |
|---------|-------------|-------|
| **The Professional Learner** | Software engineer or knowledge worker consuming technical content | Efficiently learn from conference talks, tutorials, and industry videos; build a searchable knowledge base |
| **The Academic Researcher** | Student or researcher analyzing educational content | Extract key insights from lectures, seminars, and academic presentations; create structured study materials |
| **The Content Curator** | Professional who shares insights with teams | Summarize videos for team sharing; extract wisdom for newsletters or presentations |

### 2.2 Primary Use Cases

#### Use Case 1: Professional Development
**User Story:** As a software engineer, I want to quickly extract actionable insights from conference talks so that I can apply new techniques to my work without watching the full video.

**Workflow:**
1. Copy YouTube URL of conference talk
2. Run `ytnotes "URL"` or `./run.sh "URL"`
3. Select provider (e.g., Gemini for long videos)
4. Receive structured summary highlighting key techniques and code examples
5. Publish to Notion for future reference

#### Use Case 2: Personal Knowledge Base
**User Story:** As a lifelong learner, I want to build a searchable library of video summaries so that I can quickly reference information I've consumed.

**Workflow:**
1. Queue multiple videos using Cursor workflow: `ytcursor "URL1"`, `ytcursor "URL2"`, etc.
2. Process batch in Cursor IDE with single command
3. All notes auto-publish to Notion database
4. Search and retrieve insights via Notion

### 2.3 Secondary Use Cases

| Use Case | Description |
|----------|-------------|
| **Content Repurposing** | Adapt video content into blog posts or newsletters |
| **Team Knowledge Sharing** | Distribute insights from industry talks to teams |
| **Academic Study** | Create comprehensive study notes from lecture videos |
| **Research Synthesis** | Extract and organize wisdom across multiple sources |

---

## 3. Functional Requirements

### 3.1 Core Features

#### FR-1: Transcript Acquisition
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| FR-1.1 | Extract video ID from YouTube URL (multiple URL formats supported) |
| FR-1.2 | Download transcript using yt-dlp (primary method) |
| FR-1.3 | Fallback to youtube-transcript-api if yt-dlp fails |
| FR-1.4 | Cache transcripts locally to avoid re-download |
| FR-1.5 | Extract video metadata (title, channel, duration, chapters) |
| FR-1.6 | Save both SRT (timestamped) and TXT (plain text) formats |

#### FR-2: AI-Powered Note Generation
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| FR-2.1 | Support multiple AI providers (Gemini, Groq, OpenRouter, Z.AI, Cursor) |
| FR-2.2 | Display transcript statistics (word count, estimated tokens) |
| FR-2.3 | Show context usage and rate limits per provider |
| FR-2.4 | Recommend optimal provider based on transcript size |
| FR-2.5 | Handle rate limiting via automatic chunking for large transcripts |
| FR-2.6 | Support custom prompt templates via `prompts/` directory |

#### FR-3: Dual Workflow System
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| FR-3.1 | **API Workflow**: Generate notes via external AI APIs |
| FR-3.2 | **Cursor Workflow**: Batch process using Cursor IDE's built-in LLM |
| FR-3.3 | Unified entry point (`main.py`) with workflow selection |
| FR-3.4 | Quick mode flag (`-q`) for automated processing |
| FR-3.5 | Interactive menus with restart navigation |

#### FR-4: Output Management
**Priority:** P0 (Must Have)

| Requirement | Description |
|-------------|-------------|
| FR-4.1 | Save notes to `YouTubeNotes/` directory |
| FR-4.2 | Generate descriptive filenames with metadata |
| FR-4.3 | Support multiple output templates (youtube-summary, study-notes, fabric-extract-wisdom) |
| FR-4.4 | Include metadata headers (source URL, provider, timestamp) |
| FR-4.5 | Smart overwriting (update existing notes on re-run) |

#### FR-5: Notion Integration
**Priority:** P1 (Should Have)

| Requirement | Description |
|-------------|-------------|
| FR-5.1 | Create pages in Notion database via API |
| FR-5.2 | Map note properties to database columns |
| FR-5.3 | Handle Notion rate limits (batch large notes) |
| FR-5.4 | Preserve rich text formatting (headings, bold, links) |
| FR-5.5 | Auto-publish or prompt for publishing option |

#### FR-6: Provider Extensibility
**Priority:** P1 (Should Have)

| Requirement | Description |
|-------------|-------------|
| FR-6.1 | Add new providers via `providers.py` config only |
| FR-6.2 | Support OpenAI-compatible APIs out of the box |
| FR-6.3 | Support custom API types (Gemini, Z.AI streaming) |
| FR-6.4 | Display provider info (name, context, free tier status) |

#### FR-7: Batch Processing
**Priority:** P1 (Should Have)

| Requirement | Description |
|-------------|-------------|
| FR-7.1 | Queue multiple videos for Cursor workflow |
| FR-7.2 | Maintain queue in `CURSOR_TASK.md` file |
| FR-7.3 | Sequential processing with auto-removal |
| FR-7.4 | Progress tracking across batch |

### 3.2 Prompt Templates

| Template | Use Case | Sections |
|----------|----------|----------|
| **youtube-summary** | Default; analytically rigorous summaries | Title, Tags, Abstract, Arguments, Definitions, Methods, Evidence, Results, Conclusion |
| **study-notes** | Comprehensive learning materials | 8-section structured notes |
| **fabric-extract-wisdom** | Insight extraction | Surprising ideas, quotes, wisdom |

### 3.3 Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| **Performance** | Transcript download < 30 seconds; Note generation < 3 minutes |
| **Reliability** | 95%+ success rate for videos with available transcripts |
| **Compatibility** | Python 3.8+, macOS/Linux/Windows via WSL |
| **Security** | API keys stored in `.env` (gitignored); no hardcoded credentials |
| **Extensibility** | New providers addable via config without code changes |
| **Usability** | Clear error messages; helpful CLI prompts; progress indicators |

---

## 4. Technical Architecture

### 4.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
├─────────────────────┬───────────────────────┬───────────────────┤
│   CLI (main.py)     │   Quick Run (run.sh)  │   Alias (ytnotes) │
└─────────────────────┴───────────────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Workflow Selection                         │
├─────────────────────────────┬───────────────────────────────────┤
│       API Workflow          │         Cursor Workflow           │
└─────────────────────────────┴───────────────────────────────────┘
              │                                 │
              ▼                                 ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   transcript_utils.py    │    │   transcript_utils.py        │
│   (Shared Utilities)     │    │   (Shared Utilities)         │
└──────────────────────────┘    └──────────────────────────────┘
              │                                 │
              ▼                                 ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   providers.py           │    │   CURSOR_TASK.md (Queue)     │
│   (Multi-Provider)       │    │   + Cursor IDE LLM           │
└──────────────────────────┘    └──────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                           │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│  Gemini  │   Groq   │OpenRouter│  Z.AI    │      Notion API    │
└──────────┴──────────┴──────────┴──────────┴────────────────────┘
```

### 4.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.8+ | Core application |
| **YouTube Download** | yt-dlp, youtube-transcript-api | Transcript extraction |
| **HTTP Client** | requests | API communication |
| **Config Management** | python-dotenv | Environment variables |
| **Notion Integration** | notion-client | Publishing to Notion |
| **IDE Integration** | Cursor | Built-in LLM workflow |

### 4.3 Data Flow

**API Workflow:**
```
YouTube URL → Video ID → Transcript Download → Metadata Extraction
→ Provider/Template Selection → API Call → Local Save → Optional Notion Publish
```

**Cursor Workflow:**
```
YouTube URL → Video ID → Transcript Download → Cache Save
→ Add to CURSOR_TASK.md Queue → Cursor IDE Processing → Notes Generation
→ Optional Notion Publish
```

### 4.4 Provider Configuration

| Provider | Model | Context | Free Tier | API Type |
|----------|-------|---------|-----------|----------|
| Cursor Built-in | User choice | 200K+ | With subscription | IDE |
| Google Gemini | gemini-2.5-flash | 1M tokens | 15 req/min | Gemini |
| Groq | Llama 3.3 70B | 128K tokens | 12K TPM limit | OpenAI |
| OpenRouter | Xiaomi MiMo-V2-Flash | 256K tokens | Free | OpenAI |
| Z.AI | GLM-4.7 | 128K tokens | Paid | Z.AI streaming |

---

## 5. Success Metrics & KPIs

### 5.1 Primary Metric: User Satisfaction

**Definition:** Quality of generated notes and user experience

| Metric | Target | Measurement |
|--------|--------|-------------|
| Note quality score | 4.5/5 | User feedback, GitHub reactions |
| Successful processing rate | 95%+ | Videos with transcripts processed successfully |
| Output consistency | High | Minimal variance across providers for same input |

### 5.2 Secondary Metrics

| Metric | Description |
|--------|-------------|
| **GitHub Stars** | Community interest and adoption |
| **Contributors** | Open source community engagement |
| **Issue Resolution Time** | Support responsiveness |
| **Provider Diversity** | Number of supported AI providers |

---

## 6. Pain Points & Limitations

### 6.1 Current Pain Points

| Pain Point | Impact | Mitigation |
|------------|--------|------------|
| **Output Consistency** | Different providers produce varying quality/length of notes | Add provider quality ratings; implement output standardization |
| **API Rate Limits** | Free tiers have TPM/RPM limits causing throttling | Smart provider recommendations; chunking for large transcripts |
| **Transcript Availability** | Some videos lack transcripts or have poor quality subtitles | Dual download method; clear error messaging |
| **Setup Complexity** | Non-technical users may struggle with API keys and env setup | Improve onboarding documentation; add setup wizard |

### 6.2 Known Limitations

| Limitation | Workaround |
|------------|------------|
| Groq 12K TPM limit | Use Gemini for transcripts > 10K tokens |
| No native GUI | Use web wrapper tools or contribute one |
| English-only transcripts | Multi-language support planned for future |

---

## 7. Roadmap & Future Development

### 7.1 Priority Focus: More AI Providers

**Status:** High Priority

The top priority for future development is expanding provider support to:
- **Claude API (Anthropic)**: For users who want Claude-specific quality
- **OpenAI (GPT-4)**: For users with existing OpenAI subscriptions
- **DeepSeek**: Emerging Chinese LLM provider
- **Together AI**: Additional OpenAI-compatible option
- **Fireworks AI**: Fast inference provider

**Implementation Approach:**
1. Add provider entries to `providers.py`
2. Test API compatibility
3. Update documentation with key acquisition instructions
4. Gather community feedback on quality/cost

### 7.2 Additional Future Enhancements

| Feature | Priority | Description |
|---------|----------|-------------|
| **Output Standardization** | P1 | Improve consistency across providers |
| **Multi-language Support** | P2 | Support non-English transcripts and output |
| **Web UI** | P2 | Browser-based interface for non-technical users |
| **Enhanced Notion Integration** | P2 | Better database management, formatting |
| **Mobile App** | P3 | iOS/Android app for on-the-go processing |
| **Audio-only Support** | P3 | Support podcasts and audio files |
| **Collaborative Features** | P3 | Share notes with teams, comments |

---

## 8. Open Source Strategy

### 8.1 Community Engagement

| Initiative | Description |
|------------|-------------|
| **Contributor Guidelines** | Clear documentation for contributing providers, templates, features |
| **Issue Templates** | Structured bug reports and feature requests |
| **PR Template** | Standardized pull request format |
| **Provider Contributions** | Community can add new providers via `providers.py` PRs |

### 8.2 Documentation Priorities

| Document | Status | Priority |
|----------|--------|----------|
| README.md | Complete | - |
| PRD.md (this document) | Complete | - |
| CURSOR_WORKFLOW_GUIDE.md | Complete | - |
| BATCH_PROCESSING_GUIDE.md | Complete | - |
| Contributor Guide | Planned | P1 |
| API Documentation | Planned | P2 |

### 8.3 Contribution Areas

We welcome contributions in:
- New AI provider integrations
- Custom prompt templates
- Bug fixes and performance improvements
- Documentation enhancements
- UI/UX improvements (web interface, etc.)
- Internationalization

---

## 9. Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **YouTube API Changes** | High | Medium | Use multiple transcript sources; community monitoring |
| **Provider API Changes** | Medium | Low | Abstract provider interface; version testing |
| **Free Tier Elimination** | Medium | Medium | Support multiple providers; paid options available |
| **Notion API Limits** | Low | Low | Batch handling implemented; optional feature |
| **Output Quality Variance** | High | High | Provider ratings; standardization efforts |

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **API Workflow** | Processing mode using external AI APIs (Gemini, Groq, etc.) |
| **Cursor Workflow** | Processing mode using Cursor IDE's built-in LLM |
| **TPM** | Tokens Per Minute - API rate limit metric |
| **SRT** | SubRip subtitle format with timestamps |
| **Notion Database** | Notion workspace for storing generated notes |

### 10.2 Related Documents

- [README.md](README.md) - User documentation and quick start guide
- [CURSOR_WORKFLOW_GUIDE.md](CURSOR_WORKFLOW_GUIDE.md) - Cursor workflow documentation
- [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md) - Batch processing guide
- [CODE_CLEANUP_ANALYSIS.md](CODE_CLEANUP_ANALYSIS.md) - Technical debt analysis

### 10.3 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 3, 2026 | Initial PRD creation |

---

**Document Status:** Ready for Review
**Next Review Date:** Upon feature completion or quarterly
**Maintainer:** Project Contributors
