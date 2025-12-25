"""
Provider configurations for YouTube Study Notes Generator.
==========================================================

To add a new provider:
1. Add an entry to PROVIDERS dict below
2. Set the appropriate api_type:
   - "openai": Standard OpenAI-compatible API (Groq, OpenRouter, Together, etc.)
   - "gemini": Google's Gemini API
   - "zai": Z.AI streaming API
3. For OpenAI-compatible APIs, just add config - no code changes needed!

Required fields:
- name: Display name shown in UI
- model: Model identifier for API calls
- nickname: Short name used in output filenames
- env_key: Environment variable name for API key
- context: Human-readable context window size
- context_tokens: Context window in tokens (for calculations)
- free: Whether the provider is free to use
- api_type: One of "openai", "gemini", "zai"
- api_url: API endpoint URL
"""

PROVIDERS = {
    # ============================================================
    # GOOGLE
    # ============================================================
    "gemini": {
        "name": "Google Gemini 2.5 Flash",
        "model": "gemini-2.5-flash",
        "nickname": "gemini2.5Flash",
        "env_key": "GEMINI_API_KEY",
        "context": "1M tokens",
        "context_tokens": 1_000_000,
        "free": True,
        "api_type": "gemini",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
    },

    # ============================================================
    # OPENAI-COMPATIBLE PROVIDERS (easy to add more!)
    # ============================================================
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "model": "llama-3.3-70b-versatile",
        "nickname": "groqL3.3.70B",
        "env_key": "GROQ_API_KEY",
        "context": "128K tokens",
        "context_tokens": 128_000,
        "rate_limit_tpm": 12_000,  # Free tier TPM limit
        "free": True,
        "api_type": "openai",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "openrouter-nova-lite": {
        "name": "OpenRouter (Xiaomi: MiMo-V2-Flash)",
        "model": "xiaomi/mimo-v2-flash:free",
        "nickname": "openrouterMimoFlash",
        "env_key": "OPENROUTER_API_KEY",
        "context": "256K tokens",
        "context_tokens": 256_000,
        "free": True,
        "api_type": "openai",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
    },

    # ============================================================
    # EXAMPLE: Add more OpenAI-compatible providers easily
    # ============================================================
    # "together": {
    #     "name": "Together AI (Llama 3.1 70B)",
    #     "model": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    #     "nickname": "together",
    #     "env_key": "TOGETHER_API_KEY",
    #     "context": "128K tokens",
    #     "context_tokens": 128_000,
    #     "free": False,
    #     "api_type": "openai",
    #     "api_url": "https://api.together.xyz/v1/chat/completions",
    # },
    # "deepseek": {
    #     "name": "DeepSeek Chat",
    #     "model": "deepseek-chat",
    #     "nickname": "deepseek",
    #     "env_key": "DEEPSEEK_API_KEY",
    #     "context": "64K tokens",
    #     "context_tokens": 64_000,
    #     "free": False,
    #     "api_type": "openai",
    #     "api_url": "https://api.deepseek.com/v1/chat/completions",
    # },
    # "fireworks": {
    #     "name": "Fireworks AI (Llama 3.1 70B)",
    #     "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    #     "nickname": "fireworks",
    #     "env_key": "FIREWORKS_API_KEY",
    #     "context": "128K tokens",
    #     "context_tokens": 128_000,
    #     "free": False,
    #     "api_type": "openai",
    #     "api_url": "https://api.fireworks.ai/inference/v1/chat/completions",
    # },

    # ============================================================
    # SPECIALIZED PROVIDERS (require custom handling in app.py)
    # ============================================================
    "zai": {
        "name": "Z.AI GLM-4.7",
        "model": "glm-4.7",
        "nickname": "zaiGLM4.7",
        "env_key": "ZAI_API_KEY",
        "context": "128K tokens",
        "context_tokens": 128_000,
        "free": False,
        "api_type": "zai",
        "api_url": "https://api.z.ai/api/coding/paas/v4/chat/completions",
    },
    
    # ============================================================
    # CURSOR (Built-in LLM - no API key needed)
    # ============================================================
    "cursor": {
        "name": "Cursor Built-in LLM",
        "model": "claude-3.5-sonnet",  # Default model, varies by user selection
        "nickname": "cursor",
        "env_key": None,  # No API key needed - uses Cursor subscription
        "context": "200K+ tokens",
        "context_tokens": 200_000,
        "free": True,  # Included in Cursor subscription
        "api_type": "cursor",  # Not used for API calls, just for identification
        "api_url": None,  # Not applicable - uses Cursor's built-in LLM
    },
}
