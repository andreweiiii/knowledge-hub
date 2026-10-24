"""Runtime settings, read once from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Ready-made settings for the AI services Knowledge Hub knows how to talk to.
# Every one of them speaks the OpenAI-compatible REST API (/embeddings, /chat/completions).
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "label": "OpenAI API",
        "base_url": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    },
    "ollama": {
        "label": "Ollama local model",
        "base_url": "http://127.0.0.1:11434/v1",
        "chat_model": "qwen2.5:7b",
        "embedding_model": "nomic-embed-text",
    },
    "compatible": {
        "label": "OpenAI-compatible API",
        "base_url": "",
        "chat_model": "",
        "embedding_model": "",
    },
    "none": {
        "label": "Retrieval only",
        "base_url": "",
        "chat_model": "",
        "embedding_model": "",
    },
}


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str = "none"
    provider_label: str = "Retrieval only"
    base_url: str = ""
    api_key: str = ""
    chat_model: str = ""
    embedding_model: str = ""
    chat_timeout: float = 120.0
    embed_timeout: float = 60.0

    data_dir: Path = ROOT_DIR / "backend" / "data"
    frontend_file: Path = ROOT_DIR / "ai-knowledge-base.html"

    # Public-demo protections. Off by default for local use.
    demo_mode: bool = False
    max_upload_mb: int = 20
    max_documents: int = 200
    chat_per_hour: int = 0  # 0 = unlimited
    uploads_per_hour: int = 0
    trust_proxy: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def ai_enabled(self) -> bool:
        return self.provider != "none" and bool(self.base_url and self.chat_model and self.embedding_model)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "knowledge.sqlite3"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        if env is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(ROOT_DIR / ".env")
            except ImportError:  # python-dotenv is optional
                pass
            env = dict(os.environ)

        provider = (env.get("AI_PROVIDER") or "").strip().lower()
        if provider not in PROVIDER_PRESETS:
            # Old .env files only had OPENAI_API_KEY; keep them working.
            provider = "openai" if env.get("OPENAI_API_KEY") else "none"
        preset = PROVIDER_PRESETS[provider]

        prefix = provider.upper()  # OPENAI_CHAT_MODEL, OLLAMA_CHAT_MODEL, ...
        api_key = (env.get("AI_API_KEY") or env.get(f"{prefix}_API_KEY") or "").strip()
        if provider == "ollama":
            api_key = api_key or "ollama"  # Ollama ignores the key but the header must exist
        base_url = (env.get("AI_BASE_URL") or env.get(f"{prefix}_BASE_URL") or preset["base_url"]).rstrip("/")
        chat_model = env.get("AI_CHAT_MODEL") or env.get(f"{prefix}_CHAT_MODEL") or preset["chat_model"]
        embedding_model = env.get("AI_EMBEDDING_MODEL") or env.get(f"{prefix}_EMBEDDING_MODEL") or preset["embedding_model"]

        if provider in {"openai", "compatible"} and not api_key:
            provider, preset = "none", PROVIDER_PRESETS["none"]

        demo = _bool(env.get("DEMO_MODE"))
        origins = [o.strip() for o in (env.get("CORS_ORIGINS") or "*").split(",") if o.strip()]
        return cls(
            provider=provider,
            provider_label=env.get("AI_PROVIDER_LABEL") or preset["label"],
            base_url=base_url if provider != "none" else "",
            api_key=api_key if provider != "none" else "",
            chat_model=chat_model if provider != "none" else "",
            embedding_model=embedding_model if provider != "none" else "",
            chat_timeout=float(_int(env.get("AI_CHAT_TIMEOUT"), 120)),
            embed_timeout=float(_int(env.get("AI_EMBED_TIMEOUT"), 60)),
            data_dir=Path(env.get("KH_DATA_DIR") or ROOT_DIR / "backend" / "data"),
            demo_mode=demo,
            max_upload_mb=_int(env.get("MAX_UPLOAD_MB"), 5 if demo else 20),
            max_documents=_int(env.get("MAX_DOCUMENTS"), 25 if demo else 200),
            chat_per_hour=_int(env.get("CHAT_PER_HOUR"), 30 if demo else 0),
            uploads_per_hour=_int(env.get("UPLOADS_PER_HOUR"), 5 if demo else 0),
            trust_proxy=_bool(env.get("TRUST_PROXY"), default=demo),
            cors_origins=origins,
        )

    def setup_hint(self) -> str:
        if self.provider == "ollama":
            return f"Start Ollama and run: ollama pull {self.chat_model} && ollama pull {self.embedding_model}"
        if self.provider == "none":
            return "AI answers are off. Set AI_PROVIDER in .env to turn them on."
        return "Check the API key and model names in .env."
