"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Aegis configuration — loaded from .env or environment variables."""

    # --- LLM ---
    # "gemini" for real calls; "fake" for a deterministic offline client (tests, CI)
    llm_provider: str = "gemini"
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-004"
    embedding_dim: int = 768

    # --- Database ---
    database_url: str = "postgresql://aegis:aegis@localhost:5432/aegis"

    # --- Guardrails ---
    # Second-stage LLM classifier for paraphrased injections. Costs one extra model call
    # per clean request; patterns alone still run when this is off.
    guardrail_llm_classifier: bool = True

    # --- RAG ---
    rag_top_k: int = 3
    rag_min_score: float = 0.0  # cosine similarity floor; chunks below this are dropped

    # --- Data ---
    data_dir: Path = REPO_ROOT / "data"

    # --- App ---
    log_level: str = "INFO"
    app_name: str = "Aegis"
    app_version: str = "0.1.0"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
