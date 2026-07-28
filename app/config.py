from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    LLM_PROVIDER: str = "ollama"

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"

    GROQ_API_URL: str = "https://api.groq.com/v1"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama"
    GROQ_MAX_OUTPUT_TOKENS: int = 1024

    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"


@lru_cache()
def get_settings() -> Settings:
    return Settings()