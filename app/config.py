import os
from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    LLM_PROVIDER: str = "ollama"
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    GROQ_API_URL: str = "https://api.groq.com/v1"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama"
    GROQ_MAX_OUTPUT_TOKENS: int = 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
