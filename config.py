"""Конфигурация приложения чата с Bionic (OpenAI-совместимый API).

Валидация настроек выполняется через pydantic. Значения по умолчанию
могут быть переопределены переменными окружения:
    BIONIC_BASE_URL  — базовый URL OpenAI-совместимого API сервера Bionic
    BIONIC_API_KEY   — API-ключ (для локальных серверов часто не требуется)
    BIONIC_MODEL     — идентификатор модели по умолчанию
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, field_validator

# Значения по умолчанию для локального сервера Bionic
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "not-needed"  # noqa: S105 — плейсхолдер для локальных серверов
DEFAULT_MODEL = "bionic"


class BionicSettings(BaseModel):
    """Настройки подключения к серверу Bionic."""

    base_url: str = Field(default=DEFAULT_BASE_URL, description="Базовый URL API")
    api_key: str = Field(default=DEFAULT_API_KEY, description="API-ключ")
    model: str = Field(default=DEFAULT_MODEL, description="Имя модели")

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("model", "api_key")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("значение не может быть пустым")
        return v.strip()

    @classmethod
    def from_env(cls) -> "BionicSettings":
        """Собрать настройки из переменных окружения (с фолбэками)."""
        return cls(
            base_url=os.getenv("BIONIC_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("BIONIC_API_KEY", DEFAULT_API_KEY),
            model=os.getenv("BIONIC_MODEL", DEFAULT_MODEL),
        )


class GenerationParams(BaseModel):
    """Параметры генерации, настраиваемые через боковую панель."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    system_prompt: str = Field(
        default="Ты — полезный ассистент. Отвечай кратко и по делу.",
        description="Системный промпт диалога",
    )
    stream: bool = Field(default=True, description="Стриминг ответа")


def load_settings() -> BionicSettings:
    return BionicSettings.from_env()
