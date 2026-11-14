"""Клиент для взаимодействия с LLM на сервере Bionic.

Использует openai>=1.0 (новый синтаксис клиента) с OpenAI-совместимым
REST API: base_url указывает на сервер Bionic.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

from openai import OpenAI

Message = dict[str, Any]  # {"role": "user"|"assistant"|"system", "content": str}


class BionicClient:
    """Тонкая обёртка над OpenAI-совместимым API Bionic."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Синхронный (нестриминговый) запрос
    # ------------------------------------------------------------------ #
    def complete(self, messages: Sequence[Message], **params: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            stream=False,
            **params,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    # Стриминговый запрос: генератор частей текста
    # ------------------------------------------------------------------ #
    def stream_chat(
        self, messages: Sequence[Message], **params: Any
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            stream=True,
            **params,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    # ------------------------------------------------------------------ #
    def list_models(self) -> list[str]:
        """Список доступных моделей на сервере (для проверки связи)."""
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception:
            return []
