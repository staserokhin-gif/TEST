"""Streamlit-приложение: чат с локальной LLM на сервере Bionic.

Запуск:
    streamlit run app.py

Переменные окружения (опционально):
    BIONIC_BASE_URL, BIONIC_API_KEY, BIONIC_MODEL
"""

from __future__ import annotations

import os

import streamlit as st
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError

from bionic_client import BionicClient
from config import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL, GenerationParams

st.set_page_config(
    page_title="Bionic Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------- #
# Состояние сессии
# ---------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "messages": [],          # список {"role": ..., "content": ...}
        "pending_user": None,    # текст только что отправленного сообщения
        "gen_params": GenerationParams(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# ---------------------------------------------------------------------- #
# Боковая панель: подключение и параметры модели
# ---------------------------------------------------------------------- #
def sidebar_controls() -> tuple[str, str, str, GenerationParams]:
    with st.sidebar:
        st.header("⚙️ Подключение к Bionic")

        base_url = st.text_input(
            "Base URL (OpenAI-совместимый API)",
            value=os.getenv("BIONIC_BASE_URL", DEFAULT_BASE_URL),
            help="Адрес сервера Bionic, например http://bionic-server:8000/v1",
        )
        api_key = st.text_input(
            "API Key",
            value=os.getenv("BIONIC_API_KEY", DEFAULT_API_KEY),
            type="password",
            help="Ключ доступа (для локальных серверов можно оставить как есть)",
        )
        model = st.text_input(
            "Модель",
            value=os.getenv("BIONIC_MODEL", DEFAULT_MODEL),
        )

        if st.button("🔌 Проверить соединение", use_container_width=True):
            try:
                client = BionicClient(base_url=base_url, api_key=api_key, model=model, timeout=10)
                models = client.list_models()
                if models:
                    st.success(f"Соединение установлено. Модели: {', '.join(models[:5])}")
                else:
                    st.warning("Сервер ответил, но список моделей пуст или недоступен.")
            except Exception as exc:  # noqa: BLE001 — показываем пользователю
                st.error(f"Не удалось подключиться: {exc}")

        st.divider()
        st.header("🎛️ Параметры генерации")

        params = GenerationParams(
            temperature=st.slider("Temperature", 0.0, 2.0, 0.7, 0.05),
            top_p=st.slider("Top-p", 0.0, 1.0, 0.95, 0.05),
            max_tokens=st.slider("Max tokens", 1, 4096, 1024, 1),
            stream=st.toggle("Стриминг ответа", value=True),
            system_prompt=st.text_area(
                "Системный промпт",
                value=GenerationParams.model_fields["system_prompt"].default,
                height=100,
            ),
        )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Очистить историю", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("↩️ Отменить шаг", use_container_width=True):
                # удаляем последнюю пару user+assistant
                msgs = st.session_state.messages
                if msgs and msgs[-1]["role"] == "assistant":
                    msgs.pop()
                if msgs and msgs[-1]["role"] == "user":
                    msgs.pop()
                st.rerun()

    return base_url, api_key, model, params


# ---------------------------------------------------------------------- #
# История сообщений
# ---------------------------------------------------------------------- #
def render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def build_request_messages(user_text: str) -> list[dict]:
    """Формирует полный контекст запроса: system + история + новое сообщение."""
    params: GenerationParams = st.session_state.gen_params
    history = st.session_state.messages
    # окна контекста: берём не более последних N пар сообщений
    max_pairs = int(os.getenv("BIONIC_CONTEXT_WINDOW_PAIRS", "20"))
    trimmed = history[-max_pairs * 2:]
    messages = [{"role": "system", "content": params.system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_text})
    return messages


# ---------------------------------------------------------------------- #
# Отправка запроса и стриминг ответа
# ---------------------------------------------------------------------- #
def send_and_stream(client: BionicClient, params: GenerationParams, user_text: str) -> None:
    request_messages = build_request_messages(user_text)

    gen_kwargs = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            if params.stream:
                for part in client.stream_chat(request_messages, **gen_kwargs):
                    full_response += part
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                full_response = client.complete(request_messages, **gen_kwargs)
                placeholder.markdown(full_response)
        except AuthenticationError:
            full_response = "❌ Ошибка авторизации: проверьте API Key."
            placeholder.error(full_response)
        except (APIConnectionError, APITimeoutError) as exc:
            full_response = (
                f"❌ Сервер Bionic недоступен ({exc.__class__.__name__}). "
                "Проверьте Base URL и запущен ли сервер."
            )
            placeholder.error(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            return
        except APIError as exc:
            full_response = f"❌ Ошибка API Bionic: {exc}"
            placeholder.error(full_response)
        except Exception as exc:  # noqa: BLE001
            full_response = f"❌ Неизвестная ошибка: {exc}"
            placeholder.error(full_response)

        if not full_response.strip():
            full_response = "(пустой ответ модели)"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------- #
# Основной цикл UI
# ---------------------------------------------------------------------- #
def main() -> None:
    st.title("🤖 Чат с LLM на сервере Bionic")
    st.caption("Локальная модель через OpenAI-совместимый REST API · стриминг ответов")

    base_url, api_key, model, params = sidebar_controls()
    st.session_state.gen_params = params

    render_history()

    client = BionicClient(base_url=base_url, api_key=api_key, model=model)

    if prompt := st.chat_input("Введите сообщение…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        send_and_stream(client, params, prompt)


main()
