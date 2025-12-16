"""
Конфигурация для Telegram-бота с RAG на основе Yandex GPT API.

Этот модуль содержит все настройки для работы бота:
- Токены доступа к Telegram и Yandex Cloud
- Модели для эмбеддингов и генерации текста
- Пути к файлам и базам данных
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# ========== YANDEX НАСТРОЙКИ ==========
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("Не установлены YANDEX_API_KEY или YANDEX_FOLDER_ID в .env файле!")

# Модели Yandex
YANDEX_GPT_MODEL = "yandexgpt-lite"
YANDEX_EMBED_MODEL = "text-search-doc"  # Для эмбеддингов

# ========== TELEGRAM НАСТРОЙКИ ==========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не установлен в .env файле!")

# ========== ПРОМПТЫ ==========
SYSTEM_PROMPT = """Ты — интеллектуальный ассистент с доступом к базе знаний.
Твоя задача — отвечать на вопросы пользователей, опираясь на предоставленный контекст.

Правила работы:
1. Используй информацию из контекста базы знаний для формирования ответа
2. Учитывай историю предыдущих сообщений для понимания контекста разговора
3. Если пользователь спрашивает про "это", "то", "предыдущий вопрос" - смотри в историю
4. Если в контексте нет информации для ответа, честно скажи об этом
5. Отвечай на русском языке четко и структурированно
6. Если уместно, используй списки и пункты для лучшей читаемости
7. Будь вежливым и профессиональным
"""

RAG_PROMPT_TEMPLATE = """Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Ответ:"""

# ========== ЛОГИРОВАНИЕ ==========
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========== FAISS НАСТРОЙКИ ==========
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "index.faiss"
FAISS_METADATA_PATH = BASE_DIR / "metadata.json"
DOCS_PATH = BASE_DIR / "data" / "docs"

# ========== RAG НАСТРОЙКИ ==========
TOP_K_RESULTS = 3
MAX_CONTEXT_LENGTH = 3000
MAX_HISTORY_LENGTH = 10

# Таймауты для запросов
REQUEST_TIMEOUT = 30

