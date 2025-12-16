"""
RAG (Retrieval-Augmented Generation) пакет для Telegram-бота (Yandex GPT версия).
Этот пакет содержит все компоненты для работы RAG-системы через Yandex API:
- Embedder: создание векторных представлений через Yandex Embeddings
- VectorStore: хранение и поиск векторов (FAISS)
- Retriever: извлечение релевантных документов
- Pipeline: координация всех компонентов
- YandexGPT: генерация ответов через Yandex GPT
"""

__version__ = "2.0.0"
__author__ = "RAG Bot Team (Yandex Edition)"

from .yandex_embedder import YandexEmbedder
from .vectorstore import FAISSVectorStore
from .retriever import DocumentRetriever
from .pipeline import RAGPipeline
from .yandex_gpt import YandexGPT

__all__ = [
    "YandexEmbedder",
    "FAISSVectorStore",
    "DocumentRetriever",
    "RAGPipeline",
    "YandexGPT"
]
