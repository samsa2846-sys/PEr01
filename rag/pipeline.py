"""
Модуль RAG-пайплайна для работы через Yandex GPT.
"""

import logging
from typing import Dict, List, Optional
from rag.yandex_embedder import YandexEmbedder
from rag.vectorstore import FAISSVectorStore
from rag.retriever import DocumentRetriever
from rag.yandex_gpt import YandexGPT
from config import (
    YANDEX_GPT_MODEL,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    TOP_K_RESULTS,
    MAX_CONTEXT_LENGTH,
)

# Настраиваем логирование
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Основной класс RAG-пайплайна.
    
    Координирует работу всех компонентов:
    1. Поиск релевантных документов
    2. Формирование контекста
    3. Генерация ответа через Yandex GPT
    """
    
    def __init__(self):
        """
        Инициализация RAG-пайплайна с Yandex компонентами.
        """
        logger.info("Инициализация RAG Pipeline (Yandex версия)...")
        
        # Инициализируем компоненты RAG
        self.embedder = YandexEmbedder()
        self.vectorstore = FAISSVectorStore()
        self.retriever = DocumentRetriever(self.embedder, self.vectorstore)
        self.llm = YandexGPT()
        
        # Пытаемся загрузить существующий индекс
        self.is_loaded = self.vectorstore.load()
        
        if self.is_loaded:
            logger.info("RAG Pipeline инициализирован с загруженным индексом")
        else:
            logger.warning("RAG Pipeline инициализирован без индекса. "
                         "Выполните индексацию документов командой /ingest.")
    
    def query(self, query: str, top_k: int = TOP_K_RESULTS) -> Dict[str, any]:
        """
        Выполняет RAG-запрос без учета истории.
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество документов для извлечения
            
        Returns:
            Словарь с ответом и метаданными
        """
        return self.query_with_history(query, history=[], top_k=top_k)
    
    def query_with_history(self, query: str, history: List[Dict] = None, 
                          top_k: int = TOP_K_RESULTS) -> Dict[str, any]:
        """
        Выполняет RAG-запрос с учетом истории диалога.
        
        Args:
            query: Текстовый запрос пользователя
            history: История предыдущих сообщений
            top_k: Количество документов для извлечения
            
        Returns:
            Словарь с ответом и метаданными
        """
        if history is None:
            history = []
        
        # Проверяем загружен ли индекс
        if not self.is_loaded:
            logger.error("Индекс не загружен, невозможно выполнить запрос")
            return {
                "answer": "❌ База знаний не загружена. Выполните команду /ingest для индексации документов.",
                "context": "",
                "sources": [],
                "model": YANDEX_GPT_MODEL,
                "clean_query": query
            }
        
        try:
            # Шаг 1: Извлекаем релевантный контекст
            context = self.retriever.retrieve_context(
                query, 
                top_k=top_k,
                max_length=MAX_CONTEXT_LENGTH
            )
            
            # Шаг 2: Получаем источники
            sources = self.retriever.get_relevant_sources(query, top_k)
            
            # Шаг 3: Формируем промпт с контекстом
            prompt_with_context = RAG_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )
            
            # Шаг 4: Формируем историю сообщений
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Добавляем историю разговора
            if history:
                # Ограничиваем историю
                max_history_items = 10 * 2  # 10 пар вопрос-ответ
                recent_history = history[-max_history_items:] if len(history) > max_history_items else history
                messages.extend(recent_history)
                logger.info(f"Добавлено {len(recent_history)} сообщений из истории")
            
            # Добавляем текущий вопрос с RAG контекстом
            messages.append({"role": "user", "content": prompt_with_context})
            
            # Шаг 5: Генерируем ответ через Yandex GPT
            logger.info(f"Генерация ответа через Yandex GPT (всего сообщений: {len(messages)})")
            
            answer = self.llm.generate_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            logger.info(f"Ответ сгенерирован, длина: {len(answer)} символов")
            
            # Возвращаем результат
            return {
                "answer": answer,
                "context": context,
                "sources": sources,
                "model": YANDEX_GPT_MODEL,
                "clean_query": query
            }
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении RAG-запроса: {e}")
            return {
                "answer": f"❌ Произошла ошибка при обработке запроса: {str(e)}",
                "context": "",
                "sources": [],
                "model": YANDEX_GPT_MODEL,
                "clean_query": query
            }
    
    def process_image(self, image_url: str, query: str = None) -> Dict[str, any]:
        """
        Обрабатывает изображение через Yandex Vision API (заглушка).
        
        Args:
            image_url: URL изображения
            query: Вопрос пользователя (опционально)
            
        Returns:
            Словарь с результатами
        """
        # Используем заглушку из YandexGPT класса
        result = self.llm.process_image(image_url, query)
        
        # Если есть вопрос, выполняем RAG-запрос по извлеченному тексту
        if query and result.get("extracted_text"):
            # Здесь можно добавить RAG-поиск по извлеченному тексту
            pass
        
        return result
    
    def index_documents(self, documents: List[str], sources: List[str]) -> bool:
        """
        Индексирует документы в векторное хранилище.
        
        Args:
            documents: Список текстов документов
            sources: Список источников (имен файлов)
            
        Returns:
            True если индексация успешна
        """
        logger.info(f"Начало индексации {len(documents)} документов (Yandex)")
        
        try:
            # Шаг 1: Создаем эмбеддинги через Yandex
            logger.info("Создание эмбеддингов через Yandex API...")
            embeddings = self.embedder.embed_texts(documents)
            
            # Шаг 2: Создаем новый индекс с правильной размерностью
            dimension = self.embedder.get_embedding_dimension()
            self.vectorstore.create_index(dimension)
            
            # Шаг 3: Добавляем документы в индекс
            self.vectorstore.add_documents(documents, embeddings, sources)
            
            # Шаг 4: Сохраняем на диск
            self.vectorstore.save()
            
            self.is_loaded = True
            logger.info("Индексация завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации документов: {e}")
            return False
    
    def get_stats(self) -> Dict[str, any]:
        """
        Возвращает статистику по RAG-системе.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.vectorstore.get_stats()
        stats.update({
            "is_loaded": self.is_loaded,
            "embed_model": "Yandex " + self.embedder.model,
            "chat_model": YANDEX_GPT_MODEL,
            "vision_model": "Не настроен",
            "api_provider": "Yandex Cloud"
        })
        return stats

