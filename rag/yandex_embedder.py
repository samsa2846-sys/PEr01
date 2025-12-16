"""
Модуль для создания эмбеддингов через Yandex API.
"""

import logging
from typing import List
import requests
import numpy as np
from config import YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_EMBED_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

class YandexEmbedder:
    """
    Класс для создания эмбеддингов текста с использованием Yandex API.
    
    Заменяет OpenAIEmbedder в RAG-системе.
    """
    
    def __init__(self):
        """
        Инициализация Yandex эмбеддера.
        """
        if not YANDEX_API_KEY:
            raise ValueError("YANDEX_API_KEY не установлен в конфигурации")
        if not YANDEX_FOLDER_ID:
            raise ValueError("YANDEX_FOLDER_ID не установлен в конфигурации")
        
        self.api_key = YANDEX_API_KEY
        self.folder_id = YANDEX_FOLDER_ID
        self.model = YANDEX_EMBED_MODEL
        self.model_uri = f"emb://{self.folder_id}/{self.model}"
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id
        }
        
        # Размерность эмбеддингов для Yandex text-search-doc
        self._dimension = 256  # Yandex возвращает 256-мерные векторы
        
        logger.info(f"YandexEmbedder инициализирован с моделью: {self.model}")
        logger.info(f"Model URI: {self.model_uri}")
        logger.info(f"Предполагаемая размерность: {self._dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создает эмбеддинг (векторное представление) для одного текста.
        
        Args:
            text: Текст для преобразования в вектор
            
        Returns:
            Список чисел с плавающей точкой - вектор эмбеддинга
        """
        try:
            # Обрезаем слишком длинные тексты (лимит Yandex API)
            if len(text) > 10000:
                text = text[:10000]
                logger.warning(f"Текст обрезан до 10000 символов")
            
            logger.debug(f"Создание эмбеддинга для текста длиной {len(text)} символов")
            
            # Формируем запрос
            payload = {
                "modelUri": self.model_uri,
                "text": text
            }
            
            # Отправляем запрос
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            # Проверяем ответ
            response.raise_for_status()
            data = response.json()
            
            # Извлекаем вектор
            if "embedding" in data:
                embedding = data["embedding"]
                # Обновляем размерность на основе реального ответа
                if len(embedding) != self._dimension:
                    self._dimension = len(embedding)
                    logger.info(f"Обновлена размерность эмбеддингов: {self._dimension}")
                logger.debug(f"Эмбеддинг создан, размерность: {len(embedding)}")
                return embedding
            else:
                logger.error(f"Неожиданный формат ответа: {data}")
                raise ValueError("Неверный формат ответа от Yandex API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса к Yandex Embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для нескольких текстов.
        
        Args:
            texts: Список текстов для преобразования
            
        Returns:
            Список векторов эмбеддингов
        """
        try:
            logger.info(f"Создание эмбеддингов для {len(texts)} текстов")
            
            embeddings = []
            for i, text in enumerate(texts, 1):
                logger.debug(f"Обработка текста {i}/{len(texts)} (длина: {len(text)} символов)")
                
                # Обрабатываем каждый текст отдельно
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            
            logger.info(f"Успешно создано {len(embeddings)} эмбеддингов")
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность векторов эмбеддингов.
        
        Returns:
            Размерность вектора (256 для Yandex text-search-doc)
        """
        return self._dimension
    
    def test_connection(self) -> bool:
        """
        Тестирует соединение с Yandex Embeddings API.
        
        Returns:
            True если соединение успешно
        """
        try:
            test_embedding = self.embed_text("test")
            if len(test_embedding) == self._dimension:
                logger.info(f"Соединение с Yandex Embeddings API успешно, размерность: {len(test_embedding)}")
                return True
            else:
                logger.warning(f"Неожиданная размерность: {len(test_embedding)}")
                return False
        except Exception as e:
            logger.error(f"Ошибка тестирования соединения: {e}")
            return False

