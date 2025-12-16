"""
Модуль для работы с Yandex GPT API.
"""

import logging
from typing import List, Dict
import requests
from config import YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_GPT_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

class YandexGPT:
    """
    Класс для взаимодействия с Yandex GPT API.
    
    Заменяет OpenAI GPT в RAG-системе.
    """
    
    def __init__(self):
        """
        Инициализация Yandex GPT клиента.
        """
        if not YANDEX_API_KEY:
            raise ValueError("YANDEX_API_KEY не установлен в конфигурации")
        if not YANDEX_FOLDER_ID:
            raise ValueError("YANDEX_FOLDER_ID не установлен в конфигурации")
        
        self.api_key = YANDEX_API_KEY
        self.folder_id = YANDEX_FOLDER_ID
        self.model_uri = f"gpt://{self.folder_id}/{YANDEX_GPT_MODEL}"
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id
        }
        
        logger.info(f"YandexGPT инициализирован с моделью: {YANDEX_GPT_MODEL}")
        logger.info(f"Model URI: {self.model_uri}")
    
    def generate_completion(self, messages: List[Dict[str, str]], 
                           temperature: float = 0.7, 
                           max_tokens: int = 1000) -> str:
        """
        Генерирует ответ на основе истории сообщений.
        
        Args:
            messages: Список сообщений в формате [{"role": "user", "content": "текст"}]
            temperature: Креативность ответа (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Текст ответа от модели
        """
        try:
            # Преобразуем сообщения в формат Yandex API
            yandex_messages = []
            for msg in messages:
                # Yandex API не поддерживает системные сообщения напрямую
                # Добавляем системный промпт как первое пользовательское сообщение
                if msg["role"] == "system":
                    yandex_messages.append({
                        "role": "user",
                        "text": f"Системная инструкция: {msg['content']}"
                    })
                else:
                    yandex_messages.append({
                        "role": msg["role"],
                        "text": msg["content"]
                    })
            
            # Формируем запрос
            payload = {
                "modelUri": self.model_uri,
                "completionOptions": {
                    "temperature": temperature,
                    "maxTokens": max_tokens
                },
                "messages": yandex_messages
            }
            
            logger.debug(f"Отправка запроса к Yandex GPT: {payload}")
            
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
            
            # Извлекаем текст ответа
            if "result" in data and "alternatives" in data["result"]:
                answer = data["result"]["alternatives"][0]["message"]["text"]
                logger.info(f"Ответ получен от Yandex GPT, длина: {len(answer)} символов")
                return answer
            else:
                logger.error(f"Неожиданный формат ответа: {data}")
                raise ValueError("Неверный формат ответа от Yandex API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса к Yandex GPT: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise
    
    def process_image(self, image_url: str, query: str = None) -> Dict:
        """
        Обработка изображений (заглушка).
        
        Note: Yandex имеет свой Vision API, но для простоты сделаем заглушку.
        В реальном проекте нужно использовать yandex.cloud/vision
        
        Args:
            image_url: URL изображения
            query: Вопрос пользователя (опционально)
            
        Returns:
            Словарь с результатами обработки
        """
        logger.warning("Vision API для Yandex не реализован. Используйте заглушку.")
        
        # Возвращаем заглушку
        return {
            "extracted_text": "Функция распознавания текста с изображений временно недоступна.",
            "rag_answer": None if not query else "Для ответа на вопрос требуется функционал распознавания текста.",
            "error": "Vision API не настроен"
        }

