# Персональный ассистент в Telegram (Haystack + Pinecone)

Домашнее задание VPg06 (Кейс 2). Агент на Haystack с памятью в Pinecone и инструментами.

## Инструменты агента
| Tool | Что делает | API |
|---|---|---|
| `get_cat_fact` | случайный факт о кошках | catfact.ninja |
| `get_dog_image` | случайная картинка собаки | dog.ceo |
| `analyze_dog_image` | картинка собаки → OpenAI vision: порода и её история; картинка уходит в Telegram с подписью | dog.ceo + OpenAI |
| **`get_weather`** (новый) | текущая погода в городе | Open-Meteo (без ключа) |

Системный промпт (`bot.py`, `SYSTEM_PROMPT`) описывает все инструменты, в том числе погоду,
и требует вызывать `get_weather`, а не выдумывать данные.

## Память
`context_store.py`: `PineconeDocumentStore` (cosine) + `OpenAIDocumentEmbedder` / `OpenAITextEmbedder`
+ `PineconeEmbeddingRetriever`. Единственный метод записи — `save_user_message`: в Pinecone
попадает **только текст сообщений пользователя** (ответы бота и результаты инструментов — нет).
Перед каждым ответом бот ищет похожие прошлые сообщения этого пользователя и передаёт их агенту
как «Контекст из памяти». Всё общение с OpenAI идёт через `OPENAI_BASE_URL`.

## Запуск
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # впишите ключи
python bot.py
```
Команды: `/start`, `/help`, `/clear` (стереть историю и память о пользователе).

## Тесты
`python test_offline.py` — реальные вызовы бесплатных API и проверка логики памяти на заглушках (ключи не нужны).

## Примеры для проверки в Telegram

1. «Расскажи факт о кошках» → `get_cat_fact`
2. «Проанализируй картинку собаки» → фото + подпись с породой
3. «Какая погода в Берлине?» → `get_weather`; «а у меня в городе?» → город берётся из памяти

## Лицензия

Все права защищены (All rights reserved).

Любое использование этого проекта, в том числе коммерческое или некоммерческое, запрещено.
