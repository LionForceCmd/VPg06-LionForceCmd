"""Тесты без ключей: python test_offline.py

1. Инструменты, не требующие ключей (кошки, собаки, погода), ходят в реальные API.
2. Логика памяти проверяется на заглушках: в Pinecone пишется ТОЛЬКО текст
   пользователя, а найденный контекст попадает в сообщения агента.
"""
import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("PINECONE_API_KEY", "test")

from haystack import Document
from haystack.dataclasses import ChatMessage

import tools
from bot import SYSTEM_PROMPT, Assistant


class FakeStore:
    def __init__(self):
        self.saved = []
        self.docs = []

    def save_user_message(self, user_id, text, username=None):
        self.saved.append((user_id, text))
        self.docs.append(Document(content=text))

    def search(self, user_id, query, top_k=5):
        return list(self.docs)

    def clear_user(self, user_id):
        self.docs.clear()


class FakeAgent:
    """Возвращает заготовленный ответ и запоминает, что получил на вход."""

    def __init__(self, answer):
        self.answer = answer
        self.last_messages = None

    def run(self, messages):
        self.last_messages = messages
        return {"last_message": ChatMessage.from_assistant(self.answer)}


def check(cond, msg):
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond:
        sys.exit(1)


# ---- 1. инструменты -------------------------------------------------------
fact = tools.get_cat_fact()
check(len(fact) > 10 and not fact.startswith("Ошибка"), f"get_cat_fact: {fact[:70]}...")

weather = tools.get_weather("Москва")
check(weather.startswith("Погода в") and "°C" in weather, f"get_weather: {weather}")
check("не найден" in tools.get_weather("ЫЫЫЫЫЫЫЫЫ"), "get_weather: неизвестный город обрабатывается")

tools.outbox.set(None)
res = tools.get_dog_image()
check(tools.outbox.get() and tools.outbox.get().startswith("http"), f"get_dog_image: {tools.outbox.get()}")

names = [t.name for t in tools.build_tools()]
check(names == ["get_cat_fact", "get_dog_image", "analyze_dog_image", "get_weather"], f"зарегистрированы тулзы: {names}")
check(all(n in SYSTEM_PROMPT for n in names), "системный промпт упоминает все инструменты")

# ---- 2. память ------------------------------------------------------------
store, agent = FakeStore(), FakeAgent("Привет!")
a = Assistant(agent, store)
a.handle(1, 100, "Меня зовут Алекс, я живу в Казани", "alex")
agent.answer = "Тебя зовут Алекс."
answer, image = a.handle(1, 100, "Как меня зовут?", "alex")

check(store.saved == [(1, "Меня зовут Алекс, я живу в Казани"), (1, "Как меня зовут?")],
      "в Pinecone записаны только сообщения пользователя")
check("Привет!" not in [t for _, t in store.saved] and "Тебя зовут Алекс." not in [t for _, t in store.saved],
      "ответы бота в Pinecone не попадают")
ctx = agent.last_messages[0].text
check("Меня зовут Алекс, я живу в Казани" in ctx, "найденный в Pinecone контекст передан агенту")
check(answer == "Тебя зовут Алекс." and image is None, "ответ агента возвращается пользователю")

a.clear(1, 100)
check(not store.docs and 100 not in a.history, "/clear чистит память и историю")
print("\nВсе проверки пройдены")
