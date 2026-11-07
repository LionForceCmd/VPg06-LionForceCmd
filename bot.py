"""Персональный ассистент в Telegram: Haystack Agent + Pinecone + инструменты."""
from __future__ import annotations

import sys
import threading
from collections import defaultdict, deque

import telebot
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from loguru import logger

import config
import tools
from context_store import ContextStore

SYSTEM_PROMPT = """Ты — дружелюбный персональный ассистент в Telegram. Отвечай кратко, \
на языке пользователя, как настоящий помощник: учитывай контекст разговора и то, \
что пользователь говорил раньше.

Тебе доступны инструменты — вызывай их, когда пользователь об этом просит:
- get_cat_fact — случайный факт о кошках («расскажи факт о кошках»);
- get_dog_image — просто показать случайную картинку собаки;
- analyze_dog_image — получить картинку собаки, определить породу и рассказать её \
историю (картинка отправляется пользователю автоматически);
- get_weather(city) — текущая погода в городе («какая погода в Москве?»). Если город \
не назван, возьми его из контекста прошлых сообщений, а если там его нет — спроси.

Правила:
- Не выдумывай погоду и факты — для них всегда вызывай инструмент.
- Раздел «Контекст из памяти» ниже — это то, что пользователь писал раньше. \
Опирайся на него, когда вопрос касается личных данных или прошлых разговоров.
- Если в памяти нет ответа, честно скажи, что не знаешь."""

HELP_TEXT = (
    "Я персональный ассистент на Haystack. Умею:\n"
    "• факт о кошках 🐱\n"
    "• проанализировать случайную картинку собаки 🐶 (порода и история)\n"
    "• рассказать погоду в любом городе ☀️\n"
    "• помнить, что ты мне писал, и отвечать по этому контексту\n\n"
    "Команды: /start, /help, /clear (стереть память и историю)."
)


def build_agent() -> Agent:
    return Agent(
        chat_generator=OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            api_base_url=config.OPENAI_BASE_URL,
            model=config.OPENAI_CHAT_MODEL,
        ),
        tools=tools.build_tools(),
        system_prompt=SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=8,
    )


class Assistant:
    """Логика ассистента отдельно от Telegram -- её удобно тестировать."""

    def __init__(self, agent, store):
        self.agent = agent
        self.store = store
        self._lock = threading.Lock()  # один запрос агента за раз (общий слот outbox)
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SHORT_TERM_MESSAGES))

    def handle(self, user_id: int, chat_id: int, text: str, username: str | None = None):
        """Возвращает (ответ, url_картинки | None)."""
        # 1. Достаём контекст из Pinecone ДО записи (чтобы не находить само сообщение)
        try:
            docs = self.store.search(user_id, text, top_k=config.CONTEXT_TOP_K)
        except Exception:
            logger.exception("Ошибка чтения контекста из Pinecone")
            docs = []
        memories = [d.content for d in docs]

        # 2. Сохраняем в Pinecone ТОЛЬКО сообщение пользователя
        try:
            self.store.save_user_message(user_id, text, username)
        except Exception:
            logger.exception("Ошибка записи в Pinecone")

        # 3. Запускаем агента
        context = "Контекст из памяти (прошлые сообщения пользователя):\n" + (
            "\n".join(f"- {m}" for m in memories) if memories else "(пусто)"
        )
        messages = [ChatMessage.from_system(context), *self.history[chat_id], ChatMessage.from_user(text)]
        with self._lock:
            tools.outbox.set(None)
            result = self.agent.run(messages=messages)
            answer = result["last_message"].text or "…"
            image = tools.outbox.get()
            tools.outbox.set(None)

        # Краткосрочная память: только текст диалога (в Pinecone это НЕ попадает)
        self.history[chat_id].append(ChatMessage.from_user(text))
        self.history[chat_id].append(ChatMessage.from_assistant(answer))
        return answer, image

    def clear(self, user_id: int, chat_id: int) -> None:
        self.history.pop(chat_id, None)
        self.store.clear_user(user_id)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN не найден в .env")
    assistant = Assistant(build_agent(), ContextStore())
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

    @bot.message_handler(commands=["start", "help"])
    def on_start(m):
        bot.reply_to(m, HELP_TEXT)

    @bot.message_handler(commands=["clear"])
    def on_clear(m):
        assistant.clear(m.from_user.id, m.chat.id)
        bot.reply_to(m, "Готово: история и память о тебе очищены.")

    @bot.message_handler(func=lambda m: bool(m.text) and not m.text.startswith("/"))
    def on_text(m):
        bot.send_chat_action(m.chat.id, "typing")
        try:
            answer, image = assistant.handle(m.from_user.id, m.chat.id, m.text, m.from_user.username)
        except Exception:
            logger.exception("Ошибка обработки сообщения")
            bot.reply_to(m, "Извини, что-то пошло не так. Попробуй ещё раз.")
            return
        if image:
            # картинка + подпись (Telegram ограничивает caption 1024 символами)
            bot.send_photo(m.chat.id, image, caption=answer[:1024], reply_to_message_id=m.message_id)
            if len(answer) > 1024:
                bot.send_message(m.chat.id, answer[1024:])
        else:
            bot.reply_to(m, answer)

    logger.info("Бот запущен, индекс Pinecone: {}", config.PINECONE_INDEX_NAME)
    bot.infinity_polling()


if __name__ == "__main__":
    main()
