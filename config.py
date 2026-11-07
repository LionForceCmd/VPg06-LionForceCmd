"""Настройки из .env. Всё обращение к OpenAI идёт через OPENAI_BASE_URL."""
import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "haystack-assistant")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "assistant")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Сколько прошлых сообщений пользователя подмешивать как контекст
CONTEXT_TOP_K = int(os.getenv("CONTEXT_TOP_K", "5"))
# Сколько последних реплик диалога держать в краткосрочной памяти
SHORT_TERM_MESSAGES = int(os.getenv("SHORT_TERM_MESSAGES", "10"))
