"""Долговременная память ассистента: Pinecone через интеграцию Haystack.

В Pinecone сохраняется ТОЛЬКО текст сообщений, отправленных пользователем.
Ответы бота, результаты инструментов и служебные сообщения не пишутся --
для этого в классе единственный метод записи ``save_user_message``.
"""
from __future__ import annotations

import time
from typing import Optional

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

import config


class ContextStore:
    def __init__(self, document_store=None, doc_embedder=None, text_embedder=None):
        self.document_store = document_store or PineconeDocumentStore(
            api_key=Secret.from_env_var("PINECONE_API_KEY"),
            index=config.PINECONE_INDEX_NAME,
            namespace=config.PINECONE_NAMESPACE,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",  # косинусное сходство
            spec={"serverless": {"region": config.PINECONE_REGION, "cloud": config.PINECONE_CLOUD}},
        )
        api_key = Secret.from_env_var("OPENAI_API_KEY")
        self.doc_embedder = doc_embedder or OpenAIDocumentEmbedder(
            api_key=api_key,
            api_base_url=config.OPENAI_BASE_URL,
            model=config.OPENAI_EMBEDDING_MODEL,
            meta_fields_to_embed=[],
        )
        self.text_embedder = text_embedder or OpenAITextEmbedder(
            api_key=api_key,
            api_base_url=config.OPENAI_BASE_URL,
            model=config.OPENAI_EMBEDDING_MODEL,
        )

        # Индексирующий pipeline: Document -> embedding -> Pinecone
        self.indexing = Pipeline()
        self.indexing.add_component("embedder", self.doc_embedder)
        self.indexing.add_component(
            "writer", DocumentWriter(self.document_store, policy=DuplicatePolicy.OVERWRITE)
        )
        self.indexing.connect("embedder.documents", "writer.documents")

        # Поисковый pipeline: вопрос -> embedding -> Pinecone (cosine)
        self.retriever = PineconeEmbeddingRetriever(document_store=self.document_store)
        self.querying = Pipeline()
        self.querying.add_component("embedder", self.text_embedder)
        self.querying.add_component("retriever", self.retriever)
        self.querying.connect("embedder.embedding", "retriever.query_embedding")

    # ------------------------------------------------------------ запись
    def save_user_message(self, user_id: int, text: str, username: Optional[str] = None) -> None:
        """Сохраняет текст сообщения пользователя. Единственный путь записи в Pinecone."""
        text = (text or "").strip()
        if not text:
            return
        meta = {"type": "user_message", "user_id": user_id, "timestamp": int(time.time())}
        if username:
            meta["username"] = username
        doc = Document(content=text, meta=meta)
        logger.info("[pinecone] save user={} text={!r}", user_id, text[:80])
        self.indexing.run({"embedder": {"documents": [doc]}})

    # ------------------------------------------------------------ чтение
    def search(self, user_id: int, query: str, top_k: int = 5) -> list[Document]:
        """Ближайшие по косинусному сходству прошлые сообщения этого пользователя."""
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "user_message"},
                {"field": "meta.user_id", "operator": "==", "value": user_id},
            ],
        }
        out = self.querying.run({
            "embedder": {"text": query},
            "retriever": {"filters": filters, "top_k": top_k},
        })
        docs = out["retriever"]["documents"]
        logger.info("[pinecone] search user={} query={!r} -> {} docs", user_id, query[:60], len(docs))
        return docs

    def clear_user(self, user_id: int) -> None:
        """Удаляет все сообщения пользователя (команда /clear)."""
        self.document_store.delete_by_filter(filters={
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "user_message"},
                {"field": "meta.user_id", "operator": "==", "value": user_id},
            ],
        })
