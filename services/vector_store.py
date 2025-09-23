from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import hashlib

import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions


DEFAULT_DB_DIR = os.environ.get("CHROMA_DB_DIR", os.path.abspath("./chroma_db"))
DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "doc_summaries")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(
        self,
        persist_directory: str = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        _ensure_dir(persist_directory)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model_name
        )
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        if not ids:
            ids = [
                _sha1(f"{m.get('source_file','')}-{m.get('modality','')}-{i}-{doc[:64]}")
                for i, (doc, m) in enumerate(zip(documents, metadatas))
            ]
        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        results = self.collection.query(
            query_texts=[query], n_results=k, where=where or {}
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return docs, metas, ids

    def reset(self) -> None:
        # DANGER: Deletes and recreates the collection
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=self.embedder
        )

