"""
xcore_ai/embeddings.py
-----------------------
Wrapper LangChain pour SentenceTransformers.
Cache le modèle en mémoire (singleton par nom de modèle).
"""

from __future__ import annotations
from typing import List
from functools import lru_cache

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


# ── Cache global des modèles chargés ──────────────────────
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        print(f"  [embeddings] Chargement du modèle : {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class XCoreEmbeddings(Embeddings):
    """
    Embeddings LangChain-compatibles basés sur SentenceTransformers.

    Usage:
        emb = XCoreEmbeddings("all-MiniLM-L6-v2")
        vectors = emb.embed_documents(["def plugin(): ...", "class Service:"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = _get_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed une liste de documents (pour l'indexation)."""
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed une requête unique (pour la recherche)."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def __repr__(self):
        return f"XCoreEmbeddings(model={self.model_name!r})"