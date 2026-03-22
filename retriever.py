"""
xcore_ai/retriever.py
----------------------
Recherche sémantique dans la base vectorielle ChromaDB.
Détecte l'intention de la requête pour adapter le contexte
injecté dans le prompt.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback si langchain-chroma pas installé

from config import XCoreAIConfig
from embeddings import XCoreEmbeddings


# ── Résultat de recherche enrichi ─────────────────────────

class RetrievalResult:
    def __init__(self, documents: list, scores: list[float]):
        self.documents = documents
        self.scores = scores

    def build_context(self, max_chars: int = 6000) -> str:
        """
        Construit le bloc de contexte injecté dans le prompt.
        Tronque si le contexte dépasse max_chars.
        """
        parts = []
        total = 0
        for doc, score in zip(self.documents, self.scores):
            meta = doc.metadata
            header = (
                f"# Source: {meta.get('relative_path', meta.get('source', '?'))}"
                f" (score={score:.2f})"
            )
            if meta.get("classes"):
                header += f"\n# Classes: {meta['classes']}"
            if meta.get("functions"):
                header += f"\n# Functions: {meta['functions']}"
            block = f"{header}\n\n{doc.page_content}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    parts.append(block[:remaining] + "\n... [tronqué]")
                break
            parts.append(block)
            total += len(block)
        return "\n\n" + ("─" * 60) + "\n\n".join(parts)

    def source_list(self) -> list[str]:
        seen = set()
        sources = []
        for doc in self.documents:
            s = doc.metadata.get("relative_path", doc.metadata.get("source", "?"))
            if s not in seen:
                seen.add(s)
                sources.append(s)
        return sources

    def __len__(self):
        return len(self.documents)


# ── Détection d'intention ─────────────────────────────────

def detect_intent(query: str, keywords: dict) -> str:
    q_lower = query.lower()
    for intent, kws in keywords.items():
        if any(kw in q_lower for kw in kws):
            return intent
    return "general"


# ── Retriever principal ───────────────────────────────────

class XCoreRetriever:

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self._db: Optional[Chroma] = None
        self._embedding_fn = XCoreEmbeddings(config.embedding_model)

    def _load_db(self) -> Chroma:
        if self._db is None:
            db_path = self.config.vector_db_path
            if not Path(db_path).exists():
                raise RuntimeError(
                    f"Index introuvable : {db_path}\n"
                    "Lance d'abord : xcore-ai index"
                )
            self._db = Chroma(
                persist_directory=db_path,
                collection_name=self.config.collection_name,
                embedding_function=self._embedding_fn,
            )
        return self._db

    def search(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """
        Recherche sémantique + filtrage par score.
        Adapte k selon l'intention détectée.
        """
        intent = detect_intent(query, self.config.intent_keywords)

        # Ajuste k selon l'intention
        k_map = {
            "generate": self.config.retrieval_k + 2,   # plus de contexte pour générer
            "explain": self.config.retrieval_k + 1,
            "debug": self.config.retrieval_k,
            "list": max(2, self.config.retrieval_k - 2),
            "architecture": self.config.retrieval_k,
            "general": self.config.retrieval_k,
        }
        effective_k = max(2, k or k_map.get(intent, self.config.retrieval_k))

        db = self._load_db()

        # Recherche avec scores
        results = db.similarity_search_with_score(query, k=effective_k)

        # Filtre par seuil de score (distance cosine → score bas = plus proche)
        filtered = [
            (doc, score)
            for doc, score in results
            if score <= (1.0 - self.config.retrieval_score_threshold)
        ]

        # Fallback : garde au moins 2 résultats même si score faible
        if len(filtered) < 2 and results:
            filtered = results[:2]

        docs = [d for d, _ in filtered]
        scores = [1.0 - s for _, s in filtered]  # convertit distance → similarité

        return RetrievalResult(documents=docs, scores=scores)

    def is_ready(self) -> bool:
        try:
            self._load_db()
            return True
        except Exception:
            return False