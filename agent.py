"""
xcore_ai/agent.py
------------------
Agent principal : orchestre le RAG + LLM + mémoire de conversation.
Point d'entrée pour toute interaction avec XCore AI.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Generator, Optional

from config import XCoreAIConfig
from retriever import XCoreRetriever, RetrievalResult, detect_intent
from llm import XCoreLLM, build_prompt


# ── Réponse structurée ────────────────────────────────────

class AgentResponse:
    def __init__(
        self,
        text: str,
        sources: list[str],
        intent: str,
        retrieval: Optional[RetrievalResult] = None,
    ):
        self.text = text
        self.sources = sources
        self.intent = intent
        self.retrieval = retrieval

    def __str__(self):
        return self.text


# ── Agent ─────────────────────────────────────────────────

class XCoreAgent:

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self.retriever = XCoreRetriever(config)
        self.llm = XCoreLLM(config)
        self._history: list[dict] = []  # [{"user": ..., "assistant": ...}]
        self._load_history()

    # ── Historique persistant ──────────────────────────────
    def _load_history(self):
        p = Path(self.config.history_path)
        if p.exists():
            try:
                with open(p) as f:
                    self._history = json.load(f)
                # Garde seulement les N derniers tours
                self._history = self._history[-self.config.max_history:]
            except Exception:
                self._history = []

    def _save_history(self):
        try:
            # Crée le dossier parent si nécessaire
            Path(self.config.history_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.history_path, "w") as f:
                json.dump(self._history[-self.config.max_history:], f, indent=2)
        except Exception:
            pass

    def clear_history(self):
        self._history = []
        p = Path(self.config.history_path)
        if p.exists():
            p.unlink()

    def history_summary(self) -> str:
        if not self._history:
            return "Aucun historique."
        lines = []
        for i, turn in enumerate(self._history[-5:], 1):
            lines.append(f"  [{i}] Vous: {turn['user'][:60]}...")
        return "\n".join(lines)

    # ── Vérifications pré-requête ──────────────────────────
    def check_ready(self) -> tuple[bool, str]:
        """Vérifie que le RAG et le LLM sont disponibles."""
        if not self.retriever.is_ready():
            return False, (
                "Index vectoriel introuvable.\n"
                "Lance d'abord : xcore-ai index"
            )
        ok, msg = self.llm.is_available()
        if not ok:
            return False, msg
        return True, "OK"

    # ── Requête principale (streaming) ────────────────────
    def ask_stream(self, query: str) -> Generator[str, None, str]:
        """
        Traite une question et streame la réponse token par token.
        Usage:
            for token in agent.ask_stream("create a plugin"):
                print(token, end="", flush=True)
        Retourne la réponse complète à la fin (via StopIteration.value).
        """
        # 1. Récupère le contexte pertinent
        retrieval = self.retriever.search(query)

        # 2. Construit le contexte textuel
        context = retrieval.build_context(max_chars=5000)

        # 3. Construit les messages (system + history + query)
        messages = build_prompt(
            system=self.config.system_prompt,
            context=context,
            history=self._history[-(self.config.max_history // 2):],
            query=query,
        )

        # 4. Stream la réponse
        full_response = ""
        for token in self.llm.stream(messages):
            full_response += token
            yield token

        # 5. Sauvegarde dans l'historique
        self._history.append({
            "user": query,
            "assistant": full_response,
        })
        self._save_history()

        return full_response

    # ── Requête complète (non-streaming) ──────────────────
    def ask(self, query: str) -> AgentResponse:
        """Version synchrone pour usage programmatique."""
        retrieval = self.retriever.search(query)
        context = retrieval.build_context()
        messages = build_prompt(
            system=self.config.system_prompt,
            context=context,
            history=self._history[-(self.config.max_history // 2):],
            query=query,
        )
        response_text = self.llm.generate(messages)

        self._history.append({
            "user": query,
            "assistant": response_text,
        })
        self._save_history()

        # CORRIGÉ : detect_intent importé en tête de fichier, plus de re-import relatif erroné
        intent = detect_intent(query, self.config.intent_keywords)

        return AgentResponse(
            text=response_text,
            sources=retrieval.source_list(),
            intent=intent,
            retrieval=retrieval,
        )