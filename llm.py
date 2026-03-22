"""
xcore_ai/llm.py
----------------
Client Ollama avec streaming, gestion du timeout
et vérification de disponibilité du modèle.
"""

from __future__ import annotations
import json
from typing import Generator, Optional

import requests

from config import XCoreAIConfig


# ── Prompt builder ────────────────────────────────────────

def build_prompt(
    system: str,
    context: str,
    history: list[dict],
    query: str,
) -> list[dict]:
    """
    Construit les messages pour l'API Ollama /api/chat.
    Format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    messages = [{"role": "system", "content": system}]

    # Contexte RAG injecté comme message système secondaire
    if context.strip():
        messages.append({
            "role": "system",
            "content": (
                "Voici des extraits du code source du framework XCore "
                "qui sont pertinents pour répondre à la question :\n\n"
                f"{context}\n\n"
                "Base ta réponse sur ces extraits."
            ),
        })

    # Historique de conversation (sliding window)
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # Question actuelle
    messages.append({"role": "user", "content": query})

    return messages


# ── Client LLM ────────────────────────────────────────────

class XCoreLLM:

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self.base_url = config.ollama_base_url.rstrip("/")

    # ── Vérification disponibilité ─────────────────────────
    def is_available(self) -> tuple[bool, str]:
        """Vérifie qu'Ollama tourne et que le modèle est disponible."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model = self.config.llm_model
            # Cherche une correspondance partielle (ex: "deepseek-coder:6.7b")
            found = any(model in m or m in model for m in models)
            if not found:
                available = ", ".join(models) if models else "aucun"
                return False, (
                    f"Modèle '{model}' introuvable.\n"
                    f"Modèles disponibles : {available}\n"
                    f"Lance : ollama pull {model}"
                )
            return True, "OK"
        except requests.ConnectionError:
            return False, (
                f"Ollama inaccessible sur {self.base_url}\n"
                "Lance : ollama serve"
            )
        except Exception as e:
            return False, f"Erreur Ollama : {e}"

    # ── Génération avec streaming ──────────────────────────
    def stream(
        self,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream la réponse token par token depuis Ollama.
        Yields des fragments de texte.
        """
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
                "stop": ["<|EOT|>", "```\n\n```"],
            },
        }

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.config.llm_timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.Timeout:
            yield f"\n\n[Timeout après {self.config.llm_timeout}s — augmente llm_timeout dans la config]"
        except requests.RequestException as e:
            yield f"\n\n[Erreur réseau : {e}]"

    # ── Génération complète (non-streaming) ───────────────
    def generate(self, messages: list[dict]) -> str:
        """Retourne la réponse complète (pour usage programmatique)."""
        return "".join(self.stream(messages))