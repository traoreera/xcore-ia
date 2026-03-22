"""
xcore_ai/config.py
------------------
Configuration centrale de XCore AI.
Chargeable depuis un fichier YAML ou via l'environnement.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class XCoreAIConfig:
    # ── Chemins ────────────────────────────────────────────
    framework_path: str = "./xcore"
    vector_db_path: str = "./vector_db"
    collection_name: str = "xcore_Ia"
    history_path: str = "./xcore_ai_history.json"

    # ── Modèles ────────────────────────────────────────────
    llm_model: str = "novaforgeai/deepseek-coder:6.7b-optimized"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"

    # ── RAG ────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_k: int = 6          # nb de chunks récupérés
    retrieval_score_threshold: float = 0.35

    # ── LLM ────────────────────────────────────────────────
    llm_temperature: float = 0.1  # bas = déterministe pour du code
    llm_max_tokens: int = 2048
    llm_timeout: int = 1200        # secondes

    # ── UI ─────────────────────────────────────────────────
    show_sources: bool = True
    show_retrieval_score: bool = False
    max_history: int = 20         # nb de tours gardés en mémoire

    # ── Prompt système ─────────────────────────────────────
    system_prompt: str = field(default_factory=lambda: (
        "You are XCore AI, an expert assistant specialized exclusively in the XCore Python framework.\n"
        "You have deep knowledge of XCore's plugin system, service injection, decorators, and architecture.\n"
        "When asked to generate code, always use XCore patterns: @plugin, @service, get_service, BasePlugin, BaseService.\n"
        "Generate clean, production-ready Python code. Include docstrings. Use type hints.\n"
        "If the question is not about XCore or Python development, politely redirect to XCore topics.\n"
        "Always base your answers on the provided context from the XCore source code."
    ))

    # ── Classement des intentions ──────────────────────────
    intent_keywords: dict = field(default_factory=lambda: {
        "generate": ["create", "génère", "make", "build", "écris", "write", "crée", "implement"],
        "explain": ["explain", "explique", "how", "comment", "what", "qu'est", "describe", "montre"],
        "debug": ["error", "erreur", "bug", "fix", "corrige", "doesn't work", "marche pas", "exception"],
        "list": ["list", "liste", "all", "tous", "available", "disponible", "show me all"],
        "architecture": ["architecture", "structure", "diagram", "schéma", "overview", "overview"],
    })

    @classmethod
    def from_yaml(cls, path: str) -> "XCoreAIConfig":
        """Charge la config depuis un fichier YAML."""
        if not HAS_YAML:
            raise ImportError("PyYAML requis : pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_env(cls) -> "XCoreAIConfig":
        """Surcharge les valeurs depuis les variables d'environnement XCORE_AI_*."""
        cfg = cls()
        mapping = {
            "XCORE_AI_FRAMEWORK_PATH": "framework_path",
            "XCORE_AI_DB_PATH": "vector_db_path",
            "XCORE_AI_LLM_MODEL": "llm_model",
            "XCORE_AI_EMBEDDING_MODEL": "embedding_model",
            "XCORE_AI_OLLAMA_URL": "ollama_base_url",
            "XCORE_AI_RETRIEVAL_K": ("retrieval_k", int),
            "XCORE_AI_TEMPERATURE": ("llm_temperature", float),
        }
        for env_key, attr in mapping.items():
            val = os.environ.get(env_key)
            if val is None:
                continue
            if isinstance(attr, tuple):
                attr_name, cast = attr
                setattr(cfg, attr_name, cast(val))
            else:
                setattr(cfg, attr, val)
        return cfg

    def to_yaml(self, path: str):
        """Sauvegarde la config courante en YAML."""
        if not HAS_YAML:
            raise ImportError("PyYAML requis : pip install pyyaml")
        import dataclasses
        data = {k: v for k, v in dataclasses.asdict(self).items()
                if not callable(v)}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)