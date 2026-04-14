"""
xcore_ai/dataset/builder.py
-----------------------------
Construit un dataset d'entraînement pour le fine-tuning LoRA
de DeepSeek Coder sur le framework XCore.

Format du dataset :
{"conversations": [
    {
        "role": "system", 
        "content": ...
    },
    {
        "role": "user", 
        "content": ...}, 
    
    {   
        "role": "assistant", 
        "content": ...
    }


Sources automatiques :
    1. Fichiers Python du framework (patterns extraits par AST)
    2. Exemples écrits à la main (examples/*.py)
    3. Paires générées par le LLM (augmentation de données)
"""

from __future__ import annotations
import ast
import json
import re
from pathlib import Path

from config import XCoreAIConfig
from indexer import load_source_files


# ──────────────────────────────────────────────────────────
# System prompt XCore
# ──────────────────────────────────────────────────────────

XCORE_SYSTEM_PROMPT = """You are an expert developer of the XCore Python framework.
XCore uses decorator-based dependency injection: @plugin for plugins, @service for singleton services.
Always write idiomatic XCore code with proper type hints and docstrings."""


# ──────────────────────────────────────────────────────────
# Templates d'instructions
# ──────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATES = {
    "plugin": [
        "Create a {name} plugin using XCore",
        "Write a plugin called {name} with dependency injection",
        "Implement a {name} plugin that uses {deps}",
        "Generate an XCore plugin for {name}",
    ],
    "service": [
        "Create a {name} service for XCore",
        "Write a singleton service called {name}",
        "Implement a {name} service using @service decorator",
        "Generate an injectable {name} service for XCore",
    ],
    "app": [
        "Bootstrap an XCore application with {plugins}",
        "Write the main.py for an XCore app using {plugins}",
        "Create an XCoreApp that loads {plugins}",
    ],
    "explain": [
        "Explain how {name} works in XCore",
        "How do I use {name} in XCore?",
        "What does {name} do in the XCore framework?",
        "Describe the {name} pattern in XCore",
    ],
}


# ──────────────────────────────────────────────────────────
# Extraction de paires depuis le code source
# ──────────────────────────────────────────────────────────

class ASTPatternExtractor:
    """
    Parcourt l'AST Python pour extraire des paires
    (instruction → code) à partir du framework existant.
    """

    def extract_from_file(self, source: str, filepath: str) -> list[dict]:
        pairs = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return pairs

        for node in ast.walk(tree):
            # Extrait chaque classe décorée @plugin ou @service
            if isinstance(node, ast.ClassDef):
                decorators = [
                    d.func.id if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                    else d.id if isinstance(d, ast.Name)
                    else None
                    for d in node.decorator_list
                ]
                decorators = [d for d in decorators if d]

                if "plugin" in decorators or "service" in decorators:
                    kind = "plugin" if "plugin" in decorators else "service"
                    name = node.name.replace("Plugin", "").replace("Service", "").lower()
                    code = self._extract_class_source(source, node)

                    # Génère plusieurs instructions pour le même code
                    for tpl in INSTRUCTION_TEMPLATES[kind][:2]:
                        instruction = tpl.format(
                            name=name,
                            deps=self._guess_deps(node),
                            plugins=name,
                        )
                        pairs.append({
                            "instruction": instruction,
                            "input": "",
                            "output": code,
                            "source_file": filepath,
                            "type": kind,
                        })

        return pairs

    def _extract_class_source(self, source: str, node: ast.ClassDef) -> str:
        """Extrait le source exact d'une classe depuis les numéros de ligne AST."""
        lines = source.splitlines()
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") else start + 30
        return "\n".join(lines[start:end])

    def _guess_deps(self, class_node: ast.ClassDef) -> str:
        """Devine les dépendances injectées depuis les annotations __init__."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                args = [
                    arg.arg for arg in node.args.args
                    if arg.arg not in ("self", "config")
                ]
                return ", ".join(args) if args else "services"
        return "services"


# ──────────────────────────────────────────────────────────
# Augmentation par le LLM
# ──────────────────────────────────────────────────────────

AUGMENT_PROMPT = """You are a dataset builder for the XCore Python framework.
Given this XCore code example, generate 3 different natural language instructions
that someone might ask to get this code.
Output ONLY a JSON array of strings.
Example output: ["Create a cache plugin", "Write a TTL cache using XCore", "Make a caching service"]

Code:
{code}
"""


class LLMAugmenter:
    """Utilise le LLM pour générer des variantes d'instructions pour le même code."""

    def __init__(self, llm):
        self.llm = llm

    def augment(self, code: str) -> list[str]:
        """Génère 3 instructions alternatives pour un bloc de code."""
        messages = [
            {"role": "system", "content": "Output ONLY valid JSON array of strings."},
            {"role": "user", "content": AUGMENT_PROMPT.format(code=code[:800])},
        ]
        raw = self.llm.generate(messages)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(s) for s in result[:3]]
        except json.JSONDecodeError:
            pass
        return []


# ──────────────────────────────────────────────────────────
# Builder principal
# ──────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Construit le dataset complet pour le fine-tuning LoRA.

    Usage:
        builder = DatasetBuilder(config)
        builder.build(output_path="./dataset/xcore_train.jsonl", augment=True)
    """

    def __init__(self, config: XCoreAIConfig, llm=None):
        self.config = config
        self.extractor = ASTPatternExtractor()
        self.augmenter = LLMAugmenter(llm) if llm else None

    # ── Extraction depuis le framework ────────────────────
    def _mine_framework(self) -> list[dict]:
        pairs = []
        for doc in load_source_files(self.config.framework_path):
            if doc.metadata.get("language") != "python":
                continue
            extracted = self.extractor.extract_from_file(
                doc.page_content,
                doc.metadata.get("relative_path", "?"),
            )
            pairs.extend(extracted)
        return pairs

    # ── Exemples manuels ──────────────────────────────────
    def _load_examples(self, examples_dir: str) -> list[dict]:
        """
        Charge les exemples manuels depuis examples/*.jsonl ou examples/*.json.
        Supporte les deux formats : Alpaca et Conversations.
        Format Alpaca  : {"instruction": "...", "output": "..."}
        Format Conversations : {"conversations": [{...}, {...}, {...}]}
        """
        pairs = []
        root = Path(examples_dir)
        if not root.exists():
            return pairs

        for f in root.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            pairs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        for f in root.glob("*.json"):
            with open(f) as fh:
                try:
                    data = json.load(fh)
                    if isinstance(data, list):
                        pairs.extend(data)
                except json.JSONDecodeError:
                    pass

        return pairs

    # ── Déduplication ─────────────────────────────────────
    @staticmethod
    def _deduplicate(pairs: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for p in pairs:
            # Support des deux formats
            if "conversations" in p:
                msgs = p["conversations"]
                instruction = next(
                    (m["content"] for m in msgs if m["role"] == "user"), ""
                )
                output = next(
                    (m["content"] for m in msgs if m["role"] == "assistant"), ""
                )
            else:
                instruction = p.get("instruction", "")
                output = p.get("output", "")

            key = (instruction.lower().strip(), output[:100])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # ── Validation qualité ────────────────────────────────
    @staticmethod
    def _filter_quality(pairs: list[dict]) -> list[dict]:
        good = []
        for p in pairs:
            # Support des deux formats pendant la transition
            if "conversations" in p:
                msgs = p["conversations"]
                instruction = next(
                    (m["content"] for m in msgs if m["role"] == "user"), ""
                )
                output = next(
                    (m["content"] for m in msgs if m["role"] == "assistant"), ""
                )
            else:
                instruction = p.get("instruction", "")
                output = p.get("output", "")

            # Filtre les paires trop courtes ou sans code Python réel
            if (
                len(instruction) >= 10
                and len(output) >= 50
                and any(kw in output for kw in ("def ", "class ", "import "))
            ):
                good.append(p)
        return good

    # ── Conversion vers le format Conversations ───────────
    @staticmethod
    def _to_conversations(pairs: list[dict]) -> list[dict]:
        """
        Convertit toutes les paires vers le format Conversations unifié.
        Les paires déjà au bon format sont conservées telles quelles.
        """
        final = []
        for p in pairs:
            # Déjà au format Conversations → on s'assure juste que system est présent
            if "conversations" in p:
                msgs = p["conversations"]
                has_system = any(m["role"] == "system" for m in msgs)
                if not has_system:
                    msgs = [{"role": "system", "content": XCORE_SYSTEM_PROMPT}] + msgs
                final.append({"conversations": msgs})
            # Format Alpaca → conversion
            else:
                user_content = p.get("instruction", "")
                if p.get("input", "").strip():
                    user_content += f"\n\n{p['input']}"
                final.append({
                    "conversations": [
                        {"role": "system",    "content": XCORE_SYSTEM_PROMPT},
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": p.get("output", "")},
                    ]
                })
        return final

    # ── Build complet ─────────────────────────────────────
    def build(
        self,
        output_path: str = "./dataset/xcore_train.jsonl",
        examples_dir: str = "./examples",
        augment: bool = False,
        augment_top_n: int = 20,
    ) -> dict:
        """
        Construit et sauvegarde le dataset au format Conversations.

        Args:
            output_path:   Fichier de sortie (.jsonl)
            examples_dir:  Dossier d'exemples manuels
            augment:       Si True, utilise le LLM pour générer des variantes
            augment_top_n: Nombre de paires à augmenter

        Returns:
            Stats {"total": n, "from_framework": n, "from_examples": n, "augmented": n}
        """
        print("  [dataset] Mining du framework...")
        framework_pairs = self._mine_framework()
        print(f"  [dataset] {len(framework_pairs)} paires extraites du framework")

        print("  [dataset] Chargement des exemples manuels...")
        example_pairs = self._load_examples(examples_dir)
        print(f"  [dataset] {len(example_pairs)} exemples manuels")

        all_pairs = framework_pairs + example_pairs
        augmented_count = 0

        # Augmentation par LLM
        if augment and self.augmenter:
            print(f"  [dataset] Augmentation LLM sur {augment_top_n} paires...")
            top_pairs = all_pairs[:augment_top_n]
            for pair in top_pairs:
                # Récupère le code selon le format de la paire
                if "conversations" in pair:
                    msgs = pair["conversations"]
                    code = next(
                        (m["content"] for m in msgs if m["role"] == "assistant"), ""
                    )
                else:
                    code = pair.get("output", "")

                variants = self.augmenter.augment(code)
                for v in variants:
                    all_pairs.append({
                        "instruction": v,
                        "input": "",
                        "output": code,
                        "type": pair.get("type", "unknown") + "_augmented",
                    })
                augmented_count += len(variants)

        # Nettoyage
        all_pairs = self._deduplicate(all_pairs)
        all_pairs = self._filter_quality(all_pairs)

        # Conversion vers le format Conversations unifié
        final = self._to_conversations(all_pairs)

        # Sauvegarde JSONL
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for item in final:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Sauvegarde JSON lisible
        json_path = output.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

        stats = {
            "total": len(final),
            "from_framework": len(framework_pairs),
            "from_examples": len(example_pairs),
            "augmented": augmented_count,
            "output_jsonl": str(output),
            "output_json": str(json_path),
        }

        print(f"  [dataset] Dataset sauvegardé : {output}")
        print(f"  [dataset] Total : {len(final)} paires")

        return stats

    # ── Preview ───────────────────────────────────────────
    def preview(self, n: int = 5) -> list[dict]:
        """Retourne les n premières paires converties sans sauvegarder."""
        pairs = self._mine_framework()
        pairs = self._filter_quality(pairs)
        return self._to_conversations(pairs[:n])