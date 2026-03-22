"""
xcore_ai/generator.py
----------------------
Génère des projets XCore complets depuis une description.
L'IA génère chaque fichier un par un, les valide (syntaxe + exécution),
et les écrit sur le disque dans une structure cohérente.

Usage:
    gen = ProjectGenerator(config)
    gen.generate("create a REST API plugin with JWT auth and PostgreSQL")
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Generator

from config import XCoreAIConfig
from retriever import XCoreRetriever
from llm import XCoreLLM
from tools.executor import CodeExecutor, validate_syntax, extract_code_blocks


# ──────────────────────────────────────────────────────────
# Plan de génération
# ──────────────────────────────────────────────────────────

PLAN_PROMPT = """You are XCore AI. Given this project description, output ONLY a JSON array
of files to generate. Each item: {{"path": "relative/path.py", "description": "what it does"}}.
Include: services, plugins, main.py, and optionally tests.
Keep it to 4-6 files max. Output ONLY the JSON array, no markdown, no explanation.

Project: {description}

Framework context:
{context}
"""

CODEGEN_PROMPT = """You are XCore AI, expert in the XCore Python framework.
Generate the file: {path}
Description: {description}

Use these XCore patterns from the source:
{context}

Already generated files for context:
{already_generated}

Rules:
- Use plugin, service, get_service, , BasePlugin, BaseService from xcore_framework
- Include proper type hints and docstrings
- Output ONLY the Python code, no explanation, no markdown fences
"""


# ──────────────────────────────────────────────────────────
# Résultat de génération
# ──────────────────────────────────────────────────────────

class GeneratedFile:
    def __init__(self, path: str, code: str, valid: bool, error: str = ""):
        self.path = path
        self.code = code
        self.valid = valid
        self.error = error

    def write(self, base_dir: str):
        dest = Path(base_dir) / self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.code, encoding="utf-8")
        return dest


class GenerationResult:
    def __init__(self, project_name: str, output_dir: str):
        self.project_name = project_name
        self.output_dir = output_dir
        self.files: list[GeneratedFile] = []

    def add(self, f: GeneratedFile):
        self.files.append(f)

    @property
    def success_count(self):
        return sum(1 for f in self.files if f.valid)

    def summary(self) -> str:
        lines = [f"  Projet : {self.project_name}",
                 f"  Dossier : {self.output_dir}",
                 f"  Fichiers : {self.success_count}/{len(self.files)} valides"]
        for f in self.files:
            status = "OK" if f.valid else f"ERR: {f.error[:60]}"
            lines.append(f"    {'✓' if f.valid else '✗'} {f.path}  [{status}]")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Générateur principal
# ──────────────────────────────────────────────────────────

class ProjectGenerator:

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self.retriever = XCoreRetriever(config)
        self.llm = XCoreLLM(config)
        self.executor = CodeExecutor(
            timeout=10,
            framework_path=config.framework_path,
        )

    # ── Plan ──────────────────────────────────────────────
    def _plan(self, description: str) -> list[dict]:
        """Demande au LLM la liste des fichiers à générer."""
        retrieval = self.retriever.search(description, k=4)
        context = retrieval.build_context(max_chars=2000)

        prompt = PLAN_PROMPT.format(description=description, context=context)
        messages = [
            {"role": "system", "content": "Output ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ]

        raw = self.llm.generate(messages)

        # Nettoie le JSON (le LLM peut ajouter des backticks)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            plan = json.loads(raw)
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass

        # Fallback : structure minimale
        return [
            {"path": "services/app_service.py", "description": f"Main service for: {description}"},
            {"path": "plugins/app_plugin.py", "description": f"Main plugin for: {description}"},
            {"path": "main.py", "description": "XCoreApp bootstrap"},
        ]

    # ── Génération d'un fichier ────────────────────────────
    def _generate_file(
        self,
        path: str,
        description: str,
        already_generated: dict[str, str],
    ) -> GeneratedFile:
        """Génère un seul fichier avec contexte RAG + fichiers déjà générés."""
        retrieval = self.retriever.search(
            f"{description} xcore plugin service get_service", k=5
        )
        context = retrieval.build_context(max_chars=3000)

        # Résumé des fichiers déjà générés
        already_str = ""
        for prev_path, prev_code in already_generated.items():
            already_str += f"\n# {prev_path}\n{prev_code[:400]}...\n"

        prompt = CODEGEN_PROMPT.format(
            path=path,
            description=description,
            context=context,
            already_generated=already_str or "(aucun)",
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are XCore AI. Output ONLY Python code. "
                    "No markdown fences, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        code = self.llm.generate(messages).strip()

        # Retire les backticks si le LLM les a quand même ajoutés
        code = re.sub(r"^```(?:python|py)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)

        # Valide la syntaxe
        ok, error = validate_syntax(code)
        return GeneratedFile(path=path, code=code, valid=ok, error=error)

    # ── Génération du projet complet ──────────────────────
    def generate(
        self,
        description: str,
        output_dir: str | None = None,
        project_name: str | None = None,
    ) -> GenerationResult:
        """
        Génère un projet XCore complet depuis une description.

        Args:
            description: Description du projet en langage naturel
            output_dir:  Dossier de sortie (défaut: ./generated/<name>)
            project_name: Nom du projet (déduit si absent)

        Returns:
            GenerationResult avec tous les fichiers générés
        """
        # Déduit le nom du projet
        if not project_name:
            words = re.sub(r"[^a-zA-Z0-9 ]", "", description).split()
            project_name = "_".join(w.lower() for w in words[:3]) or "xcore_project"

        if not output_dir:
            output_dir = f"./generated/{project_name}"

        result = GenerationResult(project_name=project_name, output_dir=output_dir)

        # 1. Obtient le plan
        print(f"  [gen] Planification du projet '{project_name}'...")
        plan = self._plan(description)
        print(f"  [gen] Plan : {len(plan)} fichiers à générer")
        for item in plan:
            print(f"         · {item['path']}")

        # 2. Génère chaque fichier
        already_generated: dict[str, str] = {}
        for item in plan:
            path = item.get("path", "unknown.py")
            desc = item.get("description", description)
            print(f"\n  [gen] Génération : {path}")

            generated = self._generate_file(path, desc, already_generated)

            if generated.valid:
                # Écrit sur disque
                dest = generated.write(output_dir)
                already_generated[path] = generated.code
                print(f"         Syntaxe OK → {dest}")
            else:
                print(f"         Erreur syntaxe : {generated.error}")

            result.add(generated)

        # 3. Génère le README du projet
        self._write_project_readme(result, description)

        return result

    # ── Streaming pour le CLI ──────────────────────────────
    def generate_stream(
        self,
        description: str,
        output_dir: str | None = None,
    ) -> Generator[str, None, GenerationResult]:
        """Version streaming : yield des messages de progression."""
        words = re.sub(r"[^a-zA-Z0-9 ]", "", description).split()
        project_name = "_".join(w.lower() for w in words[:3]) or "xcore_project"

        if not output_dir:
            output_dir = f"./generated/{project_name}"

        result = GenerationResult(project_name=project_name, output_dir=output_dir)

        yield f"Planification du projet '{project_name}'...\n"
        plan = self._plan(description)
        yield f"Plan : {len(plan)} fichiers\n"
        for item in plan:
            yield f"  · {item['path']}\n"
        yield "\n"

        already_generated: dict[str, str] = {}
        for item in plan:
            path = item.get("path", "unknown.py")
            desc = item.get("description", description)
            yield f"Génération de {path}...\n"

            generated = self._generate_file(path, desc, already_generated)
            if generated.valid:
                generated.write(output_dir)
                already_generated[path] = generated.code
                yield f"  ✓ {path} — syntaxe OK\n"
            else:
                yield f"  ✗ {path} — {generated.error}\n"

            result.add(generated)

        self._write_project_readme(result, description)
        yield f"\nProjet généré dans : {output_dir}\n"

        return result

    # ── README du projet généré ───────────────────────────
    def _write_project_readme(self, result: GenerationResult, description: str):
        lines = [
            f"# {result.project_name}",
            "",
            f"> Généré par XCore AI",
            f"> Description : {description}",
            "",
            "## Fichiers générés",
            "",
        ]
        for f in result.files:
            status = "✓" if f.valid else "✗ (syntaxe invalide)"
            lines.append(f"- `{f.path}` {status}")

        lines += [
            "",
            "## Lancement",
            "",
            "```bash",
            "pip install -e ../../  # installe xcore_framework",
            "python main.py",
            "```",
        ]

        readme_path = Path(result.output_dir) / "README.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text("\n".join(lines), encoding="utf-8")