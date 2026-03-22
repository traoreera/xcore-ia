"""
xcore_ai/agentic.py
---------------------
Agent en boucle autonome : génère du code, l'exécute,
détecte les erreurs et se corrige automatiquement jusqu'à
3 tentatives. Combine RAG + exécution + auto-correction.

C'est la couche "Copilot" de XCore AI.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generator

from config import XCoreAIConfig
from retriever import XCoreRetriever
from llm import XCoreLLM, build_prompt
from executor import CodeExecutor, ExecutionResult, extract_first_code, validate_syntax


MAX_RETRIES = 3


# ──────────────────────────────────────────────────────────
# Prompt de correction
# ──────────────────────────────────────────────────────────

FIX_PROMPT = """You are XCore AI. The code you generated has an error.

Original request: {query}

Your previous code:
```python
{code}
```

Error ({error_type}, line {error_line}):
{error_message}

XCore context:
{context}

Fix the code. Output ONLY the corrected Python code, no explanation, no markdown fences.
"""


# ──────────────────────────────────────────────────────────
# Résultat de l'agent
# ──────────────────────────────────────────────────────────

@dataclass
class AgenticResult:
    query: str
    final_code: str
    execution: ExecutionResult | None
    attempts: int
    success: bool
    explanation: str = ""

    def summary(self) -> str:
        status = "✓ Succès" if self.success else "✗ Échec"
        return (
            f"{status} après {self.attempts} tentative(s)\n"
            f"Exécution : {self.execution.summary() if self.execution else 'non exécutée'}"
        )


# ──────────────────────────────────────────────────────────
# Agent en boucle
# ──────────────────────────────────────────────────────────

class AgenticLoop:
    """
    Boucle de génération auto-correctrice.

    Workflow :
        1. RAG → contexte
        2. LLM → code
        3. Validation syntaxe
        4. Exécution sandbox
        5. Si erreur → LLM corrige avec le traceback
        6. Répète jusqu'à succès ou MAX_RETRIES

    Usage:
        loop = AgenticLoop(config)
        result = loop.run("create a notification plugin with email service")
        print(result.final_code)
    """

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self.retriever = XCoreRetriever(config)
        self.llm = XCoreLLM(config)
        self.executor = CodeExecutor(
            timeout=10,
            framework_path=config.framework_path,
        )

    # ── Génération initiale ────────────────────────────────
    def _initial_generate(self, query: str) -> tuple[str, str]:
        """Retourne (code_brut_llm, contexte)."""
        retrieval = self.retriever.search(query)
        context = retrieval.build_context(max_chars=4000)

        messages = build_prompt(
            system=(
                self.config.system_prompt
                + "\nOutput ONLY Python code. No markdown fences, no explanation."
            ),
            context=context,
            history=[],
            query=query,
        )

        raw = self.llm.generate(messages)

        # Essaie d'extraire le code s'il est dans des backticks
        code = extract_first_code(raw) or raw.strip()

        return code, context

    # ── Correction ────────────────────────────────────────
    def _fix(self, query: str, code: str, error: ExecutionResult | None, context: str) -> str:
        """Demande au LLM de corriger le code après une erreur."""
        error_type = error.error_type if error else "SyntaxError"
        error_line = error.error_line if error else 0
        error_msg = error.stderr[:500] if error else "Syntax error"

        fix_prompt = FIX_PROMPT.format(
            query=query,
            code=code,
            error_type=error_type,
            error_line=error_line,
            error_message=error_msg,
            context=context[:2000],
        )

        messages = [
            {
                "role": "system",
                "content": "You are XCore AI. Output ONLY corrected Python code.",
            },
            {"role": "user", "content": fix_prompt},
        ]

        raw = self.llm.generate(messages)
        return extract_first_code(raw) or raw.strip()

    # ── Boucle principale ─────────────────────────────────
    def run(self, query: str, execute: bool = True) -> AgenticResult:
        """
        Exécute la boucle complète RAG → génération → validation → correction.

        Args:
            query:   La demande de l'utilisateur
            execute: Si False, skip l'exécution (validation syntaxe seulement)
        """
        code, context = self._initial_generate(query)
        last_execution: ExecutionResult | None = None

        for attempt in range(1, MAX_RETRIES + 1):

            # Validation syntaxe d'abord (rapide)
            syntax_ok, syntax_error = validate_syntax(code)
            if not syntax_ok:
                fake_result = ExecutionResult(
                    success=False,
                    stderr=syntax_error,
                    error_type="SyntaxError",
                    error_line=0,
                )
                if attempt < MAX_RETRIES:
                    code = self._fix(query, code, fake_result, context)
                    continue
                else:
                    return AgenticResult(
                        query=query,
                        final_code=code,
                        execution=fake_result,
                        attempts=attempt,
                        success=False,
                    )

            # Exécution si demandée
            if execute:
                result = self.executor.run(code)
                last_execution = result

                if result.success:
                    return AgenticResult(
                        query=query,
                        final_code=code,
                        execution=result,
                        attempts=attempt,
                        success=True,
                    )
                elif attempt < MAX_RETRIES:
                    # Corrige et réessaie
                    code = self._fix(query, code, result, context)
                else:
                    # Dernière tentative échouée
                    return AgenticResult(
                        query=query,
                        final_code=code,
                        execution=result,
                        attempts=attempt,
                        success=False,
                    )
            else:
                # Pas d'exécution → syntaxe OK suffit
                return AgenticResult(
                    query=query,
                    final_code=code,
                    execution=None,
                    attempts=attempt,
                    success=True,
                )

        # CORRIGÉ : return explicite en cas d'épuisement de la boucle (était manquant)
        return AgenticResult(
            query=query,
            final_code=code,
            execution=last_execution,
            attempts=MAX_RETRIES,
            success=False,
        )

    # ── Streaming avec progression visible ────────────────
    def run_stream(self, query: str, execute: bool = True) -> Generator[str, None, AgenticResult]:
        """
        Même logique mais streame les tokens de génération
        et les messages de progression.
        """
        retrieval = self.retriever.search(query)
        context = retrieval.build_context(max_chars=4000)

        messages = build_prompt(
            system=(
                self.config.system_prompt
                + "\nOutput ONLY Python code. No markdown fences, no explanation."
            ),
            context=context,
            history=[],
            query=query,
        )

        yield "Génération du code...\n```python\n"
        raw = ""
        for token in self.llm.stream(messages):
            raw += token
            yield token
        yield "\n```\n"

        code = extract_first_code(raw) or raw.strip()
        last_execution: ExecutionResult | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            syntax_ok, syntax_error = validate_syntax(code)
            if not syntax_ok:
                yield f"\n✗ Erreur syntaxe : {syntax_error}\n"
                if attempt < MAX_RETRIES:
                    yield f"Correction automatique (tentative {attempt + 1})...\n```python\n"
                    fake = ExecutionResult(
                        success=False,
                        stderr=syntax_error,
                        error_type="SyntaxError",
                        error_line=0,
                    )
                    code = self._fix(query, code, fake, context)
                    yield code
                    yield "\n```\n"
                    continue
                # CORRIGÉ : break explicite si dernière tentative
                break

            if execute:
                yield f"\nExécution (tentative {attempt})... "
                result = self.executor.run(code)
                last_execution = result

                if result.success:
                    out = result.stdout.strip()
                    yield f"✓ OK{f' — {out[:100]}' if out else ''}\n"
                    return AgenticResult(
                        query=query,
                        final_code=code,
                        execution=result,
                        attempts=attempt,
                        success=True,
                    )
                else:
                    yield f"✗ {result.error_type} : {result.stderr[:150]}\n"
                    if attempt < MAX_RETRIES:
                        yield f"\nCorrection automatique (tentative {attempt + 1})...\n```python\n"
                        code = self._fix(query, code, result, context)
                        yield code
                        yield "\n```\n"
            else:
                yield "\n✓ Syntaxe valide\n"
                return AgenticResult(
                    query=query,
                    final_code=code,
                    execution=None,
                    attempts=attempt,
                    success=True,
                )

        # CORRIGÉ : return de secours toujours présent en fin de générateur
        return AgenticResult(
            query=query,
            final_code=code,
            execution=last_execution,
            attempts=MAX_RETRIES,
            success=False,
        )