"""
xcore_ai/tools/executor.py
---------------------------
Exécute du code Python généré par le LLM dans un sous-processus
isolé avec timeout. Capture stdout, stderr et les exceptions.
L'agent peut ainsi valider automatiquement le code qu'il génère.
"""

from __future__ import annotations
import ast
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────
# Résultat d'exécution
# ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    error_type: str = ""
    error_line: int = 0
    duration_ms: float = 0.0

    @property
    def output(self) -> str:
        return self.stdout.strip() or self.stderr.strip()

    def summary(self) -> str:
        if self.success:
            out = self.stdout.strip()
            return f"OK — {out[:200]}" if out else "OK (pas de sortie)"
        return f"ERREUR ({self.error_type}) ligne {self.error_line}: {self.stderr[:300]}"

    def __bool__(self):
        return self.success


# ──────────────────────────────────────────────────────────
# Extraction du code depuis la réponse LLM
# ──────────────────────────────────────────────────────────

def extract_code_blocks(text: str) -> list[str]:
    """
    Extrait tous les blocs ```python ... ``` d'une réponse LLM.
    Si aucun bloc trouvé, retourne le texte brut si ça ressemble à du code.
    """
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]

    # Fallback : si la réponse entière ressemble à du code Python
    stripped = text.strip()
    if any(kw in stripped for kw in ("def ", "class ", "import ", "from ", "@")):
        return [stripped]

    return []


def extract_first_code(text: str) -> str | None:
    """Retourne le premier bloc de code Python trouvé."""
    blocks = extract_code_blocks(text)
    return blocks[0] if blocks else None


# ──────────────────────────────────────────────────────────
# Validation statique avant exécution
# ──────────────────────────────────────────────────────────

def validate_syntax(code: str) -> tuple[bool, str]:
    """Vérifie la syntaxe Python sans exécuter."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError ligne {e.lineno}: {e.msg}"


def is_safe(code: str) -> tuple[bool, str]:
    """
    Vérifications de sécurité basiques.
    Bloque les appels dangereux évidents.
    """
    dangerous = [
        ("os.system", "appel système direct"),
        ("subprocess.call", "subprocess sans liste"),
        ("__import__('os')", "import dynamique os"),
        ("eval(", "eval dynamique"),
        ("exec(", "exec dynamique"),
        ("open('/etc", "accès /etc"),
        ("shutil.rmtree('/'", "suppression racine"),
    ]
    for pattern, reason in dangerous:
        if pattern in code:
            return False, f"Code potentiellement dangereux : {reason}"
    return True, ""


# ──────────────────────────────────────────────────────────
# Exécuteur principal
# ──────────────────────────────────────────────────────────

class CodeExecutor:
    """
    Exécute du code Python dans un subprocess isolé.

    Usage:
        executor = CodeExecutor(timeout=10, framework_path="./xcore_framework")
        result = executor.run(code_string)
        print(result.summary())
    """

    def __init__(
        self,
        timeout: int = 15,
        framework_path: str = "./xcore_framework",
        extra_paths: list[str] = None,
    ):
        self.timeout = timeout
        self.framework_path = framework_path
        self.extra_paths = extra_paths or []

    def _build_preamble(self) -> str:
        """Ajoute les imports nécessaires pour que le code XCore tourne."""
        paths = [str(Path(self.framework_path).parent.resolve())]
        paths.extend(self.extra_paths)
        path_lines = "\n".join(
            f'sys.path.insert(0, {repr(p)})' for p in paths
        )
        return textwrap.dedent(f"""
import sys
{path_lines}
import logging
logging.basicConfig(level=logging.WARNING)
""").strip()

    def run(self, code: str, safe_check: bool = True) -> ExecutionResult:
        """
        Exécute le code et retourne un ExecutionResult.

        Args:
            code:       Code Python à exécuter
            safe_check: Si True, vérifie les appels dangereux avant exécution
        """
        import time

        # 1. Validation syntaxique
        ok, msg = validate_syntax(code)
        if not ok:
            return ExecutionResult(
                success=False,
                stderr=msg,
                error_type="SyntaxError",
            )

        # 2. Vérification de sécurité
        if safe_check:
            ok, msg = is_safe(code)
            if not ok:
                return ExecutionResult(
                    success=False,
                    stderr=msg,
                    error_type="SecurityError",
                )

        # 3. Écrit dans un fichier temporaire
        preamble = self._build_preamble()
        full_code = f"{preamble}\n\n{code}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name

        # 4. Exécute dans un subprocess
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(Path(self.framework_path).parent),
            )
            duration = (time.perf_counter() - start) * 1000

            # Parse l'erreur pour extraire la ligne
            error_type = ""
            error_line = 0
            if proc.returncode != 0 and proc.stderr:
                lines = proc.stderr.strip().splitlines()
                for line in lines:
                    m = re.search(r'File ".+", line (\d+)', line)
                    if m:
                        error_line = int(m.group(1)) - len(preamble.splitlines()) - 1
                if lines:
                    last = lines[-1]
                    if ":" in last:
                        error_type = last.split(":")[0].strip()

            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                error_type=error_type,
                error_line=max(0, error_line),
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stderr=f"Timeout après {self.timeout}s",
                error_type="TimeoutError",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stderr=str(e),
                error_type=type(e).__name__,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def run_from_llm_response(self, llm_text: str) -> ExecutionResult | None:
        """
        Extrait le code d'une réponse LLM et l'exécute.
        Retourne None si aucun code trouvé.
        """
        code = extract_first_code(llm_text)
        if not code:
            return None
        return self.run(code)