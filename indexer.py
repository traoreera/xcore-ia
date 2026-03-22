"""
xcore_ai/indexer.py
-------------------
Parse le framework XCore, découpe en chunks intelligents
(respect des frontières de fonctions/classes) et stocke
les embeddings dans ChromaDB.
"""

import os
import ast
import hashlib
from pathlib import Path
from typing import Generator


from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback si langchain-chroma pas installé

from config import XCoreAIConfig
from embeddings import XCoreEmbeddings


# ──────────────────────────────────────────────────────────
# Splitter Python-aware (respecte fonctions et classes)
# ──────────────────────────────────────────────────────────

def make_python_splitter(chunk_size: int = 800, overlap: int = 100):
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


def make_markdown_splitter(chunk_size: int = 600, overlap: int = 80):
    return RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )


# ──────────────────────────────────────────────────────────
# Extraction AST — métadonnées riches par chunk
# ──────────────────────────────────────────────────────────

def extract_python_metadata(source: str, filepath: str) -> dict:
    """Extrait classes, fonctions et decorators depuis l'AST Python."""
    meta = {"classes": [], "functions": [], "decorators": []}
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                meta["classes"].append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                meta["functions"].append(node.name)
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        meta["decorators"].append(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        meta["decorators"].append(dec.attr)
    except SyntaxError:
        pass
    return meta


# ──────────────────────────────────────────────────────────
# Chargement des fichiers sources
# ──────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".rst": "rst",
}


def load_source_files(framework_path: str) -> Generator[Document, None, None]:
    """Charge tous les fichiers supportés depuis le dossier du framework."""
    root = Path(framework_path)
    if not root.exists():
        raise FileNotFoundError(f"Framework path introuvable : {framework_path}")

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        ext = filepath.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        # Ignore __pycache__, .git, venv, dist
        parts = filepath.parts
        if any(p in parts for p in ("__pycache__", ".git", "venv", "dist", "build", ".eggs")):
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if len(content.strip()) < 20:
            continue

        lang = SUPPORTED_EXTENSIONS[ext]
        meta = {
            "source": str(filepath),
            "relative_path": str(filepath.relative_to(root)),
            "language": lang,
            "filename": filepath.name,
        }
        if lang == "python":
            py_meta = extract_python_metadata(content, str(filepath))
            meta.update({
                "classes": ", ".join(py_meta["classes"]),
                "functions": ", ".join(py_meta["functions"]),
                "decorators": ", ".join(py_meta["decorators"]),
            })

        yield Document(page_content=content, metadata=meta)


# ──────────────────────────────────────────────────────────
# Indexeur principal
# ──────────────────────────────────────────────────────────

class XCoreIndexer:

    def __init__(self, config: XCoreAIConfig):
        self.config = config
        self.embedding_fn = XCoreEmbeddings(config.embedding_model)
        self._py_splitter = make_python_splitter(
            config.chunk_size, config.chunk_overlap
        )
        self._md_splitter = make_markdown_splitter(
            config.chunk_size, config.chunk_overlap
        )

    # ── chunk un document selon sa langue ──────────────────
    def _split(self, doc: Document) -> list[Document]:
        lang = doc.metadata.get("language", "text")
        if lang == "python":
            return self._py_splitter.split_documents([doc])
        elif lang in ("markdown", "rst", "text"):
            return self._md_splitter.split_documents([doc])
        return [doc]

    # ── fingerprint du contenu pour skip si déjà indexé ───
    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ── indexation complète ────────────────────────────────
    def index(self, force: bool = False) -> int:
        """
        Indexe le framework XCore dans ChromaDB.
        Retourne le nombre de chunks stockés.

        Args:
            force: Si True, recrée l'index complet même si déjà existant.
        """
        db_path = self.config.vector_db_path
        collection = self.config.collection_name

        # Supprime l'ancien index si force=True
        if force and Path(db_path).exists():
            import shutil
            shutil.rmtree(db_path)
            print(f"  [index] Ancien index supprimé : {db_path}")

        print(f"  [index] Lecture du framework : {self.config.framework_path}")

        all_chunks: list[Document] = []
        file_count = 0

        for doc in load_source_files(self.config.framework_path):
            chunks = self._split(doc)
            # Ajoute le fingerprint dans les métadonnées
            for chunk in chunks:
                chunk.metadata["fingerprint"] = self._fingerprint(chunk.page_content)
            all_chunks.extend(chunks)
            file_count += 1

        if not all_chunks:
            print("  [index] Aucun fichier trouvé. Vérifie framework_path.")
            return 0

        print(f"  [index] {file_count} fichiers → {len(all_chunks)} chunks")
        print(f"  [index] Calcul des embeddings (peut prendre 1-2 min)...")

        # Batch pour éviter les OOM sur les gros repos
        BATCH = 128
        db = None
        for i in range(0, len(all_chunks), BATCH):
            batch = all_chunks[i : i + BATCH]
            if db is None:
                db = Chroma.from_documents(
                    batch,
                    self.embedding_fn,
                    persist_directory=db_path,
                    collection_name=collection,
                )
            else:
                db.add_documents(batch)
            print(f"  [index] Batch {i // BATCH + 1}/{(len(all_chunks) - 1) // BATCH + 1} indexé")

        # CORRIGÉ : db.persist() est déprécié depuis ChromaDB ≥0.4 — la persistance
        # est automatique via persist_directory. L'appel est supprimé.

        print(f"  [index] Index sauvegardé → {db_path}")
        return len(all_chunks)

    # ── stats de l'index existant ──────────────────────────
    def stats(self) -> dict:
        db_path = self.config.vector_db_path
        if not Path(db_path).exists():
            return {"indexed": False, "chunks": 0}
        try:
            db = Chroma(
                persist_directory=db_path,
                collection_name=self.config.collection_name,
                embedding_function=self.embedding_fn,
            )
            count = db._collection.count()
            return {"indexed": True, "chunks": count, "path": db_path}
        except Exception as e:
            return {"indexed": False, "error": str(e)}