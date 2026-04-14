"""
xcore_ai/cli.py
----------------
Interface CLI de XCore AI.
Commandes disponibles :
  xcore-ai index          — indexe le framework
  xcore-ai chat           — mode interactif
  xcore-ai ask "question" — question directe
  xcore-ai status         — état du système
  xcore-ai clear          — vide l'historique
"""

from __future__ import annotations
import sys
import time
import warnings
import argparse
from pathlib import Path

# Supprime le warning de compatibilité Pydantic V1 / Python ≥3.14
# (provient de langchain_core, pas de notre code — sans impact fonctionnel)
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible",
    category=UserWarning,
)
# Supprime le warning de dépréciation Chroma de langchain_community
# (remplacé par langchain-chroma dans retriever.py et indexer.py)
warnings.filterwarnings(
    "ignore",
    message=".*Chroma.*deprecated.*",
    category=DeprecationWarning,
)


# ── Couleurs ANSI (sans dépendance externe) ───────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[40m"

    @staticmethod
    def strip(text: str) -> str:
        """Retire les codes ANSI pour la sortie non-TTY."""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)


def _print(text: str, color: str = "", bold: bool = False, end: str = "\n"):
    if not sys.stdout.isatty():
        print(C.strip(text), end=end)
    else:
        prefix = (C.BOLD if bold else "") + color
        print(f"{prefix}{text}{C.RESET}", end=end)


def _banner():
    banner = f"""
{C.CYAN}{C.BOLD}
  ██╗  ██╗ ██████╗ ██████╗ ██████╗ ███████╗     █████╗ ██╗
  ╚██╗██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔══██╗██║
   ╚███╔╝ ██║     ██║   ██║██████╔╝█████╗      ███████║██║
   ██╔██╗ ██║     ██║   ██║██╔══██╗██╔══╝      ██╔══██║██║
  ██╔╝ ██╗╚██████╗╚██████╔╝██║  ██║███████╗    ██║  ██║██║
  ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝
{C.RESET}{C.DIM}  Copilote IA pour le framework XCore —{C.RESET}
"""
    print(banner)


def _separator():
    _print("─" * 60, C.DIM)


def _print_sources(sources: list[str]):
    if not sources:
        return
    _print("\n  Sources :", C.DIM)
    for s in sources:
        _print(f"    · {s}", C.DIM)


# ── Commande : index ──────────────────────────────────────

def cmd_index(config, args):
    # CORRIGÉ : import relatif unifié
    from indexer import XCoreIndexer

    _print("\n  Indexation du framework XCore...\n", C.CYAN, bold=True)
    start = time.time()

    indexer = XCoreIndexer(config)
    count = indexer.index(force=getattr(args, "force", False))

    elapsed = time.time() - start
    if count > 0:
        _print(f"\n  {count} chunks indexés en {elapsed:.1f}s", C.GREEN, bold=True)
        _print(f"  Index sauvegardé : {config.vector_db_path}\n", C.DIM)
    else:
        _print("\n  Aucun fichier trouvé. Vérifie framework_path dans la config.", C.RED)


# ── Commande : status ─────────────────────────────────────

def cmd_status(config, args):
    # CORRIGÉ : imports relatifs uniformisés (étaient mélangés avec/sans point)
    from indexer import XCoreIndexer
    from llm import XCoreLLM

    _print("\n  Statut XCore AI\n", C.CYAN, bold=True)
    _separator()

    # Index
    indexer = XCoreIndexer(config)
    stats = indexer.stats()
    if stats["indexed"]:
        _print(f"  Index vectoriel  : {C.GREEN}OK{C.RESET}  ({stats['chunks']} chunks)")
    else:
        _print(f"  Index vectoriel  : {C.RED}Absent{C.RESET}  → lance: xcore-ai index")

    # Ollama
    llm = XCoreLLM(config)
    ok, msg = llm.is_available()
    if ok:
        _print(f"  Ollama / LLM     : {C.GREEN}OK{C.RESET}  ({config.llm_model})")
    else:
        _print(f"  Ollama / LLM     : {C.RED}Indisponible{C.RESET}")
        _print(f"    {msg}", C.DIM)

    # Config
    _print(f"\n  Framework path   : {config.framework_path}", C.DIM)
    _print(f"  Embedding model  : {config.embedding_model}", C.DIM)
    _print(f"  Retrieval k      : {config.retrieval_k}", C.DIM)
    _print(f"  LLM temperature  : {config.llm_temperature}", C.DIM)
    _print("")


# ── Commande : ask ────────────────────────────────────────

def cmd_ask(config, args):
    # CORRIGÉ : import relatif unifié
    from agent import XCoreAgent

    query = " ".join(args.question)
    if not query.strip():
        _print("  Question vide.", C.RED)
        return

    agent = XCoreAgent(config)
    ok, msg = agent.check_ready()
    if not ok:
        _print(f"\n  {msg}", C.RED)
        return

    _print(f"\n  {C.BOLD}Vous :{C.RESET} {query}\n")
    _print(f"  {C.CYAN}{C.BOLD}XCore AI :{C.RESET} ", end="")

    response_text = ""
    try:
        for token in agent.ask_stream(query):
            print(token, end="", flush=True)
            response_text += token
    except KeyboardInterrupt:
        pass

    print()

    if config.show_sources:
        retrieval = agent.retriever.search(query)
        _print_sources(retrieval.source_list())

    print()


# ── Commande : chat (mode interactif) ────────────────────

def cmd_chat(config, args):
    # CORRIGÉ : import relatif unifié (était sans point)
    from agent import XCoreAgent

    _banner()
    agent = XCoreAgent(config)

    ok, msg = agent.check_ready()
    if not ok:
        _print(f"\n  Prérequis manquants :\n  {msg}\n", C.RED)
        return

    _print(f"  Modèle  : {config.llm_model}", C.DIM)
    _print(f"  Index   : {config.vector_db_path}", C.DIM)
    _print(f"  Commandes spéciales : /clear  /history  /status  /quit\n", C.DIM)
    _separator()
    _print("  Pose ta question sur XCore (Ctrl+C ou /quit pour quitter)\n")

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.BLUE}  vous ▸{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            _print("\n\n  À bientôt !", C.CYAN)
            break

        if not user_input:
            continue

        # Commandes spéciales
        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            _print("\n  À bientôt !", C.CYAN)
            break

        if user_input.lower() == "/clear":
            agent.clear_history()
            _print("  Historique effacé.\n", C.GREEN)
            continue

        if user_input.lower() == "/history":
            _print("\n  Historique récent :", C.CYAN)
            print(agent.history_summary())
            print()
            continue

        if user_input.lower() == "/status":
            cmd_status(config, args)
            continue

        if user_input.startswith("/help"):
            _print("\n  Commandes : /clear  /history  /status  /quit\n", C.DIM)
            continue

        # Réponse
        print(f"\n{C.CYAN}{C.BOLD}  XCore AI ▸{C.RESET} ", end="", flush=True)

        response_text = ""
        try:
            for token in agent.ask_stream(user_input):
                print(token, end="", flush=True)
                response_text += token
        except KeyboardInterrupt:
            _print("\n  [interrompu]", C.DIM)

        print("\n")

        if config.show_sources:
            retrieval = agent.retriever.search(user_input)
            _print_sources(retrieval.source_list())

        print()
        _separator()
        print()


# ── Commande : clear ──────────────────────────────────────

def cmd_clear(config, args):
    from agent import XCoreAgent
    agent = XCoreAgent(config)
    agent.clear_history()
    _print("  Historique de conversation effacé.", C.GREEN)


# ── Commande : generate ───────────────────────────────────

def cmd_generate(config, args):
    from generator import ProjectGenerator

    description = " ".join(args.description)
    if not description.strip():
        _print("  Description vide.", C.RED)
        return

    _print(f"\n  Génération du projet XCore...\n", C.CYAN, bold=True)
    _print(f"  Description : {description}\n", C.DIM)

    gen = ProjectGenerator(config)

    try:
        for msg in gen.generate_stream(description, output_dir=getattr(args, "output", None)):
            _print(f"  {msg}", end="")
    except Exception as e:
        _print(f"\n  Erreur : {e}", C.RED)
        return

    print()


# ── Commande : run (agentic — génère + exécute + corrige) ─

def cmd_run(config, args):
    from agentic import AgenticLoop

    query = " ".join(args.query)
    no_exec = getattr(args, "no_exec", False)

    _print(f"\n  XCore AI Agent\n", C.CYAN, bold=True)
    _print(f"  Requête : {query}\n", C.DIM)
    _separator()

    loop = AgenticLoop(config)

    full = ""
    result = None
    try:
        gen = loop.run_stream(query, execute=not no_exec)
        while True:
            try:
                token = next(gen)
                print(token, end="", flush=True)
                full += token
            except StopIteration as e:
                result = e.value
                break
    except KeyboardInterrupt:
        _print("\n  [interrompu]", C.DIM)
        return

    print()
    _separator()
    if result:
        if result.success:
            _print(f"\n  {result.summary()}", C.GREEN)
        else:
            _print(f"\n  {result.summary()}", C.RED)
    print()


# ── Commande : dataset ────────────────────────────────────

def cmd_dataset(config, args):
    # CORRIGÉ : imports relatifs uniformisés
    from dataset.builder import DatasetBuilder
    from llm import XCoreLLM

    subcommand = getattr(args, "dataset_cmd", "build")

    if subcommand == "preview":
        _print("\n  Aperçu du dataset XCore\n", C.CYAN, bold=True)
        builder = DatasetBuilder(config)
        pairs = builder.preview(n=5)
        for i, p in enumerate(pairs, 1):
            _print(f"\n  [{i}] {p['instruction']}", C.BOLD)
            _print(f"  Type : {p.get('type', '?')}", C.DIM)
            _print(f"  Code ({len(p['output'])} chars) :\n", C.DIM)
            print(p["output"][:300] + ("..." if len(p["output"]) > 300 else ""))
        return

    if subcommand == "lora":
        from dataset.lora_config import generate_config
        _print("\n  Génération de la config LoRA Axolotl\n", C.CYAN, bold=True)
        generate_config(
            dataset_path=getattr(args, "dataset_path", "dataset/xcore_train.jsonl"),
            output_dir=".",
            export=getattr(args, "export", False),
        )
        return

    # Build
    _print("\n  Construction du dataset XCore AI\n", C.CYAN, bold=True)

    augment = getattr(args, "augment", False)
    llm = XCoreLLM(config) if augment else None
    builder = DatasetBuilder(config, llm=llm)

    output = getattr(args, "output", "dataset/xcore_train.jsonl")
    examples = getattr(args, "examples", "./examples")

    stats = builder.build(
        output_path=output,
        examples_dir=examples,
        augment=augment,
        augment_top_n=20,
    )

    _separator()
    _print(f"\n  Dataset prêt :", C.GREEN, bold=True)
    _print(f"    Total paires    : {stats['total']}", C.DIM)
    _print(f"    Depuis framework: {stats['from_framework']}", C.DIM)
    _print(f"    Exemples manuels: {stats['from_examples']}", C.DIM)
    _print(f"    Augmentées LLM  : {stats['augmented']}", C.DIM)
    _print(f"    Fichier JSONL   : {stats['output_jsonl']}", C.DIM)
    _print(f"\n  Fine-tuning LoRA : xcore-ai dataset lora\n", C.DIM)


# ── Point d'entrée principal ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="xcore-ai",
        description="XCore AI — Copilote IA pour le framework XCore",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Chemin vers le fichier de config YAML (défaut: xcore_ai.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # index
    p_index = subparsers.add_parser("index", help="Indexe le framework XCore")
    p_index.add_argument("--force", "-f", action="store_true",
                         help="Recrée l'index depuis zéro")

    # chat
    subparsers.add_parser("chat", help="Mode interactif")

    # ask
    p_ask = subparsers.add_parser("ask", help="Question directe")
    p_ask.add_argument("question", nargs="+", help="Ta question")

    # status
    subparsers.add_parser("status", help="État du système")

    # clear
    subparsers.add_parser("clear", help="Efface l'historique")

    # generate — génère un projet complet
    p_gen = subparsers.add_parser("generate", help="Génère un projet XCore complet")
    p_gen.add_argument("description", nargs="+", help="Description du projet")
    p_gen.add_argument("--output", "-o", default=None, help="Dossier de sortie")

    # run — agent agentic avec auto-correction
    p_run = subparsers.add_parser("run", help="Génère + exécute + corrige automatiquement")
    p_run.add_argument("query", nargs="+", help="Ta demande de code")
    p_run.add_argument("--no-exec", action="store_true", help="Ne pas exécuter le code")

    # dataset — construit le dataset pour le fine-tuning
    p_ds = subparsers.add_parser("dataset", help="Construit le dataset LoRA")
    ds_sub = p_ds.add_subparsers(dest="dataset_cmd")
    ds_build = ds_sub.add_parser("build", help="Construit le dataset")
    ds_build.add_argument("--output", "-o", default="dataset/xcore_train.jsonl")
    ds_build.add_argument("--examples", default="./examples")
    ds_build.add_argument("--augment", action="store_true", help="Augmente via LLM")
    ds_sub.add_parser("preview", help="Aperçu des 5 premières paires")
    ds_lora = ds_sub.add_parser("lora", help="Génère la config Axolotl")
    ds_lora.add_argument("--dataset-path", default="dataset/xcore_train.jsonl")
    ds_lora.add_argument("--export", action="store_true", help="Inclut le script d'export Ollama")

    args = parser.parse_args()

    # Charge la config — CORRIGÉ : import relatif unifié
    from config import XCoreAIConfig
    config_path = args.config or "xcore_ai.yaml"
    if Path(config_path).exists():
        try:
            config = XCoreAIConfig.from_yaml(config_path)
            _print(f"  Config chargée : {config_path}", C.DIM)
        except Exception as e:
            _print(f"  Erreur config YAML : {e} — utilisation des valeurs par défaut", C.YELLOW)
            config = XCoreAIConfig.from_env()
    else:
        config = XCoreAIConfig.from_env()

    # Dispatch
    dispatch = {
        "index":    cmd_index,
        "chat":     cmd_chat,
        "ask":      cmd_ask,
        "status":   cmd_status,
        "clear":    cmd_clear,
        "generate": cmd_generate,
        "run":      cmd_run,
        "dataset":  cmd_dataset,
    }

    if args.command is None:
        # Pas de commande → mode chat par défaut
        cmd_chat(config, args)
    elif args.command in dispatch:
        dispatch[args.command](config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()