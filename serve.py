"""
web/server.py
--------------
Serveur FastAPI pour l'interface web XCore AI.
Expose les endpoints REST + SSE (Server-Sent Events) pour le streaming.

Démarrage :
    uvicorn web.server:app --reload --port 8000
    # puis ouvre : http://localhost:8000
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import XCore AI
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import XCoreAIConfig
from agent import XCoreAgent
from agentic import AgenticLoop
"""
web/server.py
--------------
Serveur FastAPI pour l'interface web XCore AI.
Expose les endpoints REST + SSE (Server-Sent Events) pour le streaming.

Démarrage :
    uvicorn web.server:app --reload --port 8000
    # puis ouvre : http://localhost:8000
"""

#from __future__ import annotations
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import XCore AI
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import XCoreAIConfig
from agent import XCoreAgent
from agentic import AgenticLoop
from generator import ProjectGenerator
from indexer import XCoreIndexer
from dataset.builder import DatasetBuilder


# ──────────────────────────────────────────────────────────
# App + config
# ──────────────────────────────────────────────────────────

app = FastAPI(title="XCore AI", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert les fichiers statiques
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Config globale (chargée une fois)
_config: XCoreAIConfig | None = None
_agent: XCoreAgent | None = None


def get_config() -> XCoreAIConfig:
    global _config
    if _config is None:
        cfg_path = Path("xcore_ai.yaml")
        if cfg_path.exists():
            _config = XCoreAIConfig.from_yaml(str(cfg_path))
        else:
            _config = XCoreAIConfig.from_env()
    return _config


def get_agent() -> XCoreAgent:
    global _agent
    if _agent is None:
        _agent = XCoreAgent(get_config())
    return _agent


# ──────────────────────────────────────────────────────────
# Modèles Pydantic
# ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"        # chat | agentic | generate
    session_id: str = "default"

class IndexRequest(BaseModel):
    force: bool = False

class DatasetRequest(BaseModel):
    augment: bool = False
    output: str = "dataset/xcore_train.jsonl"


# ──────────────────────────────────────────────────────────
# SSE helper
# ──────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_to_sse(gen) -> AsyncGenerator[str, None]:
    """
    Convertit un générateur synchrone en flux SSE asynchrone.
    Chaque token est envoyé comme event SSE.
    """
    loop = asyncio.get_event_loop()
    full_text = ""

    # Wrap le générateur synchrone pour ne pas bloquer l'event loop
    def consume():
        chunks = []
        try:
            while True:
                token = next(gen)
                chunks.append(token)
        except StopIteration as e:
            return chunks, e.value
        except Exception as e:
            return chunks, None

    # Exécute dans un thread pour ne pas bloquer
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = loop.run_in_executor(pool, consume)

        # Polling pendant l'exécution
        # (pour un vrai streaming on utiliserait une queue)
        tokens, final = await future

    for token in tokens:
        full_text += token
        yield sse_event({"type": "token", "content": token})
        await asyncio.sleep(0)  # yield control

    yield sse_event({"type": "done", "content": full_text})


# ──────────────────────────────────────────────────────────
# Routes HTML
# ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>XCore AI</h1><p>templates/index.html not found</p>")


# ──────────────────────────────────────────────────────────
# API — Chat avec streaming SSE
# ──────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream la réponse du LLM via SSE.
    Le client écoute avec EventSource.
    """
    config = get_config()

    async def event_generator():
        try:
            if req.mode == "agentic":
                loop_agent = AgenticLoop(config)
                gen = loop_agent.run_stream(req.message, execute=True)
            elif req.mode == "generate":
                gen_agent = ProjectGenerator(config)
                gen = gen_agent.generate_stream(req.message)
            else:
                agent = get_agent()
                gen = agent.ask_stream(req.message)

            # Streaming synchrone dans un thread
            import concurrent.futures
            import queue as q_module

            token_queue: q_module.Queue = q_module.Queue()
            done_event = asyncio.Event()

            def producer():
                try:
                    for token in gen:
                        token_queue.put(("token", token))
                except StopIteration as e:
                    token_queue.put(("done", None))
                except Exception as ex:
                    token_queue.put(("error", str(ex)))
                finally:
                    token_queue.put(("end", None))

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(pool, producer)
                full = ""
                while True:
                    try:
                        kind, val = token_queue.get(timeout=0.05)
                    except q_module.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if kind == "token":
                        full += val
                        yield sse_event({"type": "token", "content": val})
                    elif kind == "error":
                        yield sse_event({"type": "error", "content": val})
                        break
                    elif kind in ("done", "end"):
                        yield sse_event({"type": "done", "content": full})
                        break

                    await asyncio.sleep(0)

        except Exception as e:
            yield sse_event({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────
# API — Status du système
# ──────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    config = get_config()
    indexer = XCoreIndexer(config)
    stats = indexer.stats()

    from xcore_ai.llm import XCoreLLM
    llm = XCoreLLM(config)
    llm_ok, llm_msg = llm.is_available()

    return JSONResponse({
        "index": {
            "ready": stats["indexed"],
            "chunks": stats.get("chunks", 0),
            "path": stats.get("path", config.vector_db_path),
        },
        "llm": {
            "ready": llm_ok,
            "model": config.llm_model,
            "message": llm_msg,
            "url": config.ollama_base_url,
        },
        "config": {
            "framework_path": config.framework_path,
            "embedding_model": config.embedding_model,
            "retrieval_k": config.retrieval_k,
            "temperature": config.llm_temperature,
        },
    })


# ──────────────────────────────────────────────────────────
# API — Indexation
# ──────────────────────────────────────────────────────────

@app.post("/api/index/stream")
async def index_stream(req: IndexRequest):
    """Stream la progression de l'indexation."""
    config = get_config()

    async def gen():
        indexer = XCoreIndexer(config)
        yield sse_event({"type": "token", "content": "Démarrage de l'indexation...\n"})
        await asyncio.sleep(0)

        import concurrent.futures
        loop = asyncio.get_event_loop()

        def do_index():
            return indexer.index(force=req.force)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, do_index)
            count = await future

        msg = f"\n✓ {count} chunks indexés avec succès.\n"
        yield sse_event({"type": "token", "content": msg})
        yield sse_event({"type": "done", "content": msg})

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache"})


# ──────────────────────────────────────────────────────────
# API — Historique
# ──────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history():
    agent = get_agent()
    return JSONResponse({"history": agent._history[-50:]})


@app.delete("/api/history")
async def clear_history():
    agent = get_agent()
    agent.clear_history()
    return JSONResponse({"status": "cleared"})


# ──────────────────────────────────────────────────────────
# API — Dataset
# ──────────────────────────────────────────────────────────

@app.post("/api/dataset/preview")
async def dataset_preview():
    config = get_config()
    builder = DatasetBuilder(config)
    pairs = builder.preview(n=5)
    return JSONResponse({"pairs": pairs})


# ──────────────────────────────────────────────────────────
# Démarrage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)
from generator import ProjectGenerator
from indexer import XCoreIndexer
from dataset.builder import DatasetBuilder


# ──────────────────────────────────────────────────────────
# App + config
# ──────────────────────────────────────────────────────────

app = FastAPI(title="XCore AI", version="1.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert les fichiers statiques
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Config globale (chargée une fois)
_config: XCoreAIConfig | None = None
_agent: XCoreAgent | None = None


def get_config() -> XCoreAIConfig:
    global _config
    if _config is None:
        cfg_path = Path("xcore_ai.yaml")
        if cfg_path.exists():
            _config = XCoreAIConfig.from_yaml(str(cfg_path))
        else:
            _config = XCoreAIConfig.from_env()
    return _config


def get_agent() -> XCoreAgent:
    global _agent
    if _agent is None:
        _agent = XCoreAgent(get_config())
    return _agent


# ──────────────────────────────────────────────────────────
# Modèles Pydantic
# ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"        # chat | agentic | generate
    session_id: str = "default"

class IndexRequest(BaseModel):
    force: bool = False

class DatasetRequest(BaseModel):
    augment: bool = False
    output: str = "dataset/xcore_train.jsonl"


# ──────────────────────────────────────────────────────────
# SSE helper
# ──────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_to_sse(gen) -> AsyncGenerator[str, None]:
    """
    Convertit un générateur synchrone en flux SSE asynchrone.
    Chaque token est envoyé comme event SSE.
    """
    loop = asyncio.get_event_loop()
    full_text = ""

    # Wrap le générateur synchrone pour ne pas bloquer l'event loop
    def consume():
        chunks = []
        try:
            while True:
                token = next(gen)
                chunks.append(token)
        except StopIteration as e:
            return chunks, e.value
        except Exception as e:
            return chunks, None

    # Exécute dans un thread pour ne pas bloquer
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = loop.run_in_executor(pool, consume)

        # Polling pendant l'exécution
        # (pour un vrai streaming on utiliserait une queue)
        tokens, final = await future

    for token in tokens:
        full_text += token
        yield sse_event({"type": "token", "content": token})
        await asyncio.sleep(0)  # yield control

    yield sse_event({"type": "done", "content": full_text})


# ──────────────────────────────────────────────────────────
# Routes HTML
# ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>XCore AI</h1><p>templates/index.html not found</p>")


# ──────────────────────────────────────────────────────────
# API — Chat avec streaming SSE
# ──────────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream la réponse du LLM via SSE.
    Le client écoute avec EventSource.
    """
    config = get_config()

    async def event_generator():
        try:
            if req.mode == "agentic":
                loop_agent = AgenticLoop(config)
                gen = loop_agent.run_stream(req.message, execute=True)
            elif req.mode == "generate":
                gen_agent = ProjectGenerator(config)
                gen = gen_agent.generate_stream(req.message)
            else:
                agent = get_agent()
                gen = agent.ask_stream(req.message)

            # Streaming synchrone dans un thread
            import concurrent.futures
            import queue as q_module

            token_queue: q_module.Queue = q_module.Queue()
            done_event = asyncio.Event()

            def producer():
                try:
                    for token in gen:
                        token_queue.put(("token", token))
                except StopIteration as e:
                    token_queue.put(("done", None))
                except Exception as ex:
                    token_queue.put(("error", str(ex)))
                finally:
                    token_queue.put(("end", None))

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(pool, producer)
                full = ""
                while True:
                    try:
                        kind, val = token_queue.get(timeout=0.05)
                    except q_module.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if kind == "token":
                        full += val
                        yield sse_event({"type": "token", "content": val})
                    elif kind == "error":
                        yield sse_event({"type": "error", "content": val})
                        break
                    elif kind in ("done", "end"):
                        yield sse_event({"type": "done", "content": full})
                        break

                    await asyncio.sleep(0)

        except Exception as e:
            yield sse_event({"type": "error", "content": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────
# API — Status du système
# ──────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    config = get_config()
    indexer = XCoreIndexer(config)
    stats = indexer.stats()

    from llm import XCoreLLM
    llm = XCoreLLM(config)
    llm_ok, llm_msg = llm.is_available()

    return JSONResponse({
        "index": {
            "ready": stats["indexed"],
            "chunks": stats.get("chunks", 0),
            "path": stats.get("path", config.vector_db_path),
        },
        "llm": {
            "ready": llm_ok,
            "model": config.llm_model,
            "message": llm_msg,
            "url": config.ollama_base_url,
        },
        "config": {
            "framework_path": config.framework_path,
            "embedding_model": config.embedding_model,
            "retrieval_k": config.retrieval_k,
            "temperature": config.llm_temperature,
        },
    })


# ──────────────────────────────────────────────────────────
# API — Indexation
# ──────────────────────────────────────────────────────────

@app.post("/api/index/stream")
async def index_stream(req: IndexRequest):
    """Stream la progression de l'indexation."""
    config = get_config()

    async def gen():
        indexer = XCoreIndexer(config)
        yield sse_event({"type": "token", "content": "Démarrage de l'indexation...\n"})
        await asyncio.sleep(0)

        import concurrent.futures
        loop = asyncio.get_event_loop()

        def do_index():
            return indexer.index(force=req.force)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, do_index)
            count = await future

        msg = f"\n✓ {count} chunks indexés avec succès.\n"
        yield sse_event({"type": "token", "content": msg})
        yield sse_event({"type": "done", "content": msg})

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache"})


# ──────────────────────────────────────────────────────────
# API — Historique
# ──────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history():
    agent = get_agent()
    return JSONResponse({"history": agent._history[-50:]})


@app.delete("/api/history")
async def clear_history():
    agent = get_agent()
    agent.clear_history()
    return JSONResponse({"status": "cleared"})


# ──────────────────────────────────────────────────────────
# API — Dataset
# ──────────────────────────────────────────────────────────

@app.post("/api/dataset/preview")
async def dataset_preview():
    config = get_config()
    builder = DatasetBuilder(config)
    pairs = builder.preview(n=5)
    return JSONResponse({"pairs": pairs})


# ──────────────────────────────────────────────────────────
# Démarrage
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)