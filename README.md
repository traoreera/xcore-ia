# XCore AI — Copilote IA pour le framework XCore

Copilote intelligent spécialisé sur ton framework XCore.  
Basé sur **DeepSeek Coder 6.7B** (via Ollama) + **RAG** (ChromaDB + SentenceTransformers).

---

## Architecture

```
xcore_framework/          ← ton framework (indexé)
   core/
   services/
   plugins/

xcore_ai/                 ← le copilote
   config.py              ← configuration centralisée
   embeddings.py          ← wrapper SentenceTransformers
   indexer.py             ← parse + chunk + vectorise
   retriever.py           ← recherche sémantique ChromaDB
   llm.py                 ← client Ollama (streaming)
   agent.py               ← orchestrateur RAG + mémoire
   cli.py                 ← interface CLI

vector_db/                ← index ChromaDB (généré)
xcore_ai.yaml             ← configuration
```

---

## Installation

### 1. Ollama + DeepSeek Coder

```bash
# Installe Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Télécharge le modèle
ollama pull deepseek-coder:6.7b

# Lance le serveur (dans un terminal séparé)
ollama serve
```

### 2. Dépendances Python

```bash
pip install langchain langchain-community chromadb sentence-transformers requests pyyaml
```

### 3. Installe XCore AI

```bash
cd xcore_ai
pip install -e .
```

---

## Utilisation

### Étape 1 — Indexe ton framework

```bash
xcore-ai index
```

Options :
```bash
xcore-ai index --force     # Recrée l'index depuis zéro
xcore-ai --config mon_projet.yaml index
```

### Étape 2 — Vérifie l'état

```bash
xcore-ai status
```

Sortie attendue :
```
  Index vectoriel  : OK  (1247 chunks)
  Ollama / LLM     : OK  (deepseek-coder:6.7b)
  Framework path   : ./xcore_framework
```

### Étape 3 — Chat interactif

```bash
xcore-ai chat
```

```
  vous ▸ create a plugin with database and logger injection
  XCore AI ▸ Here's a complete XCore plugin...
```

### Question directe (scripts, CI)

```bash
xcore-ai ask "génère un plugin de cache avec TTL"
xcore-ai ask "comment fonctionne @inject dans XCore ?"
xcore-ai ask "debug : AttributeError sur BaseService"
```

---

## Configuration (xcore_ai.yaml)

```yaml
framework_path: "./xcore_framework"
vector_db_path: "./vector_db"
llm_model: "deepseek-coder:6.7b"
embedding_model: "all-MiniLM-L6-v2"
chunk_size: 800
retrieval_k: 6
llm_temperature: 0.1
show_sources: true
```

Variables d'environnement disponibles :
```bash
XCORE_AI_FRAMEWORK_PATH=./my_framework
XCORE_AI_LLM_MODEL=codellama:7b
XCORE_AI_TEMPERATURE=0.2
```

---

## Usage programmatique

```python
from xcore_ai.config import XCoreAIConfig
from xcore_ai.agent import XCoreAgent

config = XCoreAIConfig(
    framework_path="./xcore_framework",
    llm_model="deepseek-coder:6.7b",
)

agent = XCoreAgent(config)

# Réponse complète
response = agent.ask("create a REST API plugin using XCore")
print(response.text)
print("Sources:", response.sources)

# Streaming
for token in agent.ask_stream("explain the @inject decorator"):
    print(token, end="", flush=True)
```

---

## Commandes CLI

| Commande | Description |
|---|---|
| `xcore-ai index` | Indexe le framework |
| `xcore-ai index --force` | Recrée l'index |
| `xcore-ai chat` | Mode interactif |
| `xcore-ai ask "..."` | Question directe |
| `xcore-ai status` | État du système |
| `xcore-ai clear` | Efface l'historique |

Commandes spéciales dans le chat :
| Commande | Description |
|---|---|
| `/clear` | Efface l'historique de conversation |
| `/history` | Affiche les dernières questions |
| `/status` | État du système |
| `/quit` | Quitte le chat |

---

## Prochaines étapes

- **Fine-tuning LoRA** : générer un dataset `(instruction, code_xcore, output)` depuis ton repo pour affiner DeepSeek Coder directement sur tes patterns.
- **Agent avec outils** : ajout d'un outil `run_tests` pour que l'IA valide le code généré en l'exécutant.
- **Mode génération de projet** : `xcore-ai generate --template api --name mon_service` pour générer un projet XCore complet.