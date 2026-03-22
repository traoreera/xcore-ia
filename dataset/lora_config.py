"""
xcore_ai/dataset/lora_config.py
---------------------------------
Génère les fichiers de configuration pour le fine-tuning LoRA
de DeepSeek Coder avec Axolotl.

Usage:
    python -m xcore_ai.dataset.lora_config --dataset ./dataset/xcore_train.jsonl
"""

from __future__ import annotations
import argparse
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ──────────────────────────────────────────────────────────
# Config Axolotl pour DeepSeek Coder 6.7B + LoRA
# ──────────────────────────────────────────────────────────

AXOLOTL_CONFIG = {
    # Modèle de base
    "base_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "model_type": "AutoModelForCausalLM",
    "tokenizer_type": "AutoTokenizer",

    # Chargement
    "load_in_8bit": True,       # Réduit la VRAM (8 → 4 pour GPU < 16GB)
    "load_in_4bit": False,
    "strict": False,

    # Dataset
    "datasets": [
        {
            "path": "dataset/xcore_train.jsonl",
            "type": "alpaca",   # Format instruction/input/output
        }
    ],
    "dataset_prepared_path": "last_run_prepared",
    "val_set_size": 0.05,       # 5% pour la validation
    "output_dir": "./lora_out",

    # Séquences
    "sequence_len": 2048,
    "sample_packing": True,
    "pad_to_sequence_len": True,

    # LoRA
    "adapter": "lora",
    "lora_model_dir": None,
    "lora_r": 16,               # Rang LoRA (plus élevé = plus de params)
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_linear": True,
    "lora_fan_in_fan_out": None,
    "lora_target_modules": [
        "q_proj", "v_proj", "k_proj",
        "o_proj", "gate_proj", "down_proj", "up_proj",
    ],

    # Entraînement
    "gradient_accumulation_steps": 4,
    "micro_batch_size": 2,
    "num_epochs": 3,
    "optimizer": "adamw_bnb_8bit",
    "lr_scheduler": "cosine",
    "learning_rate": 0.0002,
    "train_on_inputs": False,   # Entraîne seulement sur le output
    "group_by_length": False,
    "bf16": "auto",
    "fp16": None,
    "tf32": False,

    # Logging
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "warmup_steps": 10,

    # WandB (optionnel)
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_id": None,

    # Déduplication + shuffle
    "seed": 42,
    "special_tokens": {
        "pad_token": "<|endoftext|>",
    },
}


TRAIN_SCRIPT = '''#!/bin/bash
# train_lora.sh — Lance le fine-tuning LoRA de DeepSeek Coder sur XCore
set -e

echo "=== XCore AI — Fine-tuning LoRA ==="
echo "Modèle    : deepseek-ai/deepseek-coder-6.7b-instruct"
echo "Dataset   : dataset/xcore_train.jsonl"
echo "Sortie    : lora_out/"
echo ""

# Vérifie Axolotl
if ! python -c "import axolotl" 2>/dev/null; then
    echo "Installation d'Axolotl..."
    pip install axolotl[flash-attn,deepspeed] -q
fi

# Lance l'entraînement
accelerate launch -m axolotl.cli.train lora_config.yaml

echo ""
echo "=== Fine-tuning terminé ==="
echo "Adaptateur LoRA sauvegardé dans : lora_out/"
echo ""
echo "Pour utiliser le modèle fine-tuné avec Ollama :"
echo "  python -m xcore_ai.dataset.lora_config --export"
'''

EXPORT_SCRIPT = '''#!/usr/bin/env python3
"""
export_to_ollama.py
--------------------
Fusionne les poids LoRA avec le modèle de base et
crée un Modelfile Ollama pour utiliser le modèle fine-tuné.
"""

from pathlib import Path

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("pip install peft transformers torch")
    exit(1)

BASE_MODEL  = "deepseek-ai/deepseek-coder-6.7b-instruct"
LORA_DIR    = "./lora_out"
MERGED_DIR  = "./merged_model"
GGUF_NAME   = "xcore-coder-6.7b"

print("Chargement du modèle de base...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Application des poids LoRA...")
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()

print(f"Sauvegarde du modèle fusionné → {MERGED_DIR}")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

# Génère le Modelfile Ollama
modelfile = f"""FROM {MERGED_DIR}
SYSTEM "You are XCore AI, an expert assistant for the XCore Python framework."
PARAMETER temperature 0.1
PARAMETER num_predict 2048
"""
Path("Modelfile").write_text(modelfile)

print("\\nPour créer le modèle Ollama :")
print(f"  ollama create {GGUF_NAME} -f Modelfile")
print(f"  ollama run {GGUF_NAME}")
print("\\nEt mettre à jour xcore_ai.yaml :")
print(f'  llm_model: "{GGUF_NAME}"')
'''


def generate_config(
    dataset_path: str = "dataset/xcore_train.jsonl",
    output_dir: str = ".",
    export: bool = False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = dict(AXOLOTL_CONFIG)
    config["datasets"][0]["path"] = dataset_path

    # Fichier YAML Axolotl
    if HAS_YAML:
        config_path = out / "lora_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"  Config Axolotl → {config_path}")
    else:
        import json
        config_path = out / "lora_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Config (JSON) → {config_path}  [install pyyaml pour le YAML]")

    # Script bash d'entraînement
    train_path = out / "train_lora.sh"
    train_path.write_text(TRAIN_SCRIPT)
    train_path.chmod(0o755)
    print(f"  Script train  → {train_path}")

    # Script d'export
    if export:
        export_path = out / "export_to_ollama.py"
        export_path.write_text(EXPORT_SCRIPT)
        print(f"  Script export → {export_path}")

    print("\n  Lancement :")
    print(f"    bash {train_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/xcore_train.jsonl")
    parser.add_argument("--output", default=".")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    generate_config(args.dataset, args.output, args.export)