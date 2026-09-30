"""
src/export_gguf.py — Экспорт готового адаптера в формат GGUF для LM Studio / Ollama.
"""
import os
import torch
import config
from unsloth import FastModel

ADAPTER_PATH = "adapters/raw"
OUTPUT_DIR = "models/mellstroy_gguf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()

print(f"[*] Загружаю адаптер из {ADAPTER_PATH}...")
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=config.MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="cuda:0",
)

print(f"[*] Конвертирую модель в GGUF (квантование q4_k_m)...")
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)

print("\n" + "=" * 50)
print(f"[+] УСПЕХ! GGUF-модель сохранена в: {OUTPUT_DIR}")
print("=" * 50)