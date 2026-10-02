"""
src/eval_metrics.py — Расчет честного Eval Loss и Perplexity на 110 отложенных тестах.
"""
import os
import json
import math
import torch
from unsloth import FastModel
import config

TEST_FILE = os.path.join(config.DATASET_DIR, "test_raw.jsonl")
ADAPTER_PATH = "adapters/raw"

print(f"[*] Загружаю модель для валидации на тесте...")
torch.cuda.empty_cache()

model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=config.MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="cuda:0",
)
FastModel.for_inference(model)

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Файл {TEST_FILE} не найден!")

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f if line.strip()]

print(f"[*] Оцениваю {len(test_samples)} тестовых диалогов...")

total_loss = 0.0
total_tokens = 0

model.eval()
with torch.no_grad():
    for i, sample in enumerate(test_samples):
        # Токенизируем напрямую в тензор input_ids, минуя процессор картинок
        input_ids = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).to("cuda")
        
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.item()
        
        n_tok = input_ids.shape[1]
        total_loss += loss * n_tok
        total_tokens += n_tok

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print("\n" + "=" * 55)
print(f" РЕЗУЛЬТАТЫ НА ОТЛОЖЕННОЙ ТЕСТОВОЙ ВЫБОРКЕ ({len(test_samples)} ПАР):")
print("=" * 55)
print(f"  • Validation (Eval) Loss: {avg_loss:.4f}")
print(f"  • Perplexity (PPL):        {perplexity:.2f}")
print("=" * 55)