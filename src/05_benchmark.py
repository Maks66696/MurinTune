"""
05_benchmark.py — Сравнение базовой модели и дообученного LoRA-адаптера
на отложенной выборке test_{mode}.jsonl.

Раньше был src/04_benchmark.py — полностью пустой файл (0 байт), то есть
проверка результата обучения тоже не была реализована. Здесь: генерация
ответов "было / стало" на реальных отложенных примерах + перплексия как
грубый количественный ориентир (чем ниже — тем увереннее модель в
эталонном стиле речи; это НЕ метрика качества/адекватности ответа,
это нужно оценивать глазами по печатаемым примерам).

Запуск:
    python src/05_benchmark.py --mode white --n 8
"""
from __future__ import annotations

import argparse
import json
import os

import config


def load_test_set(mode: str) -> list[dict]:
    path = os.path.join(config.DATASET_DIR, f"test_{mode}.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"{path} пуст или не найден. Сначала запустите:\n"
            f"  python src/03_build_dataset.py --mode {mode}"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(mode: str, adapter_dir: str, base_model: str, n_samples: int) -> None:
    import torch
    from unsloth import FastModel

    test_samples = load_test_set(mode)[:n_samples]
    if not test_samples:
        print("[-] Тестовая выборка пуста.")
        return

    adapter_path = os.path.join(adapter_dir, mode)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Адаптер не найден: {adapter_path}. Сначала запустите:\n"
            f"  python src/04_train.py --mode {mode}"
        )

    print(f"[*] Загружаю базовую модель: {base_model}")
    base, tokenizer = FastModel.from_pretrained(
        model_name=base_model, max_seq_length=config.MAX_SEQ_LENGTH, load_in_4bit=True,
    )
    FastModel.for_inference(base)

    print(f"[*] Загружаю дообученную модель с адаптером: {adapter_path}")
    tuned, _ = FastModel.from_pretrained(
        model_name=adapter_path, max_seq_length=config.MAX_SEQ_LENGTH, load_in_4bit=True,
    )
    FastModel.for_inference(tuned)

    def generate(model, messages, max_new_tokens=128):
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens,
            temperature=0.8, top_p=0.9, do_sample=True,
        )
        return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    def perplexity(model, full_conversation):
        text = tokenizer.apply_chat_template(full_conversation, tokenize=False)
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        return torch.exp(out.loss).item()

    for i, sample in enumerate(test_samples, 1):
        convo = sample["messages"]
        prompt_messages = convo[:-1]
        reference = convo[-1]["content"]

        base_reply = generate(base, prompt_messages)
        tuned_reply = generate(tuned, prompt_messages)
        ppl = perplexity(tuned, convo)

        print(f"\n===== Пример {i}/{len(test_samples)} =====")
        print(f"USER:           {prompt_messages[-1]['content']}")
        print(f"ЭТАЛОН:         {reference}")
        print(f"БАЗОВАЯ МОДЕЛЬ: {base_reply}")
        print(f"ДООБУЧЕННАЯ:    {tuned_reply}")
        print(f"Перплексия дообученной модели на эталоне: {ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Benchmark")
    parser.add_argument("--mode", choices=config.MODES, default="white")
    parser.add_argument("--adapter-dir", default=str(config.ADAPTERS_DIR))
    parser.add_argument("--base-model", default=config.BASE_MODEL)
    parser.add_argument("--n", type=int, default=10, help="Сколько примеров из test-сета проверить")
    args = parser.parse_args()

    main(args.mode, args.adapter_dir, args.base_model, args.n)
