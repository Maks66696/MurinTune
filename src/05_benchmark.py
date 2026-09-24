"""
05_benchmark.py — Сравнение ответов дообученной модели с реальными эталонами из test_raw.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import torch
from unsloth import FastModel

import config


def load_test_set(mode: str) -> list[dict]:
    path = os.path.join(config.DATASET_DIR, f"test_{mode}.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} пуст или не найден.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(mode: str, adapter_dir: str, n_samples: int) -> None:
    torch.cuda.empty_cache()

    test_samples = load_test_set(mode)[:n_samples]
    if not test_samples:
        print("[-] Тестовая выборка пуста.")
        return

    adapter_path = os.path.join(adapter_dir, mode)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Адаптер не найден: {adapter_path}")

    print(f"[*] Загружаю обученный адаптер: {adapter_path}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastModel.for_inference(model)

    def generate(messages, max_new_tokens=100):
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.75,          
            top_p=0.9,
            repetition_penalty=1.18,   
            do_sample=True,
        )
        return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    print("\n" + "=" * 60)
    print(f" ТЕСТ ОБУЧЕННОГО ПЕРСОНАЖА ({mode.upper()}) НА ТЕСТОВОМ СЕТЕ")
    print("=" * 60)

    for i, sample in enumerate(test_samples, 1):
        convo = sample["messages"]
        prompt_messages = convo[:-1]
        reference = convo[-1]["content"]

        ai_reply = generate(prompt_messages)

        print(f"\n[Пример {i}/{len(test_samples)}]")
        print(f"ВОПРОС:         {prompt_messages[-1]['content']}")
        print(f"РЕАЛЬНЫЙ МЕЛЛ:  {reference}")
        print(f"ОТВЕТ НЕЙРОСЕТИ:{ai_reply}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Benchmark")
    parser.add_argument("--mode", choices=config.MODES, default="raw")
    parser.add_argument("--adapter-dir", default=str(config.ADAPTERS_DIR))
    parser.add_argument("--n", type=int, default=5, help="Сколько примеров проверить")
    args = parser.parse_args()

    main(args.mode, args.adapter_dir, args.n)