"""
src/benchmark_scientific.py — Честный научный бенчмарк (Response-Only Loss & PPL)
с запуском в изолированных подпроцессах для 100% защиты от OOM и MemoryError.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import gc
import torch

import config

TEST_FILE = os.path.join(config.DATASET_DIR, "test_raw.jsonl")


def evaluate_single_model(model_name: str):
    from unsloth import FastModel

    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Файл {TEST_FILE} не найден!")

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    # Загрузка с защитой оперативной памяти (RAM)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    FastModel.for_inference(model)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for sample in test_samples:
            convo = sample["messages"]
            prompt_msgs = convo[:-1]  # System + User

            # Токенизируем вопрос для точной границы
            prompt_ids = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            prompt_len = prompt_ids.shape[1]

            # Токенизируем весь диалог целиком
            full_ids = tokenizer.apply_chat_template(
                convo,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to("cuda")

            # Маскируем всё, кроме ответа ассистента (-100 игнорируется при расчете Loss)
            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            outputs = model(input_ids=full_ids, labels=labels)
            loss = outputs.loss.item()

            resp_tokens = full_ids.shape[1] - prompt_len
            if resp_tokens > 0:
                total_loss += loss * resp_tokens
                total_tokens += resp_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    ppl = math.exp(avg_loss) if avg_loss > 0 else 0

    # Передаем метрики в родительский процесс
    print(f"__METRICS_JSON__{json.dumps({'loss': avg_loss, 'ppl': ppl})}__METRICS_JSON__")


def main():
    parser = argparse.ArgumentParser(description="Scientific Benchmark")
    parser.add_argument("--target", type=str, default=None, help="Модель для оценки")
    args = parser.parse_args()

    if args.target:
        evaluate_single_model(args.target)
        return

    # Формируем список моделей
    # Формируем список моделей v1.0
    MODELS_TO_TEST = [
        ("Базовая Gemma-4-E2B (Base)", config.BASE_MODEL),
    ]
    if os.path.exists("adapters/checkpoints_raw/checkpoint-131"):
        MODELS_TO_TEST.append(
            ("MurinTune v1.0 (Эпоха 1)", "adapters/checkpoints_raw/checkpoint-131")
        )
    if os.path.exists("adapters/checkpoints_raw/checkpoint-262"):
        MODELS_TO_TEST.append(
            ("MurinTune v1.0 (Эпоха 2)", "adapters/checkpoints_raw/checkpoint-262")
        )
    if os.path.exists("adapters/raw"):
        MODELS_TO_TEST.append(
            ("MurinTune v1.0 (Эпоха 3 / Финал)", "adapters/raw")
        )

    print("=" * 70)
    print(" НАУЧНЫЙ БЕНЧМАРК (Response-Only Loss & PPL на 110 отложенных тестах)")
    print(" Запуск в изолированных подпроцессах...")
    print("=" * 70)

    results = []

    for label, path in MODELS_TO_TEST:
        print(f"\n[*] Тестирую: {label}...")
        cmd = [sys.executable, __file__, "--target", path]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if "__METRICS_JSON__" in res.stdout:
            raw_json = res.stdout.split("__METRICS_JSON__")[1]
            metrics = json.loads(raw_json)
            loss, ppl = metrics["loss"], metrics["ppl"]
            results.append((label, loss, ppl))
            print(f"[+] Результат: Eval Loss = {loss:.4f} | PPL = {ppl:.2f}")
        else:
            print(f"[-] Ошибка выполнения для {label}:")
            print(res.stderr or res.stdout)

    if results:
        print("\n" + "=" * 70)
        print(f"{'Модель / Чекпоинт':<35} | {'Eval Loss':<12} | {'Perplexity (PPL)':<16}")
        print("-" * 70)
        base_loss = results[0][1]
        for label, loss, ppl in results:
            diff_str = f"({loss - base_loss:+.4f})" if label != results[0][0] else "(Baseline)"
            print(f"{label:<35} | {loss:<6.4f} {diff_str:<7} | {ppl:<16.2f}")
        print("=" * 70)


if __name__ == "__main__":
    main()