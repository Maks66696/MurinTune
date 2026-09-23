"""
04_train.py — Дообучение малой LLM в стиле Меллстроя (Unsloth + QLoRA).

В исходном проекте это должно было жить в notebooks/train_unsloth.ipynb —
но файл был полностью пустой (0 ячеек), то есть шаг обучения не был
реализован вообще. Это скрипт вместо ноутбука: то же самое можно
запускать и в Colab через `!python src/04_train.py --mode white`, и
на своей машине с GPU.

Для двух версий персонажа обучаются ДВА ОТДЕЛЬНЫХ LoRA-адаптера поверх
одной и той же базовой модели — переключение между "white" и "raw"
происходит просто сменой адаптера, без повторной загрузки базовой модели.

Требования: GPU с 8 ГБ VRAM для Gemma 4 E2B (базовая модель по умолчанию,
см. config.py). E4B официально требует уже ~10 ГБ — на 8-гигабайтной карте
не гарантируется.

Важно про Gemma 4: у E2B/E4B были баги с общим KV-кэшем между слоями при
use_cache=False (стандартная связка QLoRA + gradient checkpointing) — на
момент написания Unsloth уже пофиксил это в своей библиотеке. Просто
держите unsloth актуальным (requirements.txt тянет последнюю версию с
GitHub), отдельно ничего чинить не нужно.

Запуск:
    python src/04_train.py --mode white
    python src/04_train.py --mode raw --epochs 5
"""
import argparse
import os

import config


def train(mode: str, epochs: int, base_model: str, output_dir: str) -> None:
    # Импорты внутри функции: unsloth/torch тяжёлые и нужны только на шаге
    # обучения — так скрипт можно хотя бы --help без GPU-окружения.
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastModel, is_bf16_supported
    from unsloth.chat_templates import get_chat_template

    train_path = os.path.join(config.DATASET_DIR, f"train_{mode}.jsonl")
    if not os.path.exists(train_path) or os.path.getsize(train_path) == 0:
        raise FileNotFoundError(
            f"{train_path} пуст или не найден. Сначала запустите:\n"
            f"  python src/03_build_dataset.py --mode {mode}"
        )

    print(f"[*] Загружаю базовую модель: {base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)

    # Gemma 4 через FastModel настраивается именованными флагами по группам
    # слоёв, а не списком target_modules (это специфика мультимодальной
    # архитектуры Gemma 4 — см. официальный рецепт Unsloth). Мы обучаем
    # только текстовую часть.
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
    )

    dataset = load_dataset("json", data_files=train_path, split="train")
    n_examples = len(dataset)
    print(f"[*] Примеров в train_{mode}.jsonl: {n_examples}")
    if n_examples < 200:
        print(
            "[!] ВНИМАНИЕ: меньше 200 примеров — для персоны на реальной "
            "речи этого обычно мало, модель рискует переобучиться на "
            "конкретных фразах вместо стиля. См. README про расширение датасета."
        )

    def formatting_func(examples):
        # Процессор сам добавит <bos> перед обучением — убираем дубль,
        # который иначе вставляет apply_chat_template (так делает и
        # официальный рецепт Unsloth для Gemma 4).
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    checkpoints_dir = os.path.join(output_dir, f"checkpoints_{mode}")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=epochs,
            learning_rate=config.LEARNING_RATE,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.RANDOM_SEED,
            output_dir=checkpoints_dir,
            report_to="none",
        ),
    )

    trainer.train()

    adapter_path = os.path.join(output_dir, mode)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[+] LoRA-адаптер сохранён: {adapter_path}")
    print(f"[+] Проверить результат: python src/05_benchmark.py --mode {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Trainer (Unsloth QLoRA)")
    parser.add_argument("--mode", choices=config.MODES, default="white")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--base-model", default=config.BASE_MODEL)
    parser.add_argument("--output-dir", default=str(config.ADAPTERS_DIR))
    args = parser.parse_args()

    train(args.mode, args.epochs, args.base_model, args.output_dir)
