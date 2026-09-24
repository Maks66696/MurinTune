"""
04_train.py — Дообучение малой LLM в стиле Меллстроя (Unsloth + QLoRA).
"""
import argparse
import os

import config


def train(mode: str, epochs: int, base_model: str, output_dir: str) -> None:
    from unsloth import FastModel, is_bf16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    train_path = os.path.join(config.DATASET_DIR, f"train_{mode}.jsonl")
    if not os.path.exists(train_path) or os.path.getsize(train_path) == 0:
        raise FileNotFoundError(
            f"{train_path} пуст или не найден. Сначала запустите:\n"
            f"  python src/03_build_dataset.py --mode {mode}"
        )

    import torch
    torch.cuda.empty_cache()

    print(f"[*] Загружаю базовую модель: {base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
        device_map="cuda:0",
    )
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)

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
            save_strategy="epoch",
            save_total_limit=2,
        ),
    )

    # Автоматически определяет токены Gemma 4 (<|turn>user\n, <|turn>model\n)
    try:
        trainer = train_on_responses_only(trainer)
    except Exception:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
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