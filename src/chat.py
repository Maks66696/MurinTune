"""
src/chat.py — Живой интерактивный чат с обученным Меллстроем.
"""
import config
from unsloth import FastModel
import torch

ADAPTER_PATH = "adapters/raw"

torch.cuda.empty_cache()

print("[*] Загружаю персонажа...")
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=config.MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="cuda:0",
)
FastModel.for_inference(model)

print("\n" + "=" * 55)
print(" ЧАТ С МЕЛЛСТРОЕМ ЗАПУЩЕН!")
print(" Пиши реплику, задавай вопросы (для выхода напиши 'exit')")
print("=" * 55)

history = [
    {"role": "system", "content": config.SYSTEM_PROMPTS["raw"]}
]

while True:
    try:
        user_msg = input("\nТы: ").strip()
    except (KeyboardInterrupt, EOFError):
        break

    if not user_msg:
        continue
    if user_msg.lower() in ("exit", "quit", "выход"):
        print("\n[!] Выход из чата.")
        break

    history.append({"role": "user", "content": user_msg})

    inputs = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=140,
        temperature=0.75,
        top_p=0.9,
        repetition_penalty=1.18,
        do_sample=True,
    )

    reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nМеллстрой: {reply}")

    history.append({"role": "assistant", "content": reply})
    
    if len(history) > 7:
        history = [history[0]] + history[-6:]