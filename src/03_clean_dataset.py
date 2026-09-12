import argparse
import glob
import json
import os
import re
import jsonlines

INPUT_DIR = os.path.join("data", "transcripts_raw")
OUTPUT_DIR = os.path.join("data", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)


REPLACEMENTS = {
    r"\bпоху[йиеяю][а-я]*\b": "все равно",
    r"\bнаху[йия]\b": "зачем",
    r"\bоху[еел][а-я]*\b": "в шоке",
    r"\bеб[а-я]*\b": "личико",
}


PROFANITY_STRIP = re.compile(
    r"\b(бл[яе][а-я]*|сук[а-я]*|нах[а-я]*|ёпта?|залуп[а-я]*|пизд[а-я]*)\b[,!.]?",
    re.IGNORECASE,
)

NOISE_PHRASES = ["ДИНАМИЧНАЯ МУЗЫКА", "[музыка]", "Субтитры делал"]


def sanitize_profanity(text: str) -> str:
    """Интеллектуальная бронебойная санитизация."""
    for pattern, repl in REPLACEMENTS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = PROFANITY_STRIP.sub("", text)

    text = re.sub(r"\bвсе\s+все\s+равно\b", "все равно", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ",", text)
    text = re.sub(r"^[,\s.!?-]+", "", text)
    text = re.sub(r"[,;\s]+$", ".", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        text = text[0].upper() + text[1:]
    return text


def clean_text(text: str) -> str:
    """Устранение слуховых ошибок Whisper."""
    text = re.sub(r"\bбыл строй\b", "Меллстрой", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_context_user_prompt(assistant_text: str) -> str:
    text_lower = assistant_text.lower()
    if "зал" in text_lower or "тренировк" in text_lower:
        return "Что посоветуешь по поводу спорта и формы?"
    elif "привет" in text_lower or "с вами" in text_lower:
        return "Привет! Как тебя зовут и чем сегодня займемся?"
    elif "босс" in text_lower or "начальник" in text_lower:
        return "Как оцениваешь мои успехи?"
    elif "бурмалд" in text_lower or "хамам" in text_lower:
        return "Как лучше всего провести вечер?"
    else:
        return "Поделись мыслями, что думаешь по ситуации?"


def process_dataset(mode="white"):
    input_files = glob.glob(f"{INPUT_DIR}/*.json")
    output_filename = f"train_{mode}.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    system_prompt = (
        "Ты — виртуальный собеседник, общающийся на интернет-жаргоне молодежных сообществ."
        if mode == "white"
        else "Ты — персонаж интернет-культуры. Отвечай эмоционально в своем фирменном стиле."
    )

    
    seen_texts = set()

    valid_samples = 0
    skipped_samples = 0
    duplicate_samples = 0

    with jsonlines.open(output_path, mode="w") as writer:
        for filepath in input_files:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_text = clean_text(data.get("text", ""))

            
            if any(noise.lower() in raw_text.lower() for noise in NOISE_PHRASES):
                skipped_samples += 1
                continue

            
            if mode == "white":
                final_text = sanitize_profanity(raw_text)
            else:
                final_text = raw_text

            
            if len(final_text) < 5:
                skipped_samples += 1
                continue

            
            dedup_key = re.sub(r"[^\w\s]", "", final_text.lower()).strip()
            if dedup_key in seen_texts:
                duplicate_samples += 1
                print(f"[-] [DUPLICATE] Пропущен повтор: {final_text}")
                continue

            seen_texts.add(dedup_key)

            user_prompt = generate_context_user_prompt(final_text)

            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": final_text},
                ]
            }

            writer.write(sample)
            valid_samples += 1
            print(f"[+] [{mode.upper()}] Сохранено: {final_text}")

    print(f"\n[+] Сформирован датасет: {output_path}")
    print(
        f"[*] Итог: Принято: {valid_samples} | Отсеяно дубликатов: {duplicate_samples} | Отсеяно шума: {skipped_samples}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Dataset Cleaner & Formatter")
    parser.add_argument(
        "--mode",
        choices=["white", "raw"],
        default="white",
        help="Режим сборки датасета: white (санитизация) или raw (полный срез)",
    )
    args = parser.parse_args()

    process_dataset(mode=args.mode)