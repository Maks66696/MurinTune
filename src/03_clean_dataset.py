import argparse
import base64
import glob
import json
import os
import re
import jsonlines

INPUT_DIR = os.path.join("data", "transcripts_raw")
OUTPUT_DIR = os.path.join("data", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)


_ENCODED_HARD_FILTER = (
    "XGIo0YXRg9C5W9C10ZHRj9C4XVvQsC3Rj10qfNCw0L3QsNC7KD8h0LjQt3zQuNGC0Lh80L7QMy"
    "lb0LAt0Y9dKnzQsNC90YPRgdKbfcC+0YfQulvQvtCw0LXRg11b0LAt0Y9dKnzQtNGA0L7Rh1vQ"
    "sC3Rj10qfNGI0LvRjtGFW9CwLdGPXSp80YjQsNC70LDQslvQsC3Rj10qfNC/0LjQtNC+0YJb0L"
    "At0Y9dKnzQv9C40LTQsNGAW9CwLdGPXSp80LPQsNC90LTQvtC9W9CwLdGPXSp80LPQvtC90LTQ"
    "vtC9W9CwLdGPXSp80LzQuNC90LXRgjtb0LAt0Y9dKnzQvtGC0YHQvtGBW9CwLdGPXSp80YfQu9"
    "C10L1b0LAt0Y9dKnzRhtC10LvQulvQsC3Rj10qfNGB0LjRgdGM0Lpb0LAt0Y9dKnzQs9C+0LLQ"
    "vdvQsC3Rj10qfNC00LXRgNGM0L1b0LAt0Y9dKnzQvtCx0L7RgdGAW9CwLdGPXSp80LzRgNCw0L"
    "db0LAt0Y9dKnzRgtCy0LDRgNGMW9CwLdGPXSopXGI="
)

_ENCODED_STRIP_FILTER = (
    "XGIo0LHQu1vRj9C1XVvQsC3Rj10qfNGB0YPQu1vQsC3Rj10qfNC90LDRhVvQsC3Rj10qfNGR0L"
    "/RgtCwP3zQt9Cw0LvRg9C/W9CwLdGPXSp80L/QuNC30LRb0LAt0Y9dKilcYlssIS5dPz8="
)


_ENCODED_REPLACEMENTS = (
    "eyJcXGLQv9C+0YXRg1vQudC40LXRj9GOXVvQsC3Rj10qXFxiIjogItCy0YHQtSDRgNCw0LLQvdC+"
    "IiwgIlxcYtC90LDRhdGDW9C50Y9dXFxiIjogItC30LDRh9C10LwiLCAiXFxi0LTQvtGF0YNb0Y/Q"
    "tV1b0LAt0Y9dKlxcYiIjogItC80L3QvtCz0L4iLCAiXFxi0L3QuNGF0YNb0Y/QtV1b0LAt0Y9dKl"
    "xcYiIjogItC90LjRh9C10LPQviIsICJcXGLQvtGF0YNb0LXQtdC7XVvQsC3Rj10qXFxiIjogItCy"
    "INGI0L7QutC1IiwgIlxcYtC10LFb0LAt0Y9dKlxcYiIjogItGA0LDQt9C90LXRgSJ9"
)


HARD_TOXIC_PATTERN = re.compile(
    base64.b64decode(_ENCODED_HARD_FILTER).decode("utf-8"), re.IGNORECASE
)
PROFANITY_STRIP = re.compile(
    base64.b64decode(_ENCODED_STRIP_FILTER).decode("utf-8"), re.IGNORECASE
)
REPLACEMENTS = json.loads(
    base64.b64decode(_ENCODED_REPLACEMENTS).decode("utf-8")
)

NOISE_PHRASES = ["ДИНАМИЧНАЯ МУЗЫКА", "[музыка]", "Субтитры делал"]


def sanitize_profanity(text: str) -> str:
    """Мягкая замена и зачистка слов-паразитов."""
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
    skipped_noise = 0
    dropped_toxic = 0
    duplicate_samples = 0

    with jsonlines.open(output_path, mode="w") as writer:
        for filepath in input_files:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_text = clean_text(data.get("text", ""))

            # Отсекаем технический шум Whisper
            if any(noise.lower() in raw_text.lower() for noise in NOISE_PHRASES):
                skipped_noise += 1
                continue

            # В РЕЖИМЕ WHITE: Проверяем на тяжелый токсичный бан-лист
            if mode == "white" and HARD_TOXIC_PATTERN.search(raw_text):
                dropped_toxic += 1
                print(f"[-] [HARD FILTER] Отклонен неприемлемый контент: {raw_text}")
                continue

            # Санитизируем оставшийся текст
            if mode == "white":
                final_text = sanitize_profanity(raw_text)
            else:
                final_text = raw_text

            if len(final_text) < 5:
                skipped_noise += 1
                continue

            # Дедупликация
            dedup_key = re.sub(r"[^\w\s]", "", final_text.lower()).strip()
            if dedup_key in seen_texts:
                duplicate_samples += 1
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

    print(f"\n==========================================")
    print(f"[+] Сформирован датасет: {output_path}")
    print(f"[*] Принято образцов в базу: {valid_samples}")
    print(f"[-] Отсеяно тяжелого 18+ контента: {dropped_toxic}")
    print(f"[-] Отсеяно дубликатов: {duplicate_samples}")
    print(f"[-] Отсеяно шума: {skipped_noise}")
    print(f"==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Dataset Cleaner & Formatter")
    parser.add_argument(
        "--mode",
        choices=["white", "raw"],
        default="white",
        help="Режим: white (фильтрация) или raw (полный срез)",
    )
    args = parser.parse_args()

    process_dataset(mode=args.mode)