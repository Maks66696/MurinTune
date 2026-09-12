import argparse
import glob
import hashlib
import json
import os
import re
import jsonlines

INPUT_DIR = os.path.join("data", "transcripts_raw")
OUTPUT_DIR = os.path.join("data", "dataset")
FILTERS_FILE = os.path.join("data", "filters.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

GENERIC_USER_PROMPTS = [
    "Что думаешь по этому поводу?",
    "Какие новости, друн?",
    "Что интересного расскажешь?",
    "Как настроение сегодня?",
    "Поделись мыслями, как дела?",
    "Что там у тебя происходит?",
    "Есть какой-нибудь совет на сегодня?",
    "Что скажешь на это?",
]

DEFAULT_FILTERS = {
    "protected_fragments": [],
    "blocked_substrings": [],
    "blocked_stems": [],
    "gambling_terms": [],
    "safe_analogs": [],
    "safe_ochko_forms": [],
    "target_analogs": {},
    "replacements": {},
    "noise_phrases": [],
}


def load_filters():
    if not os.path.exists(FILTERS_FILE):
        print(f"[!] Файл фильтров не найден: {FILTERS_FILE}")
        return DEFAULT_FILTERS

    try:
        with open(FILTERS_FILE, "r", encoding="utf-8") as file:
            filters = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[!] Не удалось загрузить фильтры: {exc}")
        return DEFAULT_FILTERS

    result = DEFAULT_FILTERS.copy()
    result.update(filters)
    return result


FILTERS = load_filters()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_forbidden_words(text: str) -> bool:
    normalized = text.lower().replace("ё", "е")
    words = re.findall(r"[а-яa-z]+", normalized)

    protected_fragments = FILTERS["protected_fragments"]
    blocked_substrings = FILTERS["blocked_substrings"]
    blocked_stems = FILTERS["blocked_stems"]
    gambling_terms = FILTERS["gambling_terms"]
    target_analogs = FILTERS["target_analogs"]

    for word in words:
        if any(fragment in word for fragment in protected_fragments):
            continue

        if any(fragment in word for fragment in blocked_substrings):
            return True

        if any(stem in word for stem in blocked_stems):
            return True

        if any(term in word for term in gambling_terms):
            return True

        for target, safe_words in target_analogs.items():
            if target in word:
                if not any(safe in word for safe in safe_words):
                    return True

    return False


def sanitize_text(text: str) -> str:
    replacements = FILTERS.get("replacements", {})

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])([^\s])", r"\1 \2", text)
    text = re.sub(r"^[,\s.!?-]+", "", text)
    text = re.sub(r"[,\s;]+$", ".", text)
    text = text.strip()

    if text:
        text = text[0].upper() + text[1:]

    return text


def clean_text(text: str) -> str:
    text = re.sub(r"\bбыл строй\b", "Меллстрой", text, flags=re.IGNORECASE)
    return normalize_text(text)


def contains_noise(text: str) -> bool:
    lower_text = text.lower()
    return any(phrase.lower() in lower_text for phrase in FILTERS["noise_phrases"])


def generate_context_user_prompt(assistant_text: str) -> str:
    text_lower = assistant_text.lower()

    context_rules = [
        (["бабк", "деньг", "рулит", "мутит", "кошельк"], "Как там с финансами и движухой?"),
        (["сын", "сыр", "сочит", "друн"], "Что там у тебя за истории происходят?"),
        (["лагер", "поехал", "дело"], "Куда ты собрался и какие планы?"),
        (["конфетк", "танцу", "девочк", "детк"], "Что за трек или девчонка там у тебя?"),
        (["иди", "уход", "выгон"], "Мне остаться или лучше уйти?"),
        (["зал", "спорт", "трен"], "Что посоветуешь по поводу спорта и формы?"),
        (["привет", "с вами"], "Привет! Как тебя зовут и чем сегодня займемся?"),
        (["босс", "начальн"], "Как оцениваешь мои успехи?"),
        (["бурмалд", "хамам"], "Как лучше всего провести вечер?"),
    ]

    for keywords, prompt in context_rules:
        if any(keyword in text_lower for keyword in keywords):
            return prompt

    digest = hashlib.sha256(assistant_text.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(GENERIC_USER_PROMPTS)
    return GENERIC_USER_PROMPTS[index]


def make_dedup_key(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def process_dataset(mode: str = "white") -> None:
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    if not input_files:
        print(f"[!] JSON-файлы не найдены: {INPUT_DIR}")
        return

    output_filename = f"train_{mode}.jsonl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if mode == "white":
        system_prompt = "Ты — виртуальный собеседник, общающийся на интернет-жаргоне молодежных сообществ."
    else:
        system_prompt = "Ты — персонаж интернет-культуры. Отвечай эмоционально в своем фирменном стиле."

    seen_texts = set()

    valid_samples = 0
    skipped_noise = 0
    dropped_toxic = 0
    duplicate_samples = 0
    invalid_files = 0

    with jsonlines.open(output_path, mode="w") as writer:
        for filepath in input_files:
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except (json.JSONDecodeError, OSError) as exc:
                invalid_files += 1
                continue

            raw_text = data.get("text", "")

            if not isinstance(raw_text, str):
                skipped_noise += 1
                continue

            raw_text = clean_text(raw_text)

            if not raw_text or contains_noise(raw_text):
                skipped_noise += 1
                continue

            if mode == "white":
                final_text = sanitize_text(raw_text)
                if has_forbidden_words(final_text):
                    dropped_toxic += 1
                    continue
            else:
                final_text = raw_text

            if len(final_text) < 5:
                skipped_noise += 1
                continue

            dedup_key = make_dedup_key(final_text)

            if not dedup_key or dedup_key in seen_texts:
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
            print(f"[+] [{mode.upper()}] {user_prompt} -> {final_text}")

    print("\n==========================================")
    print(f"[+] Сформирован датасет: {output_path}")
    print(f"[*] Обработано JSON-файлов: {len(input_files)}")
    print(f"[+] Принято валидных образцов: {valid_samples}")
    print(f"[-] Отсеяно запрещенного контента: {dropped_toxic}")
    print(f"[-] Отсеяно дубликатов: {duplicate_samples}")
    print(f"[-] Отсеяно служебного шума: {skipped_noise}")
    print("==========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Cleaner & Formatter")
    parser.add_argument(
        "--mode",
        choices=["white", "raw"],
        default="white",
        help="Режим обработки: white или raw",
    )
    args = parser.parse_args()
    process_dataset(args.mode)


if __name__ == "__main__":
    main()