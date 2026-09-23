"""
03_build_dataset.py — Сборка train/test JSONL из сырых транскриптов.

Что изменено по сравнению со старым 03_clean_dataset.py:

1. РАНЬШЕ: весь текст одного видео (data['text']) становился ОДНОЙ репликой
   assistant. Для роликов с музыкой/монтажом это превращало реплику в
   нечитаемую "простыню" и резко уменьшало число валидных примеров
   (из 30 видео получилось 8 сэмплов).
   ТЕПЕРЬ: используем покадровые chunks (start/end/text) из 02_transcribe.py
   и склеиваем их в реплики разговорного размера (см. config.MERGE_GAP_SECONDS
   и config.MAX_UTTERANCE_CHARS) — из тех же видео получится в разы больше
   валидных, более естественных по длине примеров.

2. РАНЬШЕ: не было train/test сплита вообще (test.jsonl всегда пустой).
   ТЕПЕРЬ: сплит есть, и делается ПО ID ИСХОДНОГО ВИДЕО, а не по строкам —
   иначе почти одинаковые фразы из одного ролика попадут и в train, и в
   test, и метрики на test будут врать.

3. РАНЬШЕ: если data/filters.json отсутствовал, скрипт молча продолжал
   работать без единого реального фильтра (white-версия получалась
   "белой" только по системному промпту, а не по содержанию).
   ТЕПЕРЬ: это явное предупреждение с инструкцией, и для --mode white
   без --allow-empty-filters скрипт не даст сгенерировать датасет.

filters.json специально в .gitignore (может содержать чувствительные
списки слов) — создайте его локально по образцу data/filters.example.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from collections import defaultdict

import jsonlines

import config

INPUT_DIR = config.TRANSCRIPTS_DIR
OUTPUT_DIR = config.DATASET_DIR

DEFAULT_FILTERS = {
    "protected_fragments": [],
    "blocked_substrings": [],
    "blocked_stems": [],
    "gambling_terms": [],
    "target_analogs": {},
    "replacements": {},
    "noise_phrases": [],
}

# Более широкий пул синтетических реплик пользователя — используется как
# запасной вариант, когда не сработало ни одно контекстное правило ниже.
GENERIC_USER_PROMPTS = [
    "Что думаешь по этому поводу?",
    "Какие новости, друн?",
    "Что интересного расскажешь?",
    "Как настроение сегодня?",
    "Поделись мыслями, как дела?",
    "Что там у тебя происходит?",
    "Есть какой-нибудь совет на сегодня?",
    "Что скажешь на это?",
    "Го стрим, что сегодня в планах?",
    "Расскажи, как оно там у тебя.",
    "Го подробнее, что произошло?",
    "А что чат думает, спроси у них.",
]

CONTEXT_RULES = [
    (["бабк", "деньг", "рулит", "мутит", "кошельк"], "Как там с финансами и движухой?"),
    (["сын", "сыр", "сочит", "друн"], "Что там у тебя за истории происходят?"),
    (["лагер", "поехал", "дело"], "Куда ты собрался и какие планы?"),
    (["конфетк", "танцу", "девочк", "детк"], "Что за трек или девчонка там у тебя?"),
    (["иди", "уход", "выгон"], "Мне остаться или лучше уйти?"),
    (["зал", "спорт", "трен"], "Что посоветуешь по поводу спорта и формы?"),
    (["привет", "с вами"], "Привет! Как тебя зовут и чем сегодня займёмся?"),
    (["босс", "начальн"], "Как оцениваешь мои успехи?"),
    (["бурмалд", "хамам"], "Как лучше всего провести вечер?"),
]


def load_filters() -> dict:
    if not os.path.exists(config.FILTERS_FILE):
        print("=" * 70)
        print(f"[!] Файл фильтров не найден: {config.FILTERS_FILE}")
        print("    Скопируйте data/filters.example.json -> data/filters.json")
        print("    и заполните списки под свои требования к white-версии.")
        print("    Без него white-датасет НЕ будет реально отфильтрован!")
        print("=" * 70)
        return DEFAULT_FILTERS

    with open(config.FILTERS_FILE, "r", encoding="utf-8") as file:
        filters = json.load(file)

    result = DEFAULT_FILTERS.copy()
    result.update(filters)
    return result


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_forbidden_words(text: str, filters: dict) -> bool:
    normalized = text.lower().replace("ё", "е")
    words = re.findall(r"[а-яa-z]+", normalized)

    protected = filters["protected_fragments"]
    blocked_substrings = filters["blocked_substrings"]
    blocked_stems = filters["blocked_stems"]
    gambling_terms = filters["gambling_terms"]
    target_analogs = filters["target_analogs"]

    for word in words:
        if any(fragment in word for fragment in protected):
            continue
        if any(fragment in word for fragment in blocked_substrings):
            return True
        if any(stem in word for stem in blocked_stems):
            return True
        if any(term in word for term in gambling_terms):
            return True
        for target, safe_words in target_analogs.items():
            if target in word and not any(safe in word for safe in safe_words):
                return True
    return False


def sanitize_text(text: str, filters: dict) -> str:
    for pattern, replacement in filters.get("replacements", {}).items():
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


def contains_noise(text: str, filters: dict) -> bool:
    lower_text = text.lower()
    return any(phrase.lower() in lower_text for phrase in filters["noise_phrases"])


def iter_utterances(transcript: dict):
    """
    Склеивает покадровые chunks в реплики разговорного размера.
    Падаем обратно на data['text'] целиком, если chunks нет (старые
    транскрипты или ручной ввод без таймкодов).
    """
    chunks = transcript.get("chunks") or []
    if not chunks:
        text = transcript.get("text", "")
        if text.strip():
            yield text.strip()
        return

    buffer, buffer_len, prev_end = [], 0, None

    for ch in chunks:
        text = (ch.get("text") or "").strip()
        if not text:
            continue

        gap = (ch["start"] - prev_end) if prev_end is not None else 0.0
        would_overflow = buffer_len + len(text) + 1 > config.MAX_UTTERANCE_CHARS

        if buffer and (gap > config.MERGE_GAP_SECONDS or would_overflow):
            yield " ".join(buffer).strip()
            buffer, buffer_len = [], 0

        buffer.append(text)
        buffer_len += len(text) + 1
        prev_end = ch["end"]

    if buffer:
        yield " ".join(buffer).strip()


def generate_context_user_prompt(assistant_text: str) -> str:
    text_lower = assistant_text.lower()
    for keywords, prompt in CONTEXT_RULES:
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


def split_train_test(video_ids: list[str]) -> set[str]:
    """Детерминированный сплит по id видео (hash-based, без random.seed-гонок)."""
    n_test = max(config.MIN_TEST_SAMPLES, round(len(video_ids) * config.TEST_SIZE))
    n_test = min(n_test, len(video_ids))

    scored = sorted(
        video_ids,
        key=lambda vid: hashlib.sha256(
            f"{config.RANDOM_SEED}:{vid}".encode()
        ).hexdigest(),
    )
    return set(scored[:n_test])


def process_dataset(mode: str, allow_empty_filters: bool) -> None:
    filters = load_filters()
    filters_are_empty = filters == DEFAULT_FILTERS

    if mode == "white" and filters_are_empty and not allow_empty_filters:
        print(
            "[-] Останавливаюсь: для --mode white нужен настоящий data/filters.json "
            "(или явно передайте --allow-empty-filters, если вы это тестовый прогон)."
        )
        return

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not input_files:
        print(f"[!] JSON-транскрипты не найдены: {INPUT_DIR}")
        return

    system_prompt = config.SYSTEM_PROMPTS[mode]

    # Собираем валидные сэмплы по video_id, чтобы потом честно разбить на train/test
    samples_by_video: dict[str, list[dict]] = defaultdict(list)
    seen_texts: set[str] = set()

    stats = {"valid": 0, "noise": 0, "toxic": 0, "dup": 0, "invalid_files": 0}

    for filepath in input_files:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                transcript = json.load(file)
        except (json.JSONDecodeError, OSError):
            stats["invalid_files"] += 1
            continue

        video_id = transcript.get("id") or os.path.splitext(os.path.basename(filepath))[0]

        for raw_utt in iter_utterances(transcript):
            text = clean_text(raw_utt)

            if not text or len(text) < config.MIN_UTTERANCE_CHARS or contains_noise(text, filters):
                stats["noise"] += 1
                continue

            if mode == "white":
                final_text = sanitize_text(text, filters)
                if has_forbidden_words(final_text, filters):
                    stats["toxic"] += 1
                    continue
            else:
                final_text = text

            if len(final_text) < config.MIN_UTTERANCE_CHARS:
                stats["noise"] += 1
                continue

            dedup_key = make_dedup_key(final_text)
            if not dedup_key or dedup_key in seen_texts:
                stats["dup"] += 1
                continue
            seen_texts.add(dedup_key)

            user_prompt = generate_context_user_prompt(final_text)
            samples_by_video[video_id].append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": final_text},
                    ]
                }
            )
            stats["valid"] += 1

    if not samples_by_video:
        print("[-] Не получилось собрать ни одного валидного примера.")
        return

    test_video_ids = split_train_test(list(samples_by_video.keys()))

    train_path = os.path.join(OUTPUT_DIR, f"train_{mode}.jsonl")
    test_path = os.path.join(OUTPUT_DIR, f"test_{mode}.jsonl")

    n_train = n_test = 0
    with jsonlines.open(train_path, mode="w") as train_writer, \
         jsonlines.open(test_path, mode="w") as test_writer:
        for video_id, samples in samples_by_video.items():
            writer = test_writer if video_id in test_video_ids else train_writer
            for sample in samples:
                writer.write(sample)
                if writer is test_writer:
                    n_test += 1
                else:
                    n_train += 1

    print("\n==========================================")
    print(f"[+] train: {train_path} ({n_train} примеров)")
    print(f"[+] test:  {test_path} ({n_test} примеров)")
    print(f"[*] Видео обработано: {len(input_files) - stats['invalid_files']}")
    print(f"[-] Отсеяно как мусор/пусто: {stats['noise']}")
    print(f"[-] Отсеяно как запрещённый контент: {stats['toxic']}")
    print(f"[-] Отсеяно дубликатов: {stats['dup']}")
    if stats["invalid_files"]:
        print(f"[-] Битых JSON-файлов пропущено: {stats['invalid_files']}")
    if filters_are_empty:
        print("[!] ВНИМАНИЕ: фильтры пустые — white-датасет НЕ реально отфильтрован.")
    print("==========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="MurinTune Dataset Builder")
    parser.add_argument("--mode", choices=config.MODES, default="white")
    parser.add_argument(
        "--allow-empty-filters",
        action="store_true",
        help="Разрешить сборку white-датасета без реального data/filters.json (для тестового прогона).",
    )
    args = parser.parse_args()
    process_dataset(args.mode, args.allow_empty_filters)


if __name__ == "__main__":
    main()
