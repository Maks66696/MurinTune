"""
src/clean_dataset.py — Хирургическая постобработка датасета train_raw.jsonl.
Исправляет фонетические ослышки Whisper (кадик -> казик) и срезает
случайные перебивки ведущих по грамматическим маркерам.
"""
import re
import jsonlines
import os

DATASET_PATH = "data/dataset/train_raw.jsonl"
BACKUP_PATH = "data/dataset/train_raw.jsonl.bak"

# 1. Словарь фонетических ошибок Whisper -> правильный сленг Мелла
PHONETIC_REPLACEMENTS = [
    (r"\bрекламен кадик\b", "рекламил казик"),
    (r"\bкадик\b", "казик"),
    (r"\bкадика\b", "казика"),
    (r"\bкадику\b", "казику"),
    (r"\bкадике\b", "казике"),
    (r"\bебал головой обставал\b", "ебашил головой об стол"),
    (r"\bобставал\b", "об стол"),
    (r"\bбыл строй\b", "Меллстрой"),
    (r"\bмел строй\b", "Меллстрой"),
    (r"\bказино стрима\b", "казино-стримы"),
    (r"\bказино стримы\b", "казино-стримы"),
    (r"\bбурмал да\b", "бурмалда"),
    (r"\bбо ровка\b", "боровка"),
    (r"\s+([,.!?])", r"\1"),  # убираем пробелы перед знаками препинания
]

# 2. Паттерн вопроса ведущего во 2-м лице в начале реплики
HOST_QUESTION_PATTERN = re.compile(
    r"^[^\.\?\!]*\b(тебя|ты|тебе|тобой|твой|твоя|твое|твои)\b[^\.\?\!]*\?\s*",
    re.IGNORECASE
)

# 3. Перебивки ведущего в конце фразы ("то есть в целом история тебя запомнит...")
HOST_TAIL_PATTERN = re.compile(
    r"\s+то есть в целом история тебя запомнит в целом.*$",
    re.IGNORECASE
)


def clean_assistant_text(text: str) -> str:
    # 1. Срезаем вопрос ведущего на "ты" в начале реплики
    text = HOST_QUESTION_PATTERN.sub("", text).strip()

    # 2. Срезаем перебивку ведущего в конце (если есть)
    text = HOST_TAIL_PATTERN.sub("", text).strip()

    # 3. Применяем фонетические замены сленга
    for pattern, replacement in PHONETIC_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 4. Косметика: убираем двойные пробелы и делаем первую букву заглавной
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]

    return text


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[-] Файл {DATASET_PATH} не найден!")
        return

    # Делаем резервную копию оригинала
    if not os.path.exists(BACKUP_PATH):
        import shutil
        shutil.copyfile(DATASET_PATH, BACKUP_PATH)
        print(f"[*] Создан бэкап: {BACKUP_PATH}")

    with jsonlines.open(DATASET_PATH) as reader:
        records = list(reader)

    print(f"[*] Обрабатываю {len(records)} диалогов...")
    modified_count = 0

    cleaned_records = []
    for row in records:
        messages = row.get("messages", [])
        modified = False

        for msg in messages:
            if msg.get("role") == "assistant":
                old_content = msg["content"]
                new_content = clean_assistant_text(old_content)
                if new_content != old_content:
                    msg["content"] = new_content
                    modified = True

        if modified:
            modified_count += 1

        cleaned_records.append(row)

    with jsonlines.open(DATASET_PATH, mode="w") as writer:
        writer.write_all(cleaned_records)

    print("\n" + "=" * 60)
    print(f"[+] УСПЕШНО ОТПОЛИРОВАНО РЕПЛИК: {modified_count} из {len(records)}")
    print(f"[+] Обновленный датасет записан в: {DATASET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()