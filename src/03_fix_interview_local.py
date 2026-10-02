"""
src/03_fix_interview_local.py — Локальный умный сборщик историй из Вписки.
Работает прямо на твоем ПК, не триггерит никаких облачных фильтров,
удаляет запрещенку и склеивает разорванные слова.
"""
import json
import os
import re
import jsonlines

INPUT_JSON = "data/transcripts_raw/R6rhxJjVNCU.json"
TIKTOK_DATASET = "data/dataset/train_raw.jsonl.bak"  # берем чистый исходник из тиктоков
OUTPUT_DATASET = "data/dataset/train_raw.jsonl"

SYSTEM_PROMPT = (
    "Ты — популярный стример Меллстрой. Веди диалог как живой парень на стриме: "
    "используй свой сленг, шути, подкалывай зрителя, общайся на «ты». "
    "Не будь вежливым ботом-помощником. Отвечай коротко и с юмором. Без лишних матов."
)

# 1. ЖЕСТКИЙ БАН: строки с этими словами удаляются целиком
CRIMINAL_BLACKLIST = [
    r"педофил",
    r"изнасил",
    r"сожгу",
    r"убью",
    r"прах",
    r"usmall",
    r"gta.*rp",
    r"гта.*рп",
    r"промокод",
    r"авиасейлс",
]

# 2. Исправление сленга и частых ошибок Whisper
FIXES = [
    (r"\bрекламен кадик\b", "рекламил казик"),
    (r"\bкадик\b", "казик"),
    (r"\bкадика\b", "казика"),
    (r"\bкадику\b", "казику"),
    (r"\bкадике\b", "казике"),
    (r"\bкайзик\b", "казик"),
    (r"\bебал головой обставал\b", "ебашил головой об стол"),
    (r"\bобставал\b", "об стол"),
    (r"\bбыл строй\b", "Меллстрой"),
    (r"\bмел строй\b", "Меллстрой"),
    (r"\bказино стрима\b", "казино-стримы"),
    (r"\bказино стримы\b", "казино-стримы"),
    (r"\bдоманию\b", "лудоманию"),
    (r"\bбурмал да\b", "бурмалда"),
    (r"\bбо ровка\b", "боровка"),
    (r"\bлежит очко\b", "лижут очко"),
]

# Паттерны вопросов ведущих для отделения ответа Мелла
HOST_QUESTION_RE = re.compile(
    r"(?:^|\.\s+)(Ты\s+понимаешь[^\?\!\.]*\?|"
    r"Паришься\s+ли\s+ты[^\?\!\.]*\?|"
    r"Как\s+думаешь[^\?\!\.]*\?|"
    r"Ты\s+себя\s+считаешь[^\?\!\.]*\?|"
    r"Ты\s+хотел\s+бы[^\?\!\.]*\?|"
    r"А\s+в\s+толпе[^\?\!\.]*\?|"
    r"Счастливым\s+ты\s+можешь[^\?\!\.]*\?|"
    r"Что\s+в\s+этом\s+мире[^\?\!\.]*\?)",
    re.IGNORECASE,
)


def is_toxic(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in CRIMINAL_BLACKLIST)


def clean_text(text: str) -> str:
    for pat, rep in FIXES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_smart_dataset():
    if not os.path.exists(INPUT_JSON):
        print(f"[-] Файл {INPUT_JSON} не найден!")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    print(f"[*] Загружено {len(chunks)} сегментов транскрипта...")

    # Шаг 1. Сквозная склейка всего текста интервью с сохранением знаков
    full_interview_tokens = []
    for ch in chunks:
        t = ch.get("text", "").strip()
        if t:
            full_interview_tokens.append(t)

    full_text = " ".join(full_interview_tokens)
    full_text = clean_text(full_text)

    # Шаг 2. Нарезка на абзацы СТРОГО по законченным предложениям
    # Ищем границы предложений (. ! ?)
    raw_sentences = re.split(r"(?<=[.!?])\s+", full_text)
    print(f"[*] Выделено {len(raw_sentences)} законченных предложений.")

    clean_dialogues = []
    current_passage = []
    current_word_count = 0

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Если в предложении есть запрещенка (педофил, жесть) — пропускаем целиком
        if is_toxic(sent):
            continue

        current_passage.append(sent)
        current_word_count += len(sent.split())

        # Формируем законченную историю на 60-120 слов
        if current_word_count >= 80:
            story = " ".join(current_passage).strip()
            current_passage = []
            current_word_count = 0

            # Проверяем, есть ли внутри вопрос ведущего
            match = HOST_QUESTION_RE.search(story)
            if match:
                question_text = match.group(1).strip()
                # Ответ — всё, что идет после вопроса ведущего
                answer_text = story[match.end():].strip()
                if not question_text.endswith("?"):
                    question_text += "?"
            else:
                question_text = "Расскажи, как оно вообще было на самом деле?"
                answer_text = story

            # Причесываем ответ
            if len(answer_text.split()) >= 25 and not is_toxic(answer_text):
                # Делаем первую букву заглавной
                answer_text = answer_text[0].upper() + answer_text[1:]
                clean_dialogues.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question_text},
                        {"role": "assistant", "content": answer_text},
                    ]
                })

    print(f"[+] Собрано {len(clean_dialogues)} чистейших историй из интервью без цензурных триггеров.")

    # Шаг 3. Склеиваем с TikTok датасетом
    final_dataset = []

    # Читаем тиктоки
    if os.path.exists(TIKTOK_DATASET):
        with jsonlines.open(TIKTOK_DATASET) as reader:
            all_bak = list(reader)
            # Берем СТРОГО первые 803 оригинальных тиктока, отрезая старую кривую Вписку
            tiktok_clean = all_bak[:803]
            final_dataset.extend(tiktok_clean)
        print(f"[*] Чистых TikTok диалогов добавлено: {len(final_dataset)}")

    final_dataset.extend(clean_dialogues)

    with jsonlines.open(OUTPUT_DATASET, mode="w") as writer:
        writer.write_all(final_dataset)

    print("\n" + "=" * 60)
    print(f"[+] ИТОГОВЫЙ ДАТАСЕТ ГОТОВ: {OUTPUT_DATASET}")
    print(f"[+] ВСЕГО ДИАЛОГОВ: {len(final_dataset)}")
    print("=" * 60)


if __name__ == "__main__":
    build_smart_dataset()