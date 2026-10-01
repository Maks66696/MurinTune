"""
src/03_build_v1_dataset.py — Сборка масштабного датасета v1.0 из всех 1026 фраз:
полная модерация чернухи, удаление мусора Whisper и умная смысловая привязка вопросов.
"""
import json
import os
import re
import hashlib
import jsonlines
import config

INPUT_FILE = "data/all_phrases.json"
OUTPUT_DIR = config.DATASET_DIR

SYSTEM_PROMPT = (
    "Ты — популярный стример Меллстрой. Веди диалог как живой парень на стриме: "
    "используй свой сленг, шути, подкалывай зрителя, общайся на «ты». "
    "Не будь вежливым ботом-помощником. Отвечай коротко и с юмором. Без лишних матов."
)

# 1. СТОП-СЛОВА: Запрещенка, криминал и технический мусор Whisper
PROHIBITED_PATTERNS = [
    r"педофил",
    r"сожгу твою сестру",
    r"прах смешаю",
    r"изнасилу",
    r"dimatorzok",
    r"диматорзок",
    r"субтитры",
    r"продолжение следует",
    r"динамичная музыка",
    r"аплодисменты",
    r"редактор субтитров",
    r"регистрируйся.*ссылка",
    r"1xbet",
    r"1хбет",
]

# 2. УМНЫЙ ПОДБОРЩИК ВОПРОСОВ (по ключевым темам реплики Мелла)
THEMATIC_RULES = [
    # Закон, полиция, розыск, остров
    (["интерпол", "остров", "заперт", "розыск", "задержан", "полици", "мусор", "тюрьм", "уголовк", "зона", "стл"],
     ["Мелл, почему ты в Россию или Беларусь не летишь?", "Тебя реально Интерпол ищет?", "Что у тебя там с уголовными делами?"]),
    
    # Игры (Minecraft, Dota, Fortnite, Roblox, CS)
    (["майнкрафт", "minecraft", "дот", "доту", "fortnite", "фортнайт", "роблокс", "roblox", "кс", "сервер"],
     ["Мелл, го в Доту сыграем?", "Погнали в Майнкрафт стримить!", "Пойдешь в катку с подписчиками?"]),

    # Финансы, бабки, миллионы, крипта, сейфы
    (["бабк", "деньг", "миллион", "лям", "доллар", "богат", "нищ", "бабло", "крипт", "наличк", "сейф", "420"],
     ["Сколько сегодня поднял бабок?", "Как стать миллионером?", "Не боишься все деньги слить и остаться нищим?"]),

    # Вес, похудение, еда, спорт, фастфуд
    (["похуден", "жирн", "толст", "вес", "бургер", "биг тейсти", "мадак", "хавчик", "пив", "плов", "кальян", "спорт", "щеки"],
     ["Мелл, как там твой спор на похудение?", "Что вкусного сегодня заказывал?", "Посоветуй идеальный заказ в фастфуде."]),

    # Мурино, лор, бурмалда, друн, авокадо
    (["мурин", "молочн", "бурмалд", "друн", "авокадо", "бабан", "боровка", "панк"],
     ["Видел мемы про Мурино с твоим лицом?", "Что такое бурмалда, друн?", "Как дела в Мурино?"]),

    # Хейтеры, наезды, агрессия в чате
    (["урод", "пиздобол", "долбоеб", "свали", "пошел вон", "закрой ебало", "щенок", "неженк"],
     ["Слышь, ты че такой дерзкий в чате?", "Мелл, ты дурак?", "Задонатил 10 рублей, пошел нафиг."]),

    # Внешность, стрижка, барбер, шмотки
    (["прическ", "стрижк", "волос", "барбер", "куртк", "кожанк", "роликсы", "часы"],
     ["Оцени свою новую стрижку.", "Где такую куртку урвал?", "Как тебе твой стиль сейчас?"]),

    # Известные личности (Бист, Соловьев, Путин и т.д.)
    (["бист", "путин", "соловьев", "месси", "роналду", "володя"],
     ["Что думаешь про этих известных людей?", "Видел, что про тебя говорят в новостях?"]),
]

FALLBACK_PROMPTS = [
    "Мелл, здорово, как сам?",
    "Что интересного на стриме расскажешь?",
    "Какие мысли по жизни сегодня?",
    "Дай какой-нибудь житейский совет чату.",
    "Как настроение, братуха?",
    "Что скажешь чату на сегодня?",
]

def clean_text(text: str) -> str:
    """Удаляет мусорные зацикливания Whisper и чистит пунктуацию."""
    text = re.sub(r"\s+", " ", text).strip()
    # Срезаем повторы букв вроде 'Ойойойой' или 'Ааааааа'
    text = re.sub(r"(.)\1{6,}", r"\1\1\1", text)
    # Срезаем повторы слов (например, 'баба баба баба баба...')
    text = re.sub(r"\b(\w+)\b(?:\s+\1\b){4,}", r"\1 \1", text, flags=re.IGNORECASE)
    # Убираем обрывки 'О, тишина'
    text = re.sub(r"О,?\s*тишина\.?", "", text, flags=re.IGNORECASE).strip()
    return text

def is_prohibited_or_junk(text: str) -> bool:
    """Проверяет на криминал, запрещенку и технический мусор."""
    lower = text.lower()
    if len(text) < 12:  # Слишком короткие огрызки (вроде 'Нормально.', 'О-о-о')
        return True
    for pat in PROHIBITED_PATTERNS:
        if re.search(pat, lower):
            return True
    return False

def get_matching_prompt(assistant_text: str) -> str:
    """Подбирает идеальный вопрос под тему реплики."""
    lower = assistant_text.lower()
    for keywords, prompts in THEMATIC_RULES:
        if any(kw in lower for kw in keywords):
            h = int(hashlib.md5(assistant_text.encode()).hexdigest(), 16)
            return prompts[h % len(prompts)]
    
    h = int(hashlib.md5(assistant_text.encode()).hexdigest(), 16)
    return FALLBACK_PROMPTS[h % len(FALLBACK_PROMPTS)]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[-] Файл {INPUT_FILE} не найден! Сначала запусти src/dump_to_json.py")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"[*] Загружено {len(raw_items)} исходных реплик из {INPUT_FILE}...")
    
    valid_dialogues = []
    seen = set()
    filtered_junk = 0

    for item in raw_items:
        text = clean_text(item.get("text", ""))
        
        if is_prohibited_or_junk(text):
            filtered_junk += 1
            continue

        # Дедупликация похожих фраз
        norm_key = re.sub(r"[^\w\s]", "", text.lower())[:40]
        if norm_key in seen:
            continue
        seen.add(norm_key)

        user_prompt = get_matching_prompt(text)
        valid_dialogues.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text}
            ]
        })

    # Сплит 88% train / 12% test
    split_idx = int(len(valid_dialogues) * 0.88)
    train_data = valid_dialogues[:split_idx]
    test_data = valid_dialogues[split_idx:]

    train_path = os.path.join(OUTPUT_DIR, "train_raw.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_raw.jsonl")

    with jsonlines.open(train_path, mode="w") as writer:
        writer.write_all(train_data)

    with jsonlines.open(test_path, mode="w") as writer:
        writer.write_all(test_data)

    print("\n" + "=" * 60)
    print(f"[+] ВСЕГО ИСХОДНЫХ ФРАЗ:      {len(raw_items)}")
    print(f"[-] ОТСЕЯНО МУСОРА И ЗАПРЕТА: {filtered_junk}")
    print(f"[+] В ТРЕНИРОВКУ (train_raw): {len(train_data)} полноценных диалогов!")
    print(f"[+] В ТЕСТ (test_raw):        {len(test_data)} диалогов!")
    print("=" * 60)
    print("Датасет v1.0 полностью готов к масштабному обучению!")

if __name__ == "__main__":
    main()