"""
src/dump_to_json.py — Сборка всех сырых реплик из 1027 файлов в один красивый JSON.
"""
import glob
import json
import os

INPUT_DIR = "data/transcripts_raw"
OUTPUT_FILE = "data/all_phrases.json"

phrases = []
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

print(f"[*] Обрабатываю {len(files)} файлов...")

for f_path in files:
    try:
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Берем текст или чанки
            text = data.get("text", "").strip()
            if text and len(text) >= 10:
                phrases.append({
                    "id": data.get("id", os.path.splitext(os.path.basename(f_path))[0]),
                    "text": text
                })
    except Exception:
        continue

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(phrases, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print(f"[+] ГОТОВО! Собрано {len(phrases)} фраз в: {OUTPUT_FILE}")
print("=" * 50)