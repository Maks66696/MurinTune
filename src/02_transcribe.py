"""
02_transcribe.py — Транскрибация аудио через faster-whisper с поддержкой CUDA на Windows.
"""
import glob
import json
import os
import sys

# --- Исправление для поиска CUDA / cuDNN DLL на Windows ---
if sys.platform == "win32":
    venv_base = sys.prefix
    nvidia_dirs = [
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(venv_base, "Lib", "site-packages", "torch", "lib"),
    ]
    for p in nvidia_dirs:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + ";" + os.environ["PATH"]
            except Exception:
                pass

from faster_whisper import WhisperModel
from tqdm import tqdm

import config

AUDIO_DIR = config.RAW_AUDIO_DIR
OUTPUT_DIR = config.TRANSCRIPTS_DIR


def load_model(use_gpu: bool = True) -> WhisperModel:
    if use_gpu:
        print("[*] Пробуем запустить faster-whisper turbo на GPU (CUDA)...")
        # float16 идеален для карт серии RTX 30xx
        return WhisperModel("turbo", device="cuda", compute_type="float16")
    print("[*] Запускаем faster-whisper turbo на CPU (быстрый режим int8)...")
    return WhisperModel("turbo", device="cpu", compute_type="int8")


def transcribe_file(model: WhisperModel, audio_path: str):
    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Меллстрой, стрим, бурмалда, друн, боровка, пацаны, чат.",
    )
    full_text = []
    chunks = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            full_text.append(text)
            chunks.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                }
            )
    return " ".join(full_text), chunks, info


def transcribe_all() -> None:
    audio_files = (
        glob.glob(f"{AUDIO_DIR}/*.mp3")
        + glob.glob(f"{AUDIO_DIR}/*.wav")
        + glob.glob(f"{AUDIO_DIR}/*.mp4")
    )

    if not audio_files:
        print(f"[-] В папке {AUDIO_DIR} нет аудиофайлов! Запусти сперва 01_download.py")
        return

    try:
        model = load_model(use_gpu=True)
    except Exception as e:
        print(f"\n[!] Ошибка запуска GPU: {e}\n[!] Переходим на CPU...")
        model = load_model(use_gpu=False)

    print(f"[*] Найдено файлов для расшифровки: {len(audio_files)}\n")

    for audio_path in tqdm(audio_files, desc="Транскрибация"):
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        out_json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        if os.path.exists(out_json_path):
            continue

        try:
            result_text, chunks, info = transcribe_file(model, audio_path)
        except RuntimeError as e:
            if "cublas" in str(e) or "cuda" in str(e).lower():
                print("\n[!] Ошибка CUDA в рантайме. Переключаюсь на CPU...")
                model = load_model(use_gpu=False)
                result_text, chunks, info = transcribe_file(model, audio_path)
            else:
                raise

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": file_id,
                    "language": info.language,
                    "duration": round(info.duration, 2),
                    "text": result_text,
                    "chunks": chunks,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n[+] Файл {file_id}:")
        print(f'>>> "{result_text}"\n')


if __name__ == "__main__":
    transcribe_all()