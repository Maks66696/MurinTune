import glob
import json
import os
import site
import sys


candidates = site.getsitepackages() if hasattr(site, "getsitepackages") else []
for sp in sys.path:
    if "site-packages" in sp and sp not in candidates:
        candidates.append(sp)

for p in candidates:
    for sub in [
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
        ("nvidia", "cuda_runtime", "bin"),
    ]:
        dll_dir = os.path.join(p, *sub)
        if os.path.exists(dll_dir):
            os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

from faster_whisper import WhisperModel
from tqdm import tqdm

AUDIO_DIR = os.path.join("data", "raw_audio")
OUTPUT_DIR = os.path.join("data", "transcripts_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(use_gpu=True):
    if use_gpu:
        print("[*] Пробуем запустить faster-whisper turbo на GPU (CUDA)...")
        return WhisperModel("turbo", device="cuda", compute_type="float16")
    else:
        print(
            "[*] Запускаем faster-whisper turbo на CPU (быстрый режим int8)..."
        )
        return WhisperModel("turbo", device="cpu", compute_type="int8")


def transcribe_file(model, audio_path):
    segments, info = model.transcribe(
        audio_path, language="ru", beam_size=1, vad_filter=True
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


def transcribe_all():
    audio_files = (
        glob.glob(f"{AUDIO_DIR}/*.mp3")
        + glob.glob(f"{AUDIO_DIR}/*.wav")
        + glob.glob(f"{AUDIO_DIR}/*.mp4")
    )

    if not audio_files:
        print(f"[-] В папке {AUDIO_DIR} нет аудиофайлов!")
        return

    # Пробуем инициализировать модель на GPU
    use_gpu = True
    try:
        model = load_model(use_gpu=True)
    except Exception as e:
        print(f"[!] Ошибка запуска GPU ({e}). Переходим на CPU.")
        use_gpu = False
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
                print("\n[!] Ошибка CUDA DLL в рантайме. Переключаюсь на CPU...")
                model = load_model(use_gpu=False)
                result_text, chunks, info = transcribe_file(model, audio_path)
            else:
                raise e

        # Сохраняем результат
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