"""
config.py — единая точка правды для путей и параметров пайплайна MurinTune.

Импортируется всеми скриптами в src/ (запуск через `python src/0X_*.py`
автоматически добавляет src/ в sys.path, поэтому `import config` работает
без дополнительной настройки).
"""
from pathlib import Path

# --- Пути проекта ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_raw"
DATASET_DIR = DATA_DIR / "dataset"
ADAPTERS_DIR = ROOT_DIR / "adapters"

ARCHIVE_FILE = DATA_DIR / "downloaded_archive.txt"
FILTERS_FILE = DATA_DIR / "filters.json"
FILTERS_EXAMPLE_FILE = DATA_DIR / "filters.example.json"

for _d in (RAW_AUDIO_DIR, TRANSCRIPTS_DIR, DATASET_DIR, ADAPTERS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Датасет ---------------------------------------------------------------
# "white"  — очищенная версия (без мата/треша, только нейтральный сленг)
# "raw"    — оригинал, без сглаживания
MODES = ("white", "raw")

SYSTEM_PROMPTS = {
    "white": "Ты — виртуальный собеседник, общающийся на интернет-жаргоне молодежных сообществ.",
    "raw": "Ты — персонаж интернет-культуры. Отвечай эмоционально в своём фирменном стиле.",
}

# Склейка соседних сегментов транскрипта в реплики "разговорного" размера
MERGE_GAP_SECONDS = 1.5      # если пауза между сегментами больше — новая реплика
MIN_UTTERANCE_CHARS = 8      # короче — считаем шумом/междометием
MAX_UTTERANCE_CHARS = 280    # длиннее — режем на новую реплику

# Train/test сплит — делим по id исходного видео, а не по строкам,
# иначе почти одинаковые фразы из одного ролика утекут в обе выборки.
TEST_SIZE = 0.12
MIN_TEST_SAMPLES = 5
RANDOM_SEED = 42

# --- Обучение (Unsloth + QLoRA) --------------------------------------------
# Gemma 4 E2B. Официальная документация Unsloth прямо указывает VRAM по
# размерам: "Gemma 4 E2B trains on 8GB VRAM. E4B requires 10GB VRAM." —
# то есть E4B на вашей 3060 Ti (8 ГБ) рискует не влезть, а E2B — как раз
# заявленный минимум под вашу карту.
# https://huggingface.co/unsloth/gemma-4-E2B-it-unsloth-bnb-4bit
BASE_MODEL = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"
CHAT_TEMPLATE = "gemma-4"  # "gemma-4-thinking" — для крупных 26B/31B, не для E2B/E4B

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 5
