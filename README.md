# 🏙️ MurinTune

<p align="center">
  <a href="https://huggingface.co/mxbtv/MurinTune-E2B-GGUF">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-MurinTune--E2B--GGUF-yellow?style=for-the-badge" alt="Hugging Face">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Unsloth-QLoRA%20(4--bit)-brightgreen?style=for-the-badge" alt="Unsloth">
</p>

> 📦 **Готовая GGUF-модель для LM Studio / Ollama:** [huggingface.co/mxbtv/MurinTune-E2B-GGUF](https://huggingface.co/mxbtv/MurinTune-E2B-GGUF)

**MurinTune** — сквозной программный конвейер для автоматического сбора речевых данных, транскрибации, семантической модерации и параметрически эффективного дообучения (**QLoRA**) больших языковых моделей на базе `Gemma-4-E2B-it`. Модель адаптирована для воспроизведения эмоционально-экспрессивного разговорного дискурса, неологизмов и фольклора современного молодежного интернет-сообщества.

Проект разработан в рамках индивидуального исследовательского проекта:  
> **«Исследование методов эффективного дообучения больших языковых моделей для адаптации под индивидуальный речевой стиль»**

---

## 🏗️ Архитектура пайплайна

```text
[01_download.py]       -> Скачивание аудио (yt-dlp) в data/raw_audio/*.mp3
[02_transcribe.py]     -> ASR-распознавание (faster-whisper Turbo на CUDA) в data/transcripts_raw/*.json
[dump_json.py]         -> Агрегация всех транскриптов в data/all_phrases.json
[03_build_v1_dataset]  -> Фильтрация шума/запрещенки + семантический подбор вопросов -> train/test.jsonl
[04_train.py]          -> 4-bit QLoRA дообучение через Unsloth (RTX 3060 Ti 8GB) -> adapters/raw/
[export_gguf.py]       -> Слияние весов и квантование в GGUF (Q4_K_M) -> models/
[chat.py]              -> Интерактивный диалоговый чат с моделью в терминале
```

---

## 🚀 Установка и настройка окружения (Windows / Linux)

### 1. Клонирование и виртуальное окружение
```bash
git clone https://github.com/Maks66696/MurinTune.git
cd MurinTune
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Установка зависимостей с поддержкой CUDA
```bash
# Установка PyTorch с поддержкой CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установка Triton для Windows и Unsloth
pip install triton-windows==3.2.0.post19
pip install unsloth unsloth_zoo
pip install -r requirements.txt
```
*(Для работы `01_download.py` требуется установленный в системе `ffmpeg`)*.

---

## 🛠️ Запуск полного цикла обучения

```bash
# 1. Скачивание аудиоматериалов
python src/01_download.py "ссылка_на_профиль_или_плейлист" --limit 1000

# 2. Нейросетевая транскрибация речи
python src/02_transcribe.py

# 3. Агрегация и семантическая модерация датасета (800+ диалогов)
python src/dump_json.py
python src/03_build_v1_dataset.py

# 4. QLoRA-дообучение на видеокарте (занимает ~1-2 часа на RTX 3060 Ti)
python src/04_train.py --mode raw --epochs 3

# 5. Экспорт в формат GGUF (Q4_K_M)
python src/export_gguf.py

# 6. Запуск живого диалога в консоли
python src/chat.py
```

---

## 📊 Характеристики модели и метрики (v0.5 Beta)

| Метрика | Значение |
| :--- | :--- |
| **Базовая модель** | `unsloth/gemma-4-E2B-it-bnb-4bit` (5.1 млрд параметров) |
| **Обучаемые параметры** | 24,158,208 (0.47% от общего объема весов) |
| **Обучающий корпус** | 803 диалоговые пары (вычищены из 1026 видео) |
| **Тестовая выборка** | 110 диалоговых пар |
| **Финальная функция потерь (Loss)** | **1.725** (средний `train_loss: 2.865`) |
| **Аппаратные ресурсы** | NVIDIA GeForce RTX 3060 Ti (8.0 GB VRAM, пик 7.7 GB) |
| **Время обучения** | 2 часа 15 минут (303 шага) |

---

## ⚠️ ВНИМАНИЕ / DISCLAIMER (18+)
* Модель является **юмористической сатирой и художественной AI-пародией** (*parody / satire*).
* Содержит обилие **ненормативной лексики**, экспрессивного сленга и интернет-абсурда.
* Проект создан исключительно в образовательных и научно-исследовательских целях в сфере компьютерной лингвистики (NLP).
* Автор не разделяет и не пропагандирует высказывания, генерируемые нейросетью.

---

## 📚 Цитирование (BibTeX)

```bibtex
@misc{murintune2026,
  author = {Maks66696},
  title = {MurinTune: Pipeline for Efficient Speech Style Adaptation of LLMs},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Maks66696/MurinTune}}
}
```