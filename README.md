# 🏙️ MurinTune

<p align="center">
  <a href="https://huggingface.co/mxbtv/MurinTune-E2B-GGUF">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-MurinTune--E2B--GGUF-yellow?style=for-the-badge" alt="Hugging Face">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Unsloth-QLoRA%20(4--bit)-brightgreen?style=for-the-badge" alt="Unsloth">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

> 📦 **Официальный релиз квантованной GGUF-модели на Hugging Face:** [huggingface.co/mxbtv/MurinTune-E2B-GGUF](https://huggingface.co/mxbtv/MurinTune-E2B-GGUF)

**MurinTune** — сквозной программный комплекс для автоматизированного сбора речевых мультимедиа данных, нейросетевой транскрибации, семантической модерации корпуса и параметрически эффективного дообучения (**QLoRA / PEFT**) открытых языковых моделей на базе архитектуры `Gemma-4-E2B-it`. Модель адаптирована для воспроизведения эмоционально-экспрессивного разговорного дискурса, неологизмов и фольклора современного молодежного интернет-сообщества.

Проект разработан в рамках школьного индивидуального исследовательского проекта:  
> **«Исследование методов эффективного дообучения больших языковых моделей для адаптации под индивидуальный речевой стиль»**

---

## 📁 Структура репозитория

```text
MurinTune/
├── adapters/                     # Сохраненные веса обученных LoRA-адаптеров и чекпоинты (raw, white)
├── data/                         # Каталог данных и обучающих выборок
│   ├── dataset/                  # Сгенерированные диалоговые JSONL-датасеты
│   │   ├── .gitkeep              # Служебный файл контроля версий
│   │   ├── test_raw.jsonl        # Отложенная тестовая выборка (RAW, 110 диалогов)
│   │   ├── test.jsonl            # Общая тестовая выборка (v0.1)
│   │   ├── train_raw.jsonl       # Обучающая выборка (RAW, 803 диалога)
│   │   └── train_white.jsonl     # Обучающая выборка (WHITE, цензурированная)
│   ├── raw_audio/                # Каталог загруженных аудиодорожек (1026 файлов mp3)
│   ├── transcripts_raw/          # Посегментные JSON-транскрипты Whisper с таймкодами
│   ├── all_phrases.json          # Агрегированный массив всех извлеченных фраз
│   ├── downloaded_archive.txt    # Журнал загрузок yt-dlp (защита от повторного скачивания)
│   ├── filters.example.json      # Шаблон конфигурации стоп-слов и исключений
│   └── filters.json              # Локальный конфигуратор модерации и фильтрации
├── models/                       # Скомпилированные GGUF-модели для LM Studio / Ollama
├── notebooks/                    # Jupyter-ноутбуки для экспериментов и анализа
├── src/                          # Исходный код программного конвейера
│   ├── 01_download.py            # Пакетное скачивание аудиодорожек (yt-dlp)
│   ├── 02_transcribe.py          # ASR-распознавание речи (faster-whisper Turbo на CUDA)
│   ├── 03_build_dataset.py       # Базовая сборка датасета по правилам (v0.1 Alpha)
│   ├── 03_build_v1_dataset.py    # Продвинутая семантическая модерация корпуса (v0.5 Beta)
│   ├── 04_train.py               # 4-bit QLoRA-обучение через Unsloth (Gemma-4-E2B)
│   ├── 05_benchmark.py           # Автоматизированный бенчмарк и оценка на test-сете
│   ├── chat.py                   # Интерактивный терминальный чат с обученным персонажем
│   ├── config.py                 # Единая точка конфигурации путей и гиперпараметров
│   ├── dump_json.py              # Скрипт агрегации транскриптов в all_phrases.json
│   └── export_gguf.py            # Слияние весов LoRA и квантование в GGUF (Q4_K_M)
├── unsloth_compiled_cache/       # Бинарный кэш скомпилированных ядер Triton (в .gitignore)
├── venv/                         # Виртуальное окружение Python
├── .gitignore                    # Правила исключения временных и тяжелых файлов из Git
├── LICENSE                       # Лицензия открытого исходного кода (Apache 2.0)
├── README.md                     # Документация проекта
└── requirements.txt              # Список зависимостей проекта
```

---

## 🏗️ Архитектура сквозного конвейера (Pipeline)

1. **`01_download.py`** ➔ Пакетная загрузка аудиоматериалов (TikTok / YouTube) с ведением архива `downloaded_archive.txt`.
2. **`02_transcribe.py`** ➔ Высокоскоростная транскрибация речи на GPU с подавлением галлюцинаций Whisper (`condition_on_previous_text=False`).
3. **`dump_json.py`** ➔ Сборка всех посегментных JSON-файлов в единый реестр `all_phrases.json`.
4. **`03_build_v1_dataset.py`** ➔ Многоуровневая модерация: вычистка шума/запрещенного контента, алгоритмический подбор контекстных вопросов и формирование `train_raw.jsonl` / `test_raw.jsonl`.
5. **`04_train.py`** ➔ QLoRA-дообучение на 803 диалогах через Unsloth с оптимизацией VRAM под 8 ГБ (`device_map="cuda:0"`).
6. **`05_benchmark.py`** ➔ Прогон ответов на отложенном test-сете со штрафом за повторы (`repetition_penalty=1.18`).
7. **`export_gguf.py`** ➔ Слияние адаптера с базовой моделью и квантование в формат `Q4_K_M` для запуска в **LM Studio** и **Ollama**.
8. **`chat.py`** ➔ Интерактивная консольная среда для живого диалога.

---

## 🚀 Установка и запуск

### 1. Настройка окружения (Windows PowerShell)
```powershell
# Клонирование репозитория и создание venv
git clone https://github.com/Maks66696/MurinTune.git
cd MurinTune
python -m venv venv
.\venv\Scripts\Activate.ps1

# Установка PyTorch с поддержкой CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установка Triton для Windows и Unsloth
pip install triton-windows==3.2.0.post19
pip install unsloth unsloth_zoo
pip install -r requirements.txt
```

### 2. Запуск пайплайна
```powershell
# Сборка датасета из транскриптов
python src/dump_json.py
python src/03_build_v1_dataset.py

# Обучение модели v0.5 Beta (303 шага, ~1.5 часа на RTX 3060 Ti)
python src/04_train.py --mode raw --epochs 3

# Экспорт в формат GGUF для LM Studio
python src/export_gguf.py

# Запуск живого диалога в терминале
python src/chat.py
```

---

## 📊 Научно-технический профиль обучения (v0.5 Beta)

| Метрика | Значение | Описание |
| :--- | :--- | :--- |
| **Базовая модель** | `unsloth/gemma-4-E2B-it-bnb-4bit` | 5,128,455,712 параметров |
| **Обучаемые параметры (LoRA)** | **24,158,208** | **0.47%** от общего объема весов |
| **LoRA Rank ($r$) / Alpha ($\alpha$)** | `16` / `16` | Оптимальный ранг адаптации стиля |
| **Обучающий корпус (Train)** | **803 диалоговые пары** | Вычищены из 1026 видеоматериалов |
| **Тестовая выборка (Test)** | **110 диалоговых пар** | Отложенная выборка валидации |
| **Финальный Loss** | **1.725** (avg: `2.865`) | Глубокая сходимость без переобучения |
| **Аппаратное обеспечение** | NVIDIA GeForce RTX 3060 Ti | 8.0 GB VRAM (пик потребления 7.7 GB) |
| **Время обучения** | **2 часа 15 минут** | 303 градиентных шага (батч 8) |

---

## ⚠️ ВНИМАНИЕ / DISCLAIMER (18+)
* Модель является **юмористической сатирой и художественной AI-пародией** (*parody / satire*).
* Содержит обилие **ненормативной лексики**, экспрессивного сленга и интернет-абсурда.
* Создана исключительно в образовательных и научно-исследовательских целях в сфере компьютерной лингвистики (NLP).
* Автор проекта не разделяет и не пропагандирует высказывания, генерируемые нейросетью.

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