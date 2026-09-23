"""
01_download.py — Батч-загрузка аудио из плейлиста/профиля через yt-dlp.

Без изменений в логике по сравнению с исходной версией — только пути
теперь берутся из config.py, чтобы не расходиться с остальными скриптами.
"""
from __future__ import annotations

import argparse
import subprocess

import config

OUTPUT_DIR = config.RAW_AUDIO_DIR
ARCHIVE_FILE = config.ARCHIVE_FILE


def download_batch(url: str, max_downloads: int | None = None) -> None:
    print(f"[*] Старт загрузки из источника: {url}")
    print(f"[*] Аудио сохраняется в: {OUTPUT_DIR}")
    print(f"[*] Журнал скачанного ведётся в: {ARCHIVE_FILE}\n")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--download-archive", str(ARCHIVE_FILE),
        "--ignore-errors",
        "--no-abort-on-error",
        "--retries", "5",
        "--fragment-retries", "5",
        "-o", f"{OUTPUT_DIR}/%(id)s.%(ext)s",
        url,
    ]

    if max_downloads:
        cmd.extend(["--max-downloads", str(max_downloads)])

    try:
        subprocess.run(cmd, check=True)
        print("\n[+] Загрузка успешно завершена!")
    except subprocess.CalledProcessError as e:
        print(f"\n[-] Процесс завершился с кодом: {e}")
    except KeyboardInterrupt:
        print("\n[!] Скачивание прервано пользователем. Прогресс сохранён!")
    except FileNotFoundError:
        print(
            "\n[-] yt-dlp не найден. Установите его: pip install -r requirements.txt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MurinTune Batch Downloader")
    parser.add_argument(
        "url", nargs="?", help="Ссылка на профиль, плейлист или коллекцию"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество скачиваемых видео (например, --limit 50)",
    )
    args = parser.parse_args()

    target_url = args.url or input(
        "Введи ссылку на плейлист/профиль с футажами: "
    ).strip()

    if target_url:
        download_batch(target_url, max_downloads=args.limit)
    else:
        print("[-] Ссылка не была указана.")
