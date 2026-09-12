import argparse
import os
import subprocess
import sys

OUTPUT_DIR = os.path.join("data", "raw_audio")
ARCHIVE_FILE = os.path.join("data", "downloaded_archive.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_batch(url, max_downloads=None):
    print(f"[*] Старт загрузки из источника: {url}")
    print(f"[*] Аудио сохраняется в: {OUTPUT_DIR}")
    print(f"[*] Журнал скачанного ведется в: {ARCHIVE_FILE}\n")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--download-archive",
        ARCHIVE_FILE,
        "--ignore-errors",
        "--no-abort-on-error",
        "--retries",
        "5",
        "--fragment-retries",
        "5",

        "-o",
        f"{OUTPUT_DIR}/%(id)s.%(ext)s",
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
        print("\n[!] Скачивание прервано пользователем. Прогресс сохранен!")


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

    target_url = args.url
    if not target_url:
        target_url = input(
            "Введи ссылку на плейлист/профиль с футажами: "
        ).strip()

    if target_url:
        download_batch(target_url, max_downloads=args.limit)
    else:
        print("[-] Ссылка не была указана.")