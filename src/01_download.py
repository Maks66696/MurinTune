import os
import subprocess
import sys

OUTPUT_DIR = os.path.join("data", "raw_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_audio(url):
    print(f"[*] Скачиваем аудио из: {url}")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--no-playlist",  # Убери этот флаг, если вставляешь ссылку на целый плейлист
        "-o",
        f"{OUTPUT_DIR}/%(id)s.%(ext)s",
        url,
    ]

    try:
        subprocess.run(cmd, check = True)
        print("[+] Аудио успешно скачано в папку data/raw_audio!")
    except subprocess.CalledProcessError as e:
        print(f"[-] Ошибка скачивания: {e}")

if __name__ == "__main__":
    if len(sys.argv) >1:
        target_url = sys.argv[1]
    else:
        target_url = input("Введи ссылку на видео или нарезку: ").strip()

    if target_url:
        download_audio(target_url)
    else:
        print("Ссылка не была указана.")