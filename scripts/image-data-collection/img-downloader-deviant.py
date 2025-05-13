import os
import requests
from bs4 import BeautifulSoup

# --- Config ---
html_path = r"webpages-deviant\1.html"
save_dir = r"ghibli\finetuning-data\deviant"
os.makedirs(save_dir, exist_ok=True)

# --- Load and parse HTML ---
with open(html_path, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

img_tags = soup.find_all('img')
image_urls = []

for tag in img_tags:
    src = tag.get('src')
    if src and src.startswith("http") and any(ext in src for ext in [".jpg", ".jpeg", ".png"]):
        image_urls.append(src)

print(f"[INFO] Found {len(image_urls)} image URLs.")

# --- Download images ---
for idx, url in enumerate(image_urls):
    try:
        filename = os.path.basename(url.split("?")[0])
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            print(f"[SKIP] {filename} already exists.")
            continue

        print(f"[DOWNLOADING] {url}")
        img_data = requests.get(url, timeout=10).content
        with open(filepath, 'wb') as f:
            f.write(img_data)
        print(f"[SAVED] {filename}")
    except Exception as e:
        print(f"[ERROR] Could not download {url}: {e}")
