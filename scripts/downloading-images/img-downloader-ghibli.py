import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
SAVE_DIR = r"C:\Users\hp\Desktop\sem8\genAI\proj\ghibli\finetuning-data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
MAX_WORKERS = 8  # You can increase this to 16 or 32 if you have good bandwidth and CPU


def fetch_page_image_links(page_url: str):
    try:
        response = requests.get(page_url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch URL: {page_url} - {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    gallery_div = soup.find("div", class_="row gallery")
    if not gallery_div:
        print("[WARNING] No 'gallery' div found on this page.")
        return []

    image_links = [urljoin(page_url, a["href"])
                   for a in gallery_div.find_all("a", href=True)]
    return image_links


def download_image(img_url):
    img_name = img_url.split("/")[-1]
    save_path = os.path.join(SAVE_DIR, img_name)

    if os.path.exists(save_path):
        return f"[SKIP] {img_name} already exists."

    try:
        img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
        with open(save_path, "wb") as f:
            f.write(img_data)
        return f"[DOWNLOADED] {img_name}"
    except Exception as e:
        return f"[ERROR] Failed to download {img_url} - {e}"


def download_images_from_ghibli_page(page_url: str):
    image_urls = fetch_page_image_links(page_url)
    if not image_urls:
        print("[INFO] No images found to download.")
        return

    print(
        f"[INFO] Starting download of {len(image_urls)} images with {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(
            download_image, url): url for url in image_urls}
        for future in as_completed(future_to_url):
            print(future.result())


page_url = "https://www.ghibli.jp/works/ged/"
download_images_from_ghibli_page(page_url)
