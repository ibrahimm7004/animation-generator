import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
CHROMEDRIVER_PATH = r"chromedriver-win64\chromedriver.exe"
TARGET_URL = "https://www.artstation.com/search?sort_by=relevance&query=studio%20ghibli&=&category_ids_include=37,13,30,67"
SAVE_DIR = r"ghibli\finetuning-data\art-station\2"
SCROLL_PAUSE_TIME = 2
MAX_SCROLLS = 15

os.makedirs(SAVE_DIR, exist_ok=True)

options = webdriver.ChromeOptions()
# options.add_argument("--headless")
options.add_argument("--log-level=3")
driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)

driver.get(TARGET_URL)
last_height = driver.execute_script("return document.body.scrollHeight")
scrolls = 0

while scrolls < MAX_SCROLLS:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
    scrolls += 1

print("[INFO] Finished scrolling. Extracting images...")

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

img_tags = soup.find_all("img")
img_urls = []

for tag in img_tags:
    src = tag.get("src", "")
    if "artstation.com/p/assets/images" in src or "artstation.com/p/assets/covers/images" in src:
        img_urls.append(src)

img_urls = list(set(img_urls))  # Deduplicate
print(f"[INFO] Found {len(img_urls)} image URLs.")

MAX_WORKERS = 12


def download_image(url):
    try:
        filename = os.path.basename(url.split("?")[0])
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            return f"[SKIP] {filename}"

        img_data = requests.get(url, timeout=10).content
        with open(filepath, "wb") as f:
            f.write(img_data)
        return f"[SAVED] {filename}"
    except Exception as e:
        return f"[ERROR] {url} — {e}"


print(
    f"[INFO] Downloading {len(img_urls)} images with {MAX_WORKERS} threads...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_image, url) for url in img_urls]
    for future in as_completed(futures):
        print(future.result())
