import requests
from bs4 import BeautifulSoup
import os
import time

BASE_URL = "https://www.gutenberg.org"
SEARCH_URL = f"{BASE_URL}/ebooks/search/?sort_order=random"
OUTPUT_DIR = "./Output"
TARGET_BOOK_COUNT = 500

def get_unique_book_urls(target_count):
    unique_urls = set()

    while len(unique_urls) < target_count:
        print(f"Collected {len(unique_urls)} / {target_count} book URLs...")
        response = requests.get(SEARCH_URL)
        soup = BeautifulSoup(response.content, "html.parser")
        book_links = soup.select("li.booklink a.link")

        for link in book_links:
            href = link.get("href")
            if href and href.startswith("/ebooks/"):
                full_url = BASE_URL + href
                unique_urls.add(full_url)

        time.sleep(1)

    return list(unique_urls)

def download_books(book_urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    downloaded_titles = []

    for i, url in enumerate(book_urls, 1):
        try:
            print(f"[{i}/{len(book_urls)}] Processing: {url}")
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            link = soup.find("a", string="Plain Text UTF-8")
            if not link:
                print("  ❌ No Plain Text UTF-8 link found.")
                continue

            download_url = BASE_URL + link["href"]

            # Get the title to use as filename
            title_tag = soup.find("h1", itemprop="name")
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                title = f"book_{i}"

            # Clean the title to make it a valid filename
            invalid_chars = r'<>:"/\|?*'
            for ch in invalid_chars:
                title = title.replace(ch, "")
            title = title.replace(" ", "_")
            filename = f"{title}.txt"

            # Download and save book
            book_response = requests.get(download_url)
            book_response.encoding = "utf-8"
            book_text = book_response.text

            file_path = os.path.join(output_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(book_text)

            downloaded_titles.append(title)
            print("  ✅ Downloaded.")

            time.sleep(0.5)
        except Exception as e:
            print(f"  ❌ Failed for {url}: {e}")

    # Save all downloaded titles to a file
    titles_path = os.path.join(output_dir, "downloaded_titles.txt")
    with open(titles_path, "w", encoding="utf-8") as f:
        for title in downloaded_titles:
            f.write(title + "\n")



# Main runner
if __name__ == "__main__":
    book_urls = get_unique_book_urls(TARGET_BOOK_COUNT)
    download_books(book_urls, OUTPUT_DIR)

