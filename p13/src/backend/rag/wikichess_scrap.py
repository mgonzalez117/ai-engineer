import json
import re
import time
from urllib.parse import urljoin
import os

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.getenv("DATA_DIR", ".data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

BASE_URL = "https://ficgs.com/"
INDEX_URL = "https://ficgs.com/wikichess.html"

# Scraping : éviter un BAN du site
MAX_PAGES = 15        # limite volontaire
REQUEST_DELAY = 1.0   # 1 seconde entre requêtes

HEADERS = {
    "User-Agent": "AI-Engineer-OpenClassrooms-Project13 (educational purpose)"
}


def safe_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"Erreur sur {url}: {e}")
        return None


def discover_wikichess_pages():
    html = safe_get(INDEX_URL)
    if not html:
        raise RuntimeError("Impossible de charger l'index Wikichess")

    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "wikichess_" in href and href.endswith(".html"):
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    links = sorted(list(links))
    return links[:MAX_PAGES]


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def chunk(text, size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Découverte des pages Wikichess...")
    urls = discover_wikichess_pages()
    print(f"{len(urls)} pages trouvées")

    all_chunks = []

    for url in urls:
        print(f"Scraping {url}")
        html = safe_get(url)

        if not html:
            continue

        text = clean_text(html)
        parts = chunk(text)

        for i, part in enumerate(parts):
            all_chunks.append({
                "id": f"{hash(url)}_{i}",
                "url": url,
                "text": part
            })

        time.sleep(REQUEST_DELAY)

    out_path = os.path.join(PROCESSED_DIR, "wikichess_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(all_chunks)} chunks écrits dans {out_path}")


if __name__ == "__main__":
    main()