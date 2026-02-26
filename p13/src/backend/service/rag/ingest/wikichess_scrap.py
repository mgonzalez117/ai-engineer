import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.getenv("DATA_DIR", ".data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

BASE_URL = "https://ficgs.com/"
INDEX_URL = "https://ficgs.com/wikichess.html"

# Scraping : éviter un BAN du site
MAX_PAGES = 15
REQUEST_DELAY = 1.0

HEADERS = {
    "User-Agent": "AI-Engineer-OpenClassrooms-Project13 (educational purpose)"
}

WIKICHESS_MARKERS = ("[ECO", "[Opening", "[Variation", "Contributors", "============")
FOOTER_MARKER = "See this chess line with the javascript viewer"


def safe_get(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"Erreur sur {url}: {e}")
        return None


def discover_wikichess_pages() -> list[str]:
    html = safe_get(INDEX_URL)
    if not html:
        raise RuntimeError("Impossible de charger l'index Wikichess")

    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "wikichess_" in href and href.endswith(".html"):
            links.add(urljoin(BASE_URL, href))

    return sorted(links)[:MAX_PAGES]


def _td_width(td) -> int | None:
    w = td.get("width")
    if not w:
        return None
    try:
        return int(re.sub(r"\D+", "", str(w)))
    except ValueError:
        return None


def _find_layout_content_td(soup: BeautifulSoup):
    """
    Trouve le <td> principal de contenu.
    Heuristique "vieux site": table layout, première ligne avec >= 5 colonnes,
    et le 3e td a une largeur ~ 400..900 (souvent 550).
    """
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if not first_row:
            continue

        # old HTML : parfois les <td> sont imbriqués
        tds = first_row.find_all("td", recursive=False)
        if not tds:
            tds = first_row.find_all("td")

        if len(tds) < 5:
            continue

        w3 = _td_width(tds[2])
        if w3 is None or not (400 <= w3 <= 900):
            continue

        return tds[2]

    return None


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Retire scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content_td = _find_layout_content_td(soup)
    node = content_td if content_td is not None else (soup.body or soup)

    text = node.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text).strip()

    # 1) Coupe le haut : le vrai contenu commence après "Position after :" (souvent)
    m = re.search(r"Position after\s*:\s*", text)
    if m:
        text = text[m.end():].strip()

    # 2) Enlève le bloc "..., last edited by ... [YYYY Month D]" si présent
    #    (on vire tout jusqu'à la fin du [date] uniquement)
    text = re.sub(
        r"^.*?\[\d{4}\s+[A-Za-z]+\s+\d{1,2}\]\s*",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    # 3) Coupe le footer (marker fiable)
    idx = text.find(FOOTER_MARKER)
    if idx != -1:
        text = text[:idx].strip()

    # Nettoyage final : paragraphes lisibles
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    return text


def chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    chunks: list[str] = []
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

    all_chunks: list[dict] = []

    for url in urls:
        print(f"Scraping {url}")
        html = safe_get(url)
        if not html:
            time.sleep(REQUEST_DELAY)
            continue

        text = extract_article_text(html)

        if len(text) < 200 and not any(m in text for m in WIKICHESS_MARKERS):
            print("  -> contenu trop court après extraction (et pas de markers), skip")
            time.sleep(REQUEST_DELAY)
            continue

        parts = chunk(text)

        for i, part in enumerate(parts):
            all_chunks.append(
                {
                    "id": f"{hash(url)}_{i}",
                    "url": url,
                    "text": part,
                }
            )

        time.sleep(REQUEST_DELAY)

    out_path = os.path.join(PROCESSED_DIR, "wikichess_chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(all_chunks)} chunks écrits dans {out_path}")


if __name__ == "__main__":
    main()