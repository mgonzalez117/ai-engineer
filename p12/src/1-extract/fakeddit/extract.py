import csv
from pathlib import Path


def iter_tsv(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row:
                yield row


def pick(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return default

def is_exploitable_image_url(url: str) -> bool:
    url = url.lower()
    if not url.startswith("http"):
        return False
    if any(url.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        return True
    if any(host in url for host in ("i.redd.it", "preview.redd.it", "i.imgur.com")):
        return True
    return False



def extract_record(rec: dict):
    # --- id ---
    rid = pick(rec, "id", "post_id", "name")
    if not rid:
        return None, "missing_id"

    # --- title ---
    title = str(pick(rec, "title", default="")).strip()
    if not title:
        return None, "missing_title"

    # --- author (optional) ---
    author = str(pick(rec, "author", default="")).strip()

    # --- image (multimodal strict) ---
    image_url = pick(rec, "image_url")
    if not image_url:
        return None, "missing_image"

    if not is_exploitable_image_url(image_url):
        return None, "non_exploitable_image_url"

    # --- label binaire ---
    two_way = pick(rec, "2_way_label", "label")
    if two_way is None:
        return None, "missing_label"

    two_way = str(two_way).strip()
    if two_way not in ("0", "1"):
        return None, "invalid_label"

    fake = (two_way == "1")

    return {
        "source": "fakeddit",
        "id": str(rid),
        "title": title,
        "image_url": image_url,
        "fake": fake,
        "author": author # champ complémentaire optionnel
    }, None
