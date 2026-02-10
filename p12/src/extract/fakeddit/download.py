import os
from pathlib import Path

FAKEDDIT_V2_FOLDER_ID = "1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm"

def download_fakeddit(out_dir: Path) -> bool:
    """
    Télécharge Fakeddit v2.0 dans out_dir.
    Retourne True si succès, False sinon.
    """
    try:
        import gdown  # type: ignore
    except Exception:
        print("[fakeddit][ERROR] gdown not available, cannot download dataset")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fakeddit] Downloading dataset into {out_dir}")

    try:
        result = gdown.download_folder(
            id=FAKEDDIT_V2_FOLDER_ID,
            output=str(out_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        print(f"[fakeddit][ERROR] Download failed: {e}")
        return False

    # gdown renvoie None ou liste vide en cas d’échec
    if not result:
        print("[fakeddit][ERROR] Download failed (no files retrieved)")
        return False

    print("[fakeddit] Download completed")
    return True
