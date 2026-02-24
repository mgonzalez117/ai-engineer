import os
from backend.rag.wikichess_scrap import main as scrape_main
from backend.rag.embeddings import main as embeddings_main


DATA_DIR = os.getenv("DATA_DIR", ".data")
DATA_FILE = os.path.join(DATA_DIR, "processed", "wikichess_chunks.jsonl")


def main(force_scrape: bool = False):
    """
    force_scrape = True -> rescrape même si fichier existe
    """

    if os.path.exists(DATA_FILE) and not force_scrape:
        print("Fichier chunks déjà présent -> skip scraping")
    else:
        print("===== ETAPE 1 : SCRAP WIKICHESS =====")
        scrape_main()

    print("\n===== ETAPE 2 : EMBEDDING + INDEXATION =====")
    embeddings_main()

    print("\nPipeline terminé avec succès.")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    main(force_scrape=force)