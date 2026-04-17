.PHONY: start restart

DATA_DIR = .data
ZIP_URL = https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip
ZIP_FILE = $(DATA_DIR)/data.zip
TARGET_FILE = $(DATA_DIR)/application_test.csv

start:
	# 0. Cloner le repo de modèles à la racine si non présent
	@if [ ! -d models ]; then \
		echo "Clonage du dépôt de modèles dans ./models..."; \
		git clone https://huggingface.co/MGonzalez117/home-credit models; \
	else \
		echo "✓ Dossier ./models déjà présent, clonage ignoré"; \
	fi

	@if [ ! -f .env ]; then cp .env.dist .env; fi

	# 1. Vérifier que unzip est installé, sinon l'installer
	@if ! command -v unzip >/dev/null 2>&1; then \
		echo "unzip introuvable, installation..."; \
		sudo apt-get update && sudo apt-get install -y unzip; \
	else \
		echo "✓ unzip déjà installé"; \
	fi

	# 2. Télécharger + décompresser les données si application_test.csv est absent
	@if [ ! -f $(TARGET_FILE) ]; then \
		echo "Fichier $(TARGET_FILE) introuvable, téléchargement des données..."; \
		mkdir -p $(DATA_DIR); \
		curl -L "$(ZIP_URL)" -o $(ZIP_FILE); \
		echo "Décompression des données..."; \
		unzip -o -q $(ZIP_FILE) -d $(DATA_DIR)/; \
		rm $(ZIP_FILE); \
		echo "Contenu de $(DATA_DIR) après unzip :"; \
		ls -R $(DATA_DIR); \
		if [ -f "$(TARGET_FILE)" ]; then \
			echo "✓ $(TARGET_FILE) trouvé à la racine de $(DATA_DIR)"; \
		elif ls $(DATA_DIR)/*application_test.csv >/dev/null 2>&1; then \
			SRC_FILE=$$(ls $(DATA_DIR)/*application_test.csv | head -n 1); \
			echo "Déplacement de $$SRC_FILE vers $(TARGET_FILE)"; \
			mv "$$SRC_FILE" "$(TARGET_FILE)"; \
			echo "✓ $(TARGET_FILE) prêt"; \
		else \
			echo "ERREUR : application_test.csv introuvable après décompression."; \
			exit 1; \
		fi; \
	else \
		echo "✓ $(TARGET_FILE) déjà présent, téléchargement ignoré"; \
	fi

	# 3. Lancer la stack + scripts + snakeviz
	@docker compose up -d api dashboard postgres pgadmin
	@echo "Attente du démarrage des services..."
	@sleep 5
	@echo "Exécution du script d'extraction..."
	@docker compose exec api python -m src.scripts.extract_single
	@echo "Exécution du script de profiling..."
	@docker compose exec api python -m src.scripts.profile.predict
	@echo "Lancement de snakeviz..."
	@docker compose up -d snakeviz
	@echo "✓ Stack complète opérationnelle !"

restart:
	@docker compose down
	@$(MAKE) start