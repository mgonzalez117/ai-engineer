.PHONY: install clean docker up

# Variables
DATA_DIR = .data
ZIP_FILE = $(DATA_DIR)/mri_dataset.zip
ZIP_URL = https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/1047_Data+Scientist+ML/P7_DSML_1047/mri_dataset_brain_cancer_oc.zip
EXTRACT_MARKER = $(DATA_DIR)/.extracted

# Règle par défaut
install: $(EXTRACT_MARKER) docker

# Créer le dossier .data s'il n'existe pas
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

# Télécharger le fichier zip seulement s'il n'existe pas
$(ZIP_FILE): $(DATA_DIR)
	@if [ ! -f $(ZIP_FILE) ]; then \
		echo "Téléchargement du dataset..."; \
		wget -O $(ZIP_FILE) $(ZIP_URL) || curl -o $(ZIP_FILE) $(ZIP_URL); \
	else \
		echo "Le fichier $(ZIP_FILE) existe déjà, pas de téléchargement."; \
	fi

# Dézipper le contenu dans .data/ et supprimer le zip
$(EXTRACT_MARKER): $(ZIP_FILE)
	@echo "Extraction du dataset dans $(DATA_DIR)..."
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)
	@echo "Suppression du fichier zip..."
	rm -f $(ZIP_FILE)
	@touch $(EXTRACT_MARKER)

# Lancer Docker Compose
docker:
	@echo "Lancement de Docker Compose..."
	docker-compose up -d

# Alias pour docker
up: docker

# Nettoyer les fichiers téléchargés
clean:
	rm -rf $(DATA_DIR)