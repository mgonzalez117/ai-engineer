SHELL := /bin/bash

RAW_DIR := ./.data/dataset/raw

MEDQUAD_DIR := $(RAW_DIR)/MedQuad-MedicalQnADataset
MEDIQAL_DIR := $(RAW_DIR)/MediQAl
ULTRAMEDICAL_DIR := $(RAW_DIR)/UltraMedical-Preference
FRENCHMEDMCQA_DIR := $(RAW_DIR)/frenchmedmcqa

MEDQUAD_REPO := https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset
MEDIQAL_REPO := https://huggingface.co/datasets/ANR-MALADES/MediQAl
ULTRAMEDICAL_REPO := https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference
FRENCHMEDMCQA_REPO := https://huggingface.co/datasets/nthngdy/frenchmedmcqa

.DEFAULT_GOAL := install

.PHONY: install all datasets dirs docker-up docker-down clean check-tools

install: check-tools datasets docker-up

all: install

check-tools:
	@command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "docker is required"; exit 1; }
	@docker compose version >/dev/null 2>&1 || { echo "docker compose is required"; exit 1; }

datasets: \
	$(MEDQUAD_DIR)/.done \
	$(MEDIQAL_DIR)/.done \
	$(ULTRAMEDICAL_DIR)/.done \
	$(FRENCHMEDMCQA_DIR)/.done

dirs:
	mkdir -p $(RAW_DIR)

$(MEDQUAD_DIR)/.done: | dirs
	@if [ ! -d "$(MEDQUAD_DIR)/.git" ]; then \
		echo ">>> Cloning keivalya/MedQuad-MedicalQnADataset"; \
		rm -rf "$(MEDQUAD_DIR)"; \
		git clone "$(MEDQUAD_REPO)" "$(MEDQUAD_DIR)"; \
	else \
		echo ">>> MedQuad already present, skipping"; \
	fi
	@touch $@

$(MEDIQAL_DIR)/.done: | dirs
	@if [ ! -d "$(MEDIQAL_DIR)/.git" ]; then \
		echo ">>> Cloning ANR-MALADES/MediQAl"; \
		rm -rf "$(MEDIQAL_DIR)"; \
		git clone "$(MEDIQAL_REPO)" "$(MEDIQAL_DIR)"; \
	else \
		echo ">>> MediQAl already present, skipping"; \
	fi
	@touch $@

$(ULTRAMEDICAL_DIR)/.done: | dirs
	@if [ ! -d "$(ULTRAMEDICAL_DIR)/.git" ]; then \
		echo ">>> Cloning TsinghuaC3I/UltraMedical-Preference"; \
		rm -rf "$(ULTRAMEDICAL_DIR)"; \
		git clone "$(ULTRAMEDICAL_REPO)" "$(ULTRAMEDICAL_DIR)"; \
	else \
		echo ">>> UltraMedical-Preference already present, skipping"; \
	fi
	@touch $@

$(FRENCHMEDMCQA_DIR)/.done: | dirs
	@if [ ! -d "$(FRENCHMEDMCQA_DIR)/.git" ]; then \
		echo ">>> Cloning nthngdy/frenchmedmcqa"; \
		rm -rf "$(FRENCHMEDMCQA_DIR)"; \
		git clone "$(FRENCHMEDMCQA_REPO)" "$(FRENCHMEDMCQA_DIR)"; \
	else \
		echo ">>> frenchmedmcqa already present, skipping"; \
	fi
	@touch $@

docker-up:
	@echo ">>> Starting Docker stack"
	docker compose up -d

docker-down:
	@echo ">>> Stopping Docker stack"
	docker compose down

clean:
	rm -rf $(RAW_DIR)