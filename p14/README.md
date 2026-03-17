# AI Engineer - Finetunez votre propre LLM
## Installation du projet

### Pré-requis : Docker

Le projet se base sur une architecture containerisée avec docker.
`docker` doit être installé sur la machine hôte.

### Initialiser le projet : Make

`make` doit être installé sur votre hôte.
Lancez la commande suivante pour initialiser le projet : 

```
make install
```

Cela va télécharger les datasets si cela n'est pas déjà fait, dans le dossier suivant : `./data/dataset/raw`.

## Datasets

* https://huggingface.co/datasets/ANR-MALADES/MediQAl
* https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference
* https://huggingface.co/datasets/nthngdy/frenchmedmcqa
* https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset