from .model import CLASS_MAPPING, COLOR_MAPPING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

def display_image(image, ax=None):
    """Affiche uniquement l'image d'origine"""
    if ax is None:
        plt.figure(figsize=(6, 8))
        ax = plt.gca()

    ax.imshow(image)
    ax.set_title("Image d'origine")
    ax.axis('off')

    if ax is plt.gca():  # Si c'est une figure standalone
        plt.tight_layout()
        plt.show()


def display_masks(image, masks, maskOnly=False, ax=None):
    """Affiche uniquement les masques superposés sur l'image"""
    if ax is None:
        plt.figure(figsize=(6, 8))
        ax = plt.gca()

    if not maskOnly:
        ax.imshow(image)

    # Créer une image colorée pour les masques
    colored_masks = np.zeros((*masks.shape, 4))
    for class_id in np.unique(masks):
        if class_id != 0:  # Ignorer le background
            mask_positions = masks == class_id
            colored_masks[mask_positions] = COLOR_MAPPING[class_id]

    ax.imshow(colored_masks, alpha=0.5)
    ax.set_title("Masques superposés")
    ax.axis('off')

    if ax is plt.gca():  # Si c'est une figure standalone
        plt.tight_layout()
        plt.show()


def display_legend(masks, ax=None):
    """Affiche uniquement la légende des classes présentes"""
    ID_TO_LABEL = {v: k for k, v in CLASS_MAPPING.items()}
    present_ids = [i for i in np.unique(masks) if i != 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    ax.axis('off')

    patches = []
    for class_id in present_ids:
        color = COLOR_MAPPING[class_id]
        label = ID_TO_LABEL.get(class_id, str(class_id))
        patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=patches, loc='center', frameon=False)

    if ax is plt.gca():  # Si c'est une figure standalone
        plt.tight_layout()
        plt.show()


def display_images_grid(image_paths, masks_data, max):
    """
    Affiche les images, les masques seuls, et les masques superposés dans une grille avec légende globale.
    """
    images_data = []
    for path in image_paths[:max]:
        img = Image.open(path)
        img_array = np.array(img)
        images_data.append(img_array)

    n_images = len(images_data)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    ID_TO_LABEL = {v: k for k, v in CLASS_MAPPING.items()}
    all_present_ids = set()

    for i, (image, masks) in enumerate(zip(images_data, masks_data)):
        # Image originale
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        # Masques seuls (sans background)
        colored_masks = np.zeros((*masks.shape, 4))
        present_ids = [id for id in np.unique(masks) if id != 0]
        all_present_ids.update(present_ids)
        for class_id in present_ids:
            mask_positions = masks == class_id
            colored_masks[mask_positions] = COLOR_MAPPING[class_id]
        axes[i, 1].imshow(colored_masks)
        axes[i, 1].set_title(f"Masques seuls {i + 1}")
        axes[i, 1].axis('off')

        # Masques superposés sur l'image
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(colored_masks, alpha=0.5)
        axes[i, 2].set_title(f"Masques superposés {i + 1}")
        axes[i, 2].axis('off')

        # Colonne pour la légende (seulement première ligne)
        axes[i, 3].axis('off')

    # Légende globale dans la première ligne, quatrième colonne
    if all_present_ids:
        patches = []
        for class_id in sorted(all_present_ids):
            color = COLOR_MAPPING[class_id]
            label = ID_TO_LABEL.get(class_id, str(class_id))
            patches.append(mpatches.Patch(color=color, label=label))
        axes[0, 3].legend(handles=patches, loc='center', frameon=False)
        axes[0, 3].set_title("Légende")

    plt.tight_layout()
    plt.show()