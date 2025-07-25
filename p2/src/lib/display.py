from .model import CLASS_MAPPING, COLOR_MAPPING
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def display_images_grid(image_paths, masks_data, max=0):
    """
    Affiche les images, les masques seuls avec légende, et les masques superposés dans une grille de 3 colonnes.
    """

    if max == 0:
        max = len(image_paths)

    images_data = []
    for path in image_paths[:max]:
        img = Image.open(path)
        img_array = np.array(img)
        images_data.append(img_array)

    n_images = len(images_data)
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    ID_TO_LABEL = {v: k for k, v in CLASS_MAPPING.items()}

    for i, (image, masks) in enumerate(zip(images_data, masks_data)):
        # Image originale
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        # Masques seuls avec légende intégrée
        colored_masks = np.zeros((*masks.shape, 4))
        present_ids = [id for id in np.unique(masks) if id != 0]

        for class_id in present_ids:
            mask_positions = masks == class_id
            colored_masks[mask_positions] = COLOR_MAPPING[class_id]

        axes[i, 1].imshow(colored_masks)
        axes[i, 1].set_title(f"Masques seuls {i + 1}")
        axes[i, 1].axis('off')

        # Ajouter la légende directement sur l'image des masques seuls
        if present_ids:
            patches = []
            for class_id in sorted(present_ids):
                color = COLOR_MAPPING[class_id]
                label = ID_TO_LABEL.get(class_id, str(class_id))
                patches.append(mpatches.Patch(color=color, label=label))
            axes[i, 1].legend(handles=patches, loc='lower right', frameon=True,
                              fancybox=True, shadow=True, fontsize='small')

        # Masques superposés sur l'image
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(colored_masks, alpha=0.5)
        axes[i, 2].set_title(f"Masques superposés {i + 1}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()