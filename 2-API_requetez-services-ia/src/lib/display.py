from .model import CLASS_MAPPING, COLOR_MAPPING, create_colored_masks_with_legend
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_images_grid(image_paths, masks_data, max=0):
    """
    Affiche les images, les masques seuls avec légende, et les masques superposés dans une grille de 3 colonnes.
    Version mise à jour utilisant la fonction extraite.
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

    for i, (image, masks) in enumerate(zip(images_data, masks_data)):
        # Image originale
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")
        axes[i, 0].axis('off')

        # Masques seuls avec légende (utilisation de la fonction extraite)
        colored_masks, patches = create_colored_masks_with_legend(masks, axes[i, 1], show_legend=True)
        axes[i, 1].imshow(colored_masks)
        axes[i, 1].set_title(f"Masques seuls {i + 1}")
        axes[i, 1].axis('off')

        # Masques superposés sur l'image
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(colored_masks, alpha=0.5)
        axes[i, 2].set_title(f"Masques superposés {i + 1}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()