from tqdm import tqdm
import mimetypes
import requests
import numpy as np
import base64
from PIL import Image
import io
import time

API_URL = "https://api-inference.huggingface.co/models/sayeed99/segformer_b3_clothes"

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

# Mapping fixe des couleurs pour chaque classe
COLOR_MAPPING = {
    0: [0, 0, 0, 0],             # Background - transparent
    1: [0.8, 0.2, 0.2, 1],       # Hat - rouge
    2: [1.0, 1.0, 0.0, 1],       # Hair - jaune fluo
    3: [1.0, 0.0, 1.0, 1],       # Sunglasses - rose fluo
    4: [0.2, 0.6, 0.8, 1],       # Upper-clothes - bleu clair
    5: [0.8, 0.2, 0.8, 1],       # Skirt - magenta
    6: [0.2, 0.2, 0.8, 1],       # Pants - bleu foncé
    7: [0.8, 0.6, 0.8, 1],       # Dress - rose
    8: [0.4, 0.2, 0.0, 1],       # Belt - marron foncé
    9: [1.0, 0.6, 0.2, 1],       # Left-shoe - orange clair
    10: [1.0, 1.0, 0.0, 1],      # Right-shoe - jaune vif
    11: [0.6, 1.0, 0.6, 1],      # Face - vert clair
    12: [0.8, 0.6, 0.4, 1],      # Left-leg - beige
    13: [0.7, 0.5, 0.3, 1],      # Right-leg - beige foncé
    14: [0.9, 0.7, 0.5, 1],      # Left-arm - chair clair
    15: [0.8, 0.6, 0.4, 1],      # Right-arm - chair
    16: [0.6, 0.2, 0.6, 1],      # Bag - violet
    17: [0.2, 0.8, 0.2, 1]       # Scarf - vert
}

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def request_for_image(path, api_token):
    computed_image = None

    headers = {
        "Authorization": f"Bearer {api_token}"
        # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
    }

    try:
        with open(path, 'rb') as f:
            image_data = f.read()
            content_type = mimetypes.guess_type(path)[0]
            response = requests.post(
                API_URL,
                headers={**headers, 'Content-Type': content_type},
                data=image_data
            )

            print(f'Api Response HTTP Code: {response.status_code}')
            response.raise_for_status()  # Lève une exception si status >= 400
            computed_image = response.json()

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    return computed_image

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask



def segment_images_batch(image_paths, api_token, max):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        image_paths (list): Liste des chemins vers les images.
        api_token (str): API Token.
        max (int): Maximum d'images à traiter via l'api
    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []

    # Limiter le nombre d'images à traiter
    images_to_process = image_paths[:max]

    # LA BARRE DE PROGRESSION SE CRÉE ICI ↓
    for path in tqdm(images_to_process, desc="🟢 Segmentation", unit="img"):
        image_response = request_for_image(path, api_token)

        original_img = Image.open(path).convert("RGB")
        width, height = get_image_dimensions(path)
        final_mask = create_masks(image_response, width, height)

        if final_mask is not None:
            batch_segmentations.append(final_mask.copy())
        else:
            batch_segmentations.append(None)

        # Pause pour éviter de surcharger l'API
        time.sleep(0.5)

    return batch_segmentations
