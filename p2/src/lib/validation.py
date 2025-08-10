import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Classes à exclure (background, chaussures, bras)
EXCLUDED_CLASSES = {0, 9, 10, 12, 13, 14, 15}

def calculate_iou(pred_mask, true_mask, class_id):
    """
    Calcule l'IoU pour une classe spécifique
    """
    pred_class = (pred_mask == class_id)
    true_class = (true_mask == class_id)

    intersection = np.logical_and(pred_class, true_class).sum()
    union = np.logical_or(pred_class, true_class).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def load_mask(mask_path):
    """
    Charge un masque depuis un fichier image
    """
    mask = Image.open(mask_path)
    return np.array(mask)


def compare_masks(masks_pred_dir, masks_dir):
    """
    Compare les masques prédits avec les masques de référence
    """
    results = []

    # Lister tous les fichiers de masques
    pred_files = sorted([f for f in os.listdir(masks_pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    true_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Trouvé {len(pred_files)} masques prédits et {len(true_files)} masques de référence")

    for pred_file in pred_files:
        # Chercher le fichier correspondant dans les masques de référence
        base_name = os.path.splitext(pred_file)[0]

        # Essayer différents formats de nommage
        possible_names = [
            pred_file,
            base_name + '.png',
            base_name + '.jpg',
            base_name + '.jpeg'
        ]

        true_file = None
        for name in possible_names:
            if name in true_files:
                true_file = name
                break

        if true_file is None:
            print(f"⚠️ Pas de masque de référence trouvé pour {pred_file}")
            continue

        # Charger les masques
        pred_mask = load_mask(os.path.join(masks_pred_dir, pred_file))
        true_mask = load_mask(os.path.join(masks_dir, true_file))

        # Vérifier que les dimensions correspondent
        if pred_mask.shape != true_mask.shape:
            print(f"⚠️ Dimensions différentes pour {pred_file}: {pred_mask.shape} vs {true_mask.shape}")
            continue

        # Obtenir les classes uniques
        pred_classes = np.unique(pred_mask)
        true_classes = np.unique(true_mask)
        all_classes = np.unique(np.concatenate([pred_classes, true_classes]))

        # Calculer l'IoU pour chaque classe
        iou_scores = []
        for class_id in all_classes:
            if class_id in EXCLUDED_CLASSES:
                continue
            iou = calculate_iou(pred_mask, true_mask, class_id)
            iou_scores.append(iou)

        # IoU moyen pour cette image
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0

        results.append({
            'image': pred_file,
            'mean_iou': mean_iou,
            'num_classes': len(iou_scores),
            'iou_scores': iou_scores
        })

        print(f"✅ {pred_file}: mIoU = {mean_iou:.3f} ({len(iou_scores)} classes)")

    return results


def analyze_results(results):
    """
    Analyse les résultats et affiche les statistiques
    """
    if not results:
        print("❌ Aucun résultat à analyser")
        return

    mean_ious = [r['mean_iou'] for r in results]

    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DE L'ANALYSE IoU")
    print("=" * 50)

    print(f"Nombre d'images analysées: {len(results)}")
    print(f"mIoU global: {np.mean(mean_ious):.3f}")
    print(f"mIoU médian: {np.median(mean_ious):.3f}")
    print(f"Écart-type: {np.std(mean_ious):.3f}")
    print(f"Min: {np.min(mean_ious):.3f}")
    print(f"Max: {np.max(mean_ious):.3f}")

    # Classification selon le tableau IoU
    excellent = sum(1 for iou in mean_ious if iou > 0.7)
    acceptable = sum(1 for iou in mean_ious if 0.5 <= iou <= 0.7)
    mauvais = sum(1 for iou in mean_ious if iou < 0.5)

    print(f"\n📈 RÉPARTITION QUALITATIVE:")
    print(f"✅ Bon (IoU > 0.7): {excellent}/{len(results)} ({excellent / len(results) * 100:.1f}%)")
    print(f"⚠️ Acceptable (0.5-0.7): {acceptable}/{len(results)} ({acceptable / len(results) * 100:.1f}%)")
    print(f"❌ Mauvais (< 0.5): {mauvais}/{len(results)} ({mauvais / len(results) * 100:.1f}%)")

    # Graphique
    plt.figure(figsize=(10, 6))
    plt.hist(mean_ious, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mean_ious), color='red', linestyle='--', label=f'Moyenne: {np.mean(mean_ious):.3f}')
    plt.axvline(0.5, color='orange', linestyle='--', label='Seuil acceptable (0.5)')
    plt.axvline(0.7, color='green', linestyle='--', label='Seuil bon (0.7)')
    plt.xlabel('IoU Score')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des scores IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
