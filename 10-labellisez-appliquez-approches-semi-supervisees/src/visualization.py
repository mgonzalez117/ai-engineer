import matplotlib.pyplot as plt
import numpy as np

def plot_labeled_comparison(X_2d, y_true, y_pred, algo_name, ari_score, figsize=(16, 6)):
    """
    Affiche côte à côte les vrais labels et les prédictions du clustering

    Args:
        X_2d: Données réduites en 2D (n_samples, 2)
        y_true: Vrais labels (n_samples,)
        y_pred: Labels prédits par clustering (n_samples,)
        algo_name: Nom de l'algorithme (str)
        ari_score: Score ARI (float)
        figsize: Taille de la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Graphique 1 : Vrais labels
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=y_true, cmap='coolwarm', alpha=0.6, s=50)
    axes[0].set_title('Données labellisées - Vrais labels', fontsize=14)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(['Normal', 'Cancer'])

    # Graphique 2 : Prédictions clustering
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=y_pred, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title(f'Clustering {algo_name} (ARI={ari_score:.3f})', fontsize=14)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    plt.tight_layout()
    return fig


def plot_unlabeled_clusters(X_2d, y_pred, algo_name, figsize=(10, 6)):
    """
    Affiche les clusters des données unlabeled

    Args:
        X_2d: Données réduites en 2D (n_samples, 2)
        y_pred: Labels prédits par clustering (n_samples,)
        algo_name: Nom de l'algorithme (str)
        figsize: Taille de la figure
    """
    fig = plt.figure(figsize=figsize)
    plt.scatter(X_2d[:, 0], X_2d[:, 1],
                c=y_pred, cmap='viridis', alpha=0.6, s=50)
    plt.title(f'Données unlabeled - Labellisation faible ({algo_name})', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Cluster')
    return fig