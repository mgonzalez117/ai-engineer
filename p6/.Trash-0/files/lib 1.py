from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def display_pr_and_roc_curves(model, X, y, cv):
    # Probabilités prédites en cross-validation (classe positive = 1)
    y_scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    # --- Courbe Précision-Rappel ---
    precision, recall, thresholds_pr = precision_recall_curve(y, y_scores)
    avg_precision = average_precision_score(y, y_scores)

    # Calcul automatique du taux de départs
    taux_depart = y.mean()

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)  # Panel gauche
    plt.plot(recall, precision, label=f'AP (AUC-PR) = {avg_precision:.2f}', color="blue")
    plt.hlines(taux_depart, 0, 1, colors='red', linestyles='--',
               label=f'Taux de départs = {taux_depart:.2f}')
    plt.xlabel("Rappel (Recall)")
    plt.ylabel("Précision (Precision)")
    plt.title("Courbe Précision–Rappel (cross-validation)")
    plt.legend()
    plt.grid(True)

    # --- Courbe ROC ---
    fpr, tpr, thresholds_roc = roc_curve(y, y_scores)
    auc_roc = roc_auc_score(y, y_scores)

    plt.subplot(1, 2, 2)  # Panel droit
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], "k--", label="Hasard (0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Courbe ROC (cross-validation)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()