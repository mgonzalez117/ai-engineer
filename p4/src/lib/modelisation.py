from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
from sklearn.model_selection import cross_val_predict

def display_cross_validate_results(cv_results):
    # Créer un DataFrame avec les métriques
    metrics_df = pd.DataFrame({
        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Moyenne': [
            cv_results['test_accuracy'].mean(),
            cv_results['test_precision'].mean(),
            cv_results['test_recall'].mean(),
            cv_results['test_f1'].mean()
        ],
        'Écart-type': [
            cv_results['test_accuracy'].std(),
            cv_results['test_precision'].std(),
            cv_results['test_recall'].std(),
            cv_results['test_f1'].std()
        ]
    })

    # Arrondir les valeurs
    metrics_df['Moyenne'] = metrics_df['Moyenne'].round(4)
    metrics_df['Écart-type'] = metrics_df['Écart-type'].round(4)

    # Transformer le tableau en HTML
    html_table = metrics_df.to_html(index=False)

    # L’afficher avec une marge à gauche
    display(HTML(f"<div style='margin-left:50px'>{html_table}</div>"))

def display_confusion_matrix(model, X, y, cv):
    # Prédictions par cross-validation
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Matrice de confusion
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()


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