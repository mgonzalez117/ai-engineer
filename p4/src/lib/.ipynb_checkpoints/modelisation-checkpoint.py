from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

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