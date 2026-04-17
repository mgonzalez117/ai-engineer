from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
from sklearn.model_selection import cross_val_predict

class NormalizationPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.colonnes_bool = ['heure_supplementaires', 'a_quitte_l_entreprise']
        self.colonnes_notes = [
            'satisfaction_employee_environnement',
            'note_evaluation_precedente',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso',
            'note_evaluation_actuelle'
        ]
    
    def normalize_average(self, x):
        """Convertit '15%' -> 0.15"""
        if isinstance(x, str):
            return round(float(x.replace('%', '')) / 100, 3)
        return x
    
    def normalize_note(self, x):
        """Normalise les notes (à adapter selon ta logique)"""
        return round((x - 1) / (4 - 1), 3)
    
    def fit(self, X, y=None):
        # Rien à apprendre ici
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Booléens (Oui/Non -> 1/0)
        for col in self.colonnes_bool:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ["oui", "true", "1"] else 0)
        
        # Genre (M/F -> 0/1)
        if "genre" in df.columns:
            df["genre"] = df["genre"].apply(lambda x: 0 if str(x).upper() == "M" else 1)
        
        # Pourcentage -> float
        if 'augementation_salaire_precedente' in df.columns:
            df['augementation_salaire_precedente'] = df['augementation_salaire_precedente'].apply(self.normalize_average)
        
        # Normalisation des notes
        for col in self.colonnes_notes:
            if col in df.columns:
                df[col] = df[col].apply(self.normalize_note)
        
        return df

class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int, handle_unknown='ignore')
        self.categories_non_ordinales = ['statut_marital', 'departement', 'poste', 'domaine_etude']
        self.feature_names = None
    
    def fit(self, X, y=None):
        # Fit une seule fois à l'entraînement
        self.encoder.fit(X[self.categories_non_ordinales])
        self.feature_names = self.encoder.get_feature_names_out(self.categories_non_ordinales)
        return self
    
    def transform(self, X):
        X = X.copy()
        # Transform seulement (pas de fit)
        encoded_data = self.encoder.transform(X[self.categories_non_ordinales])
        df_encoded = pd.DataFrame(encoded_data, columns=self.feature_names, index=X.index)
        
        # Supprimer colonnes d'origine et ajouter encodées
        X = X.drop(self.categories_non_ordinales, axis=1)
        X = pd.concat([X, df_encoded], axis=1)
        return X

class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Définir l'ordre des catégories (important pour l'ordinalité)
        self.categories = [['Aucun', 'Occasionnel', 'Frequent']]
        self.encoder = OrdinalEncoder(categories=self.categories, handle_unknown='use_encoded_value', unknown_value=-1)
        self.column = 'frequence_deplacement'
    
    def fit(self, X, y=None):
        # Fit une seule fois
        if self.column in X.columns:
            self.encoder.fit(X[[self.column]])
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = self.encoder.transform(X[[self.column]])
        return X

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or ['revenu_mensuel']
        self.scalers = {}

    def fit(self, X, y=None):
        # Fit un scaler pour chaque colonne
        for col in self.columns:
            if col in X.columns:
                scaler = StandardScaler()
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.scalers:
                X[col] = self.scalers[col].transform(X[[col]]).round(3)
        return X

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.seuil_sous_paye = None
        self.satisfaction_cols = [
            'satisfaction_employee_environnement',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso'
        ]
    
    def fit(self, X, y=None):
        # Calculer le seuil du percentile 0.4 sur les données d'entraînement
        rapport = X['revenu_mensuel'] / (X['annee_experience_totale'] + 1)
        self.seuil_sous_paye = rapport.quantile(0.4)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Rapport revenu/expérience
        X['rapport_revenu_experience'] = (
            X['revenu_mensuel'] / (X['annee_experience_totale'] + 1)
        ).round(3)
        
        # 2. Sous-payé (utilise le seuil appris à l'entraînement)
        X['sous_paye'] = (X['rapport_revenu_experience'] < self.seuil_sous_paye).astype(int)
        
        # 3. Satisfaction générale
        if all(col in X.columns for col in self.satisfaction_cols):
            X['satisfaction_generale'] = X[self.satisfaction_cols].mean(axis=1)
            X = X.drop(self.satisfaction_cols, axis=1)
        
        # 4. Démotivation importante
        X['demotivation_importante'] = (
            (X['note_evaluation_actuelle'] < X['note_evaluation_precedente']) &
            (X['satisfaction_generale'] < 0.5)
        ).astype(int)
        
        # 5. Stagnation au poste
        X['stagnation_poste'] = (
            X['annees_dans_le_poste_actuel'] / (X['annees_dans_l_entreprise'] + 1)
        ).round(3)
        
        return X

def display_cross_validate_results(cv_results):
    # Créer un DataFrame avec les métriques
    metrics_df = pd.DataFrame({
        'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
        'Moyenne': [
            cv_results['test_accuracy'].mean(),
            cv_results['test_precision'].mean(),
            cv_results['test_recall'].mean(),
            cv_results['test_f1'].mean(),
            cv_results['test_roc_auc'].mean()
        ],
        'Écart-type': [
            cv_results['test_accuracy'].std(),
            cv_results['test_precision'].std(),
            cv_results['test_recall'].std(),
            cv_results['test_f1'].std(),
            cv_results['test_roc_auc'].mean()
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