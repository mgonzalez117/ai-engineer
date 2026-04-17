import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

def train_model(
        data_path=".data/ml/app_train.csv",
        target_col="TARGET",
        seuil_utilisateur=0.1,
        model_save_path="models/model.joblib",
        random_state=42
):
    """
    Entraîne un modèle XGBoost avec SMOTE et sauvegarde le pipeline.

    Args:
        data_path: Chemin vers le fichier CSV des données
        target_col: Nom de la colonne cible
        seuil_utilisateur: Seuil de classification
        model_save_path: Chemin de sauvegarde du modèle
        random_state: Seed pour la reproductibilité

    Returns:
        dict: Dictionnaire contenant le pipeline et les métriques
    """

    # Chargement des données
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Cross-validation
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    scoring = {
        "auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    # Modèle + Pipeline
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="auc"
    )

    pipeline = Pipeline([
        ('smote', SMOTE(random_state=random_state, sampling_strategy=0.5)),
        ('clf', model)
    ])

    # Cross-validation
    print(" Cross-validation en cours...")
    cv_results = cross_validate(
        pipeline,
        X=X_train,
        y=y_train,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    # Affichage des résultats CV
    print("\n Résultats Cross-Validation :")
    cv_metrics = {}
    for metric_name in scoring.keys():
        key = f"test_{metric_name}"
        if key in cv_results:
            mean_val = np.mean(cv_results[key])
            std_val = np.std(cv_results[key])
            cv_metrics[metric_name] = {"mean": mean_val, "std": std_val}
            print(f"  {metric_name}: {mean_val:.4f} (+/- {std_val:.4f})")

    # Entraînement final
    print("\n Entraînement du modèle final...")
    pipeline.fit(X_train, y_train)

    # Prédictions sur test
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_utilisateur = (y_proba >= seuil_utilisateur).astype(int)

    # Métriques finales
    test_metrics = {
        "precision": precision_score(y_test, y_pred_utilisateur, zero_division=0),
        "recall": recall_score(y_test, y_pred_utilisateur, zero_division=0),
        "f1": f1_score(y_test, y_pred_utilisateur, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "seuil": seuil_utilisateur
    }

    print("\n Métriques sur le test set :")
    print(f"  Seuil utilisé: {test_metrics['seuil']}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Sauvegarde du modèle
    print(f"\n Sauvegarde du modèle...")
    joblib.dump(pipeline, model_save_path)
    print(f" Modèle sauvegardé dans {model_save_path}")

    return {
        "pipeline": pipeline,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba
    }