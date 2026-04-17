import src.ml.pipeline as preparation
import src.ml.model as model
from pathlib import Path
import pandas as pd

# 1. données brutes
app_train, bureau = preparation.load_raw_data()

# 1bis. Séparer X et y AVANT le pipeline
y = app_train["TARGET"]
X = app_train.drop(columns=["TARGET"])

# 2. instancier le pipeline
pipeline = preparation.ApplicationBureauPipeline()

# 3. fit + transform sur X uniquement (sans TARGET)
pipeline.fit(X, bureau)
X_processed = pipeline.transform(X, bureau)

# 4. sauvegarde pipeline
preparation.save_preprocessing_pipeline(pipeline)

# 5. sauvegarde données transformées pour train_model
Path('.data/ml').mkdir(parents=True, exist_ok=True)
df_out = pd.DataFrame(X_processed)

# on rajoute la target dans le app_train :
df_out["TARGET"] = y.values
df_out.to_csv('.data/ml/app_train.csv', index=False)

print("Préprocessing terminé et pipeline sauvegardée.")

# 6. entraînement modèle (qui lit .data/ml/app_train.csv)
results = model.train_model()
print("Modèle entraîné et sauvegardé.")