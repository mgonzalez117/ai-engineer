import pandas as pd
import json
import numpy as np

# Charger les données
app_test = pd.read_csv('.data/application_test.csv')
bureau = pd.read_csv('.data/bureau.csv')

# Prendre la première ligne
first_row = app_test.iloc[0]
sk_id_curr = first_row['SK_ID_CURR']

print(f"SK_ID_CURR: {sk_id_curr}")
print(f"\n=== Application (première ligne) ===")
print(first_row.to_dict())

# Extraire les lignes bureau correspondantes
bureau_rows = bureau[bureau['SK_ID_CURR'] == sk_id_curr]

print(f"\n=== Bureau ({len(bureau_rows)} lignes) ===")
if len(bureau_rows) > 0:
    print(bureau_rows.to_dict('records'))
else:
    print("Aucune ligne bureau pour ce SK_ID_CURR")

# Remplacer les NaN par None avant export JSON
first_row_clean = first_row.replace({np.nan: None})
bureau_rows_clean = bureau_rows.replace({np.nan: None})

# Sauvegarder dans des fichiers JSON pour faciliter les tests API
with open('.data/extract/test_sample_application.json', 'w') as f:
    json.dump(first_row_clean.to_dict(), f, indent=2)

with open('.data/extract/test_sample_bureau.json', 'w') as f:
    json.dump(bureau_rows_clean.to_dict('records'), f, indent=2)

# Sauvegarder également en CSV (inchangé)
pd.DataFrame([first_row]).to_csv('.data/extract/test_sample_application.csv', index=False)
bureau_rows.to_csv('.data/extract/test_sample_bureau.csv', index=False)

print("\n✓ Fichiers sauvegardés:")
print("  - test_sample_application.json")
print("  - test_sample_bureau.json")
print("  - test_sample_application.csv")
print("  - test_sample_bureau.csv")