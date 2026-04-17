from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from pathlib import Path

class ApplicationBureauPipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline complet qui encapsule :
    - application + bureau en entrée
    - toutes les étapes de préparation
    """

    def __init__(self):
        # objets "à retenir" entre fit et transform
        self.le_dict = {}
        self.scaler = None
        self.numeric_cols = []
        self.medians = {}
        self.bureau_one_hot_cols = []
        self.final_columns = []

    def fit(self, application: pd.DataFrame, bureau: pd.DataFrame, y=None):
        df = application.copy()

        # 1. encodage binaire
        df, self.le_dict = encode_binary_columns(df)

        # 2. one-hot
        df = one_hot_encode_categorical(df)

        # 3. anomalies
        df = handle_days_employed_anomaly(df)
        df = fix_days_birth(df)

        # 4. bureau (fit sur les colonnes one-hot bureau)
        df, self.bureau_one_hot_cols = aggregate_bureau_data(df, bureau, one_hot_cols_fit=None)

        # 5. nouvelles features
        df = create_new_features(df)

        # 6. imputation (fit)
        df, self.medians = impute_missing_values(df, fit=True)

        # 7. scaling (fit)
        df, self.scaler, self.numeric_cols = scale_numeric_features(df, fit=True)

        # 8. noms de colonnes
        df = clean_column_names(df)

        self.final_columns = df.columns.tolist()

        return self

    def transform(self, application: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
        df = application.copy()

        # 1. encodage binaire avec le_dict
        for col, le in self.le_dict.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        # 2. one-hot + alignement colonnes
        df = one_hot_encode_categorical(df)

        # 3. anomalies
        df = handle_days_employed_anomaly(df)
        df = fix_days_birth(df)

        # 4. bureau avec colonnes one-hot figées
        df, _ = aggregate_bureau_data(df, bureau, one_hot_cols_fit=self.bureau_one_hot_cols)

        # 5. nouvelles features
        df = create_new_features(df)

        # 6. imputation avec les médianes du fit
        df, _ = impute_missing_values(df, medians_fit=self.medians, fit=False)

        # 7. scaling avec scaler du fit
        df, _, _ = scale_numeric_features(df, scaler=self.scaler, fit=False)

        # 8. clean colonnes
        df = clean_column_names(df)

        # 9. réalignement sur colonnes finales (train)
        missing_cols = [c for c in self.final_columns if c not in df.columns]
        if missing_cols:
            # optimisation : une seule affectation pour toutes les colonnes manquantes
            df = df.assign(**{c: 0 for c in missing_cols})

        # on garde l’ordre des colonnes du fit
        df = df[self.final_columns]

        return df

def encode_binary_columns(df):
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) <= 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
    return df, le_dict


def one_hot_encode_categorical(df):
    return pd.get_dummies(df)


def handle_days_employed_anomaly(df):
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    return df


def fix_days_birth(df):
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    return df


def aggregate_bureau_data(df, bureau, one_hot_cols_fit=None):
    bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE'], prefix='CREDIT_ACTIVE')
    bureau = pd.get_dummies(bureau, columns=['CREDIT_CURRENCY'], prefix='CREDIT_CURRENCY', drop_first=True)
    bureau = pd.get_dummies(bureau, columns=['CREDIT_TYPE'], prefix='CREDIT_TYPE', drop_first=True)

    one_hot_cols = [c for c in bureau.columns if c.startswith(('CREDIT_ACTIVE_', 'CREDIT_CURRENCY_', 'CREDIT_TYPE_'))]

    # Si on est en transform, on aligne sur les colonnes vues au fit
    if one_hot_cols_fit is not None:
        for col in one_hot_cols_fit:
            if col not in bureau.columns:
                bureau[col] = 0
        bureau = bureau[one_hot_cols_fit + [
            c for c in bureau.columns
            if c not in one_hot_cols_fit
        ]]
        one_hot_cols = one_hot_cols_fit

    agg_kwargs = {
        'bureau_count': ('SK_ID_BUREAU', 'count'),
        'amt_credit_sum': ('AMT_CREDIT_SUM', 'sum'),
        'amt_credit_mean': ('AMT_CREDIT_SUM', 'mean'),
        'max_days_credit_enddate': ('DAYS_CREDIT_ENDDATE', 'max'),
    }
    for col in one_hot_cols:
        agg_kwargs[col] = (col, 'sum')

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(**agg_kwargs).reset_index()

    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    df['bureau_missing'] = df['bureau_count'].isna().astype(int)
    df['amt_credit_mean_missing'] = df['amt_credit_mean'].isna().astype(int)
    df['max_days_credit_enddate_missing'] = df['max_days_credit_enddate'].isna().astype(int)

    df['bureau_count'] = df['bureau_count'].fillna(0)
    df['amt_credit_sum'] = df['amt_credit_sum'].fillna(0)
    df['amt_credit_mean'] = df['amt_credit_mean'].fillna(0)
    df['max_days_credit_enddate'] = df['max_days_credit_enddate'].fillna(-1)

    # optimisation : on réassigne les colonnes en une seule fois au lieu de boucler (pour réduire __set_item__)
    existing_one_hot_cols = [c for c in one_hot_cols if c in df.columns]
    if existing_one_hot_cols:
        df[existing_one_hot_cols] = df[existing_one_hot_cols].fillna(0)

    return df, one_hot_cols


def create_new_features(df):
    df['CREDIT_DURATION'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    df['AGE_CREDIT_DURATION_RATIO'] = df['CREDIT_DURATION'] / (abs(df['DAYS_BIRTH']) / 365.25)
    df['AGE_EMPLOYED_RATIO'] = abs(df['DAYS_BIRTH']) / (abs(df['DAYS_EMPLOYED']) + 1)
    new_feat_cols = ['CREDIT_DURATION', 'AGE_CREDIT_DURATION_RATIO', 'AGE_EMPLOYED_RATIO']
    df[new_feat_cols] = df[new_feat_cols].fillna(-1)
    return df


def scale_numeric_features(df, scaler=None, fit=True):
    if fit:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler, numeric_cols
    else:
        # Utiliser les colonnes que le scaler a vues au fit
        expected_cols = list(scaler.feature_names_in_)

        # Ajouter les colonnes manquantes avec 0
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Transformer uniquement ces colonnes, dans le bon ordre
        df[expected_cols] = scaler.transform(df[expected_cols])

        return df, scaler, expected_cols


def impute_missing_values(df, medians_fit=None, fit=True):
    cols_zero = [
        'COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI',
        'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_MEDI',
        'LIVINGAPARTMENTS_AVG', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_MEDI',
        'LANDAREA_AVG', 'LANDAREA_MODE', 'LANDAREA_MEDI',
        'BASEMENTAREA_MODE', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MEDI',
        'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE', 'NONLIVINGAREA_MEDI',
        'APARTMENTS_AVG', 'APARTMENTS_MODE', 'APARTMENTS_MEDI',
        'ELEVATORS_AVG', 'ELEVATORS_MODE', 'ELEVATORS_MEDI',
        'ENTRANCES_AVG', 'ENTRANCES_MODE', 'ENTRANCES_MEDI',
        'LIVINGAREA_AVG', 'LIVINGAREA_MODE', 'LIVINGAREA_MEDI',
        'FLOORSMAX_AVG', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI'
    ]
    cols_zero = [c for c in cols_zero if c in df.columns]
    if cols_zero:
        df[cols_zero] = df[cols_zero].fillna(0)

    cols_median_with_flag = [
        'FLOORSMIN_MEDI', 'FLOORSMIN_MODE', 'FLOORSMIN_AVG',
        'YEARS_BUILD_AVG', 'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI',
        'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI',
        'TOTALAREA_MODE', 'OWN_CAR_AGE',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_EMPLOYED',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'AMT_GOODS_PRICE', 'AMT_ANNUITY',
        'CNT_FAM_MEMBERS', 'DAYS_LAST_PHONE_CHANGE'
    ]
    cols_median_with_flag = [c for c in cols_median_with_flag if c in df.columns and c not in cols_zero]

    if not cols_median_with_flag:
        return df, ({} if fit else (medians_fit or {}))

    if fit:
        medians = {c: df[c].median() for c in cols_median_with_flag}

        # optimisation : flags *_na en une seule assignation
        na_flags = {f"{c}_na": df[c].isna().astype(int) for c in cols_median_with_flag}
        df = df.assign(**na_flags)

        # remplissage médian en un block (par colonne, mais boucle courte)
        for c in cols_median_with_flag:
            df[c] = df[c].fillna(medians[c])
    else:
        medians = medians_fit or {}
        # flags *_na
        na_flags = {f"{c}_na": df[c].isna().astype(int) for c in cols_median_with_flag}
        df = df.assign(**na_flags)

        for c in cols_median_with_flag:
            df[c] = df[c].fillna(medians.get(c, df[c].median()))

    return df, medians


def clean_column_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

def load_raw_data(data_dir='.data'):
    app_train = pd.read_csv(f'{data_dir}/application_train.csv')
    bureau = pd.read_csv(f'{data_dir}/bureau.csv')
    return app_train, bureau


def save_preprocessing_pipeline(pipeline, output_dir='models'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, f'{output_dir}/pipeline.joblib')
    print(f"Pipeline sauvegardé dans {output_dir}/pipeline.joblib")

