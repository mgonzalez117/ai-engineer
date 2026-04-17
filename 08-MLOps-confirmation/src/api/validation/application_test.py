from pydantic import BaseModel, Field
from typing import Optional


# Modèle Pydantic pour application_test.csv
class ApplicationTest(BaseModel):
    """
    Modèle de validation pour les données de demande de prêt (application_test.csv)
    Basé sur la compétition Kaggle Home Credit Default Risk
    """
    SK_ID_CURR: int = Field(..., description="ID unique de la demande de prêt")
    NAME_CONTRACT_TYPE: str = Field(..., description="Type de contrat (Cash loans ou Revolving loans)")
    CODE_GENDER: str = Field(..., description="Genre du client (M, F, XNA)")
    FLAG_OWN_CAR: str = Field(..., description="Indicateur de possession de voiture (Y/N)")
    FLAG_OWN_REALTY: str = Field(..., description="Indicateur de possession de bien immobilier (Y/N)")
    CNT_CHILDREN: int = Field(..., description="Nombre d'enfants")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu total du client")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit")
    AMT_ANNUITY: Optional[float] = Field(None, gt=0, description="Montant de l'annuité du prêt")
    AMT_GOODS_PRICE: Optional[float] = Field(None, gt=0, description="Prix des biens pour lesquels le prêt est accordé")
    NAME_TYPE_SUITE: Optional[str] = Field(None, description="Qui accompagne le client lors de la demande")
    NAME_INCOME_TYPE: str = Field(..., description="Type de revenu du client")
    NAME_EDUCATION_TYPE: str = Field(..., description="Niveau d'éducation du client")
    NAME_FAMILY_STATUS: str = Field(..., description="Statut familial du client")
    NAME_HOUSING_TYPE: str = Field(..., description="Type de logement")
    REGION_POPULATION_RELATIVE: float = Field(..., description="Population relative de la région")
    DAYS_BIRTH: int = Field(..., lt=0,  description="Âge du client en jours (négatif)")
    DAYS_EMPLOYED: int = Field(..., description="Nombre de jours d'emploi (négatif)")
    DAYS_REGISTRATION: float = Field(..., description="Jours depuis l'enregistrement")
    DAYS_ID_PUBLISH: int = Field(..., description="Jours depuis la publication de l'ID")
    OWN_CAR_AGE: Optional[float] = Field(None, description="Âge de la voiture")
    FLAG_MOBIL: int = Field(..., description="Le client a fourni un téléphone mobile (1/0)")
    FLAG_EMP_PHONE: int = Field(..., description="Le client a fourni un téléphone professionnel (1/0)")
    FLAG_WORK_PHONE: int = Field(..., description="Le client a fourni un téléphone de travail (1/0)")
    FLAG_CONT_MOBILE: int = Field(..., description="Le téléphone mobile est joignable (1/0)")
    FLAG_PHONE: int = Field(..., description="Le client a fourni un téléphone fixe (1/0)")
    FLAG_EMAIL: int = Field(..., description="Le client a fourni un email (1/0)")
    OCCUPATION_TYPE: Optional[str] = Field(None, description="Type d'occupation du client")
    CNT_FAM_MEMBERS: Optional[float] = Field(None, description="Nombre de membres de la famille")
    REGION_RATING_CLIENT: int = Field(..., description="Note de la région du client")
    REGION_RATING_CLIENT_W_CITY: int = Field(..., description="Note de la région avec ville")
    WEEKDAY_APPR_PROCESS_START: str = Field(..., description="Jour de la semaine de la demande")
    HOUR_APPR_PROCESS_START: int = Field(..., description="Heure de la demande")
    REG_REGION_NOT_LIVE_REGION: int = Field(..., description="Région d'enregistrement différente de la région de résidence")
    REG_REGION_NOT_WORK_REGION: int = Field(..., description="Région d'enregistrement différente de la région de travail")
    LIVE_REGION_NOT_WORK_REGION: int = Field(..., description="Région de résidence différente de la région de travail")
    REG_CITY_NOT_LIVE_CITY: int = Field(..., description="Ville d'enregistrement différente de la ville de résidence")
    REG_CITY_NOT_WORK_CITY: int = Field(..., description="Ville d'enregistrement différente de la ville de travail")
    LIVE_CITY_NOT_WORK_CITY: int = Field(..., description="Ville de résidence différente de la ville de travail")
    ORGANIZATION_TYPE: str = Field(..., description="Type d'organisation où travaille le client")
    EXT_SOURCE_1: Optional[float] = Field(None, description="Score normalisé de source externe 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Score normalisé de source externe 2")
    EXT_SOURCE_3: Optional[float] = Field(None, description="Score normalisé de source externe 3")
    APARTMENTS_AVG: Optional[float] = Field(None, description="Moyenne des informations sur l'appartement")
    BASEMENTAREA_AVG: Optional[float] = Field(None, description="Moyenne de la surface du sous-sol")
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = Field(None, description="Moyenne des années de début d'exploitation")
    YEARS_BUILD_AVG: Optional[float] = Field(None, description="Moyenne des années de construction")
    COMMONAREA_AVG: Optional[float] = Field(None, description="Moyenne de la surface commune")
    ELEVATORS_AVG: Optional[float] = Field(None, description="Moyenne du nombre d'ascenseurs")
    ENTRANCES_AVG: Optional[float] = Field(None, description="Moyenne du nombre d'entrées")
    FLOORSMAX_AVG: Optional[float] = Field(None, description="Moyenne du nombre maximum d'étages")
    FLOORSMIN_AVG: Optional[float] = Field(None, description="Moyenne du nombre minimum d'étages")
    LANDAREA_AVG: Optional[float] = Field(None, description="Moyenne de la surface du terrain")
    LIVINGAPARTMENTS_AVG: Optional[float] = Field(None, description="Moyenne des appartements habitables")
    LIVINGAREA_AVG: Optional[float] = Field(None, description="Moyenne de la surface habitable")
    NONLIVINGAPARTMENTS_AVG: Optional[float] = Field(None, description="Moyenne des appartements non habitables")
    NONLIVINGAREA_AVG: Optional[float] = Field(None, description="Moyenne de la surface non habitable")
    APARTMENTS_MODE: Optional[float] = Field(None, description="Mode des informations sur l'appartement")
    BASEMENTAREA_MODE: Optional[float] = Field(None, description="Mode de la surface du sous-sol")
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = Field(None, description="Mode des années de début d'exploitation")
    YEARS_BUILD_MODE: Optional[float] = Field(None, description="Mode des années de construction")
    COMMONAREA_MODE: Optional[float] = Field(None, description="Mode de la surface commune")
    ELEVATORS_MODE: Optional[float] = Field(None, description="Mode du nombre d'ascenseurs")
    ENTRANCES_MODE: Optional[float] = Field(None, description="Mode du nombre d'entrées")
    FLOORSMAX_MODE: Optional[float] = Field(None, description="Mode du nombre maximum d'étages")
    FLOORSMIN_MODE: Optional[float] = Field(None, description="Mode du nombre minimum d'étages")
    LANDAREA_MODE: Optional[float] = Field(None, description="Mode de la surface du terrain")
    LIVINGAPARTMENTS_MODE: Optional[float] = Field(None, description="Mode des appartements habitables")
    LIVINGAREA_MODE: Optional[float] = Field(None, description="Mode de la surface habitable")
    NONLIVINGAPARTMENTS_MODE: Optional[float] = Field(None, description="Mode des appartements non habitables")
    NONLIVINGAREA_MODE: Optional[float] = Field(None, description="Mode de la surface non habitable")
    APARTMENTS_MEDI: Optional[float] = Field(None, description="Médiane des informations sur l'appartement")
    BASEMENTAREA_MEDI: Optional[float] = Field(None, description="Médiane de la surface du sous-sol")
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = Field(None, description="Médiane des années de début d'exploitation")
    YEARS_BUILD_MEDI: Optional[float] = Field(None, description="Médiane des années de construction")
    COMMONAREA_MEDI: Optional[float] = Field(None, description="Médiane de la surface commune")
    ELEVATORS_MEDI: Optional[float] = Field(None, description="Médiane du nombre d'ascenseurs")
    ENTRANCES_MEDI: Optional[float] = Field(None, description="Médiane du nombre d'entrées")
    FLOORSMAX_MEDI: Optional[float] = Field(None, description="Médiane du nombre maximum d'étages")
    FLOORSMIN_MEDI: Optional[float] = Field(None, description="Médiane du nombre minimum d'étages")
    LANDAREA_MEDI: Optional[float] = Field(None, description="Médiane de la surface du terrain")
    LIVINGAPARTMENTS_MEDI: Optional[float] = Field(None, description="Médiane des appartements habitables")
    LIVINGAREA_MEDI: Optional[float] = Field(None, description="Médiane de la surface habitable")
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = Field(None, description="Médiane des appartements non habitables")
    NONLIVINGAREA_MEDI: Optional[float] = Field(None, description="Médiane de la surface non habitable")
    FONDKAPREMONT_MODE: Optional[str] = Field(None, description="Mode du fonds de réparation")
    HOUSETYPE_MODE: Optional[str] = Field(None, description="Mode du type de maison")
    TOTALAREA_MODE: Optional[float] = Field(None, description="Mode de la surface totale")
    WALLSMATERIAL_MODE: Optional[str] = Field(None, description="Mode du matériau des murs")
    EMERGENCYSTATE_MODE: Optional[str] = Field(None, description="Mode de l'état d'urgence")
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="Observations du cercle social sur 30 jours")
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="Défauts du cercle social sur 30 jours")
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="Observations du cercle social sur 60 jours")
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(None, description="Défauts du cercle social sur 60 jours")
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None, description="Jours depuis le dernier changement de téléphone")
    FLAG_DOCUMENT_2: int = Field(..., description="Le client a fourni le document 2")
    FLAG_DOCUMENT_3: int = Field(..., description="Le client a fourni le document 3")
    FLAG_DOCUMENT_4: int = Field(..., description="Le client a fourni le document 4")
    FLAG_DOCUMENT_5: int = Field(..., description="Le client a fourni le document 5")
    FLAG_DOCUMENT_6: int = Field(..., description="Le client a fourni le document 6")
    FLAG_DOCUMENT_7: int = Field(..., description="Le client a fourni le document 7")
    FLAG_DOCUMENT_8: int = Field(..., description="Le client a fourni le document 8")
    FLAG_DOCUMENT_9: int = Field(..., description="Le client a fourni le document 9")
    FLAG_DOCUMENT_10: int = Field(..., description="Le client a fourni le document 10")
    FLAG_DOCUMENT_11: int = Field(..., description="Le client a fourni le document 11")
    FLAG_DOCUMENT_12: int = Field(..., description="Le client a fourni le document 12")
    FLAG_DOCUMENT_13: int = Field(..., description="Le client a fourni le document 13")
    FLAG_DOCUMENT_14: int = Field(..., description="Le client a fourni le document 14")
    FLAG_DOCUMENT_15: int = Field(..., description="Le client a fourni le document 15")
    FLAG_DOCUMENT_16: int = Field(..., description="Le client a fourni le document 16")
    FLAG_DOCUMENT_17: int = Field(..., description="Le client a fourni le document 17")
    FLAG_DOCUMENT_18: int = Field(..., description="Le client a fourni le document 18")
    FLAG_DOCUMENT_19: int = Field(..., description="Le client a fourni le document 19")
    FLAG_DOCUMENT_20: int = Field(..., description="Le client a fourni le document 20")
    FLAG_DOCUMENT_21: int = Field(..., description="Le client a fourni le document 21")
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans la dernière heure")
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans le dernier jour")
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans la dernière semaine")
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans le dernier mois")
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans le dernier trimestre")
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(None, description="Nombre de demandes au bureau de crédit dans la dernière année")

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "REGION_POPULATION_RELATIVE": 0.018801,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "DAYS_REGISTRATION": -3648.0,
                "DAYS_ID_PUBLISH": -2120,
                "OWN_CAR_AGE": None,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 0,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 1.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
                "HOUR_APPR_PROCESS_START": 10,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "EXT_SOURCE_1": 0.083037,
                "EXT_SOURCE_2": 0.262949,
                "EXT_SOURCE_3": 0.139376,
                "APARTMENTS_AVG": 0.0247,
                "BASEMENTAREA_AVG": 0.0369,
                "YEARS_BEGINEXPLUATATION_AVG": 0.9722,
                "YEARS_BUILD_AVG": 0.6192,
                "COMMONAREA_AVG": 0.0143,
                "ELEVATORS_AVG": 0.00,
                "ENTRANCES_AVG": 0.0690,
                "FLOORSMAX_AVG": 0.0833,
                "FLOORSMIN_AVG": 0.1250,
                "LANDAREA_AVG": 0.0369,
                "LIVINGAPARTMENTS_AVG": 0.0202,
                "LIVINGAREA_AVG": 0.0190,
                "NONLIVINGAPARTMENTS_AVG": 0.0000,
                "NONLIVINGAREA_AVG": 0.0000,
                "APARTMENTS_MODE": 0.0247,
                "BASEMENTAREA_MODE": 0.0369,
                "YEARS_BEGINEXPLUATATION_MODE": 0.9722,
                "YEARS_BUILD_MODE": 0.6192,
                "COMMONAREA_MODE": 0.0143,
                "ELEVATORS_MODE": 0.00,
                "ENTRANCES_MODE": 0.0690,
                "FLOORSMAX_MODE": 0.0833,
                "FLOORSMIN_MODE": 0.1250,
                "LANDAREA_MODE": 0.0369,
                "LIVINGAPARTMENTS_MODE": 0.0202,
                "LIVINGAREA_MODE": 0.0190,
                "NONLIVINGAPARTMENTS_MODE": 0.0000,
                "NONLIVINGAREA_MODE": 0.0000,
                "APARTMENTS_MEDI": 0.0247,
                "BASEMENTAREA_MEDI": 0.0369,
                "YEARS_BEGINEXPLUATATION_MEDI": 0.9722,
                "YEARS_BUILD_MEDI": 0.6192,
                "COMMONAREA_MEDI": 0.0143,
                "ELEVATORS_MEDI": 0.00,
                "ENTRANCES_MEDI": 0.0690,
                "FLOORSMAX_MEDI": 0.0833,
                "FLOORSMIN_MEDI": 0.1250,
                "LANDAREA_MEDI": 0.0369,
                "LIVINGAPARTMENTS_MEDI": 0.0202,
                "LIVINGAREA_MEDI": 0.0190,
                "NONLIVINGAPARTMENTS_MEDI": 0.0000,
                "NONLIVINGAREA_MEDI": 0.0000,
                "FONDKAPREMONT_MODE": None,
                "HOUSETYPE_MODE": None,
                "TOTALAREA_MODE": None,
                "WALLSMATERIAL_MODE": None,
                "EMERGENCYSTATE_MODE": None,
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
                "DAYS_LAST_PHONE_CHANGE": -1134.0,
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1,
                "FLAG_DOCUMENT_4": 0,
                "FLAG_DOCUMENT_5": 0,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_7": 0,
                "FLAG_DOCUMENT_8": 0,
                "FLAG_DOCUMENT_9": 0,
                "FLAG_DOCUMENT_10": 0,
                "FLAG_DOCUMENT_11": 0,
                "FLAG_DOCUMENT_12": 0,
                "FLAG_DOCUMENT_13": 0,
                "FLAG_DOCUMENT_14": 0,
                "FLAG_DOCUMENT_15": 0,
                "FLAG_DOCUMENT_16": 0,
                "FLAG_DOCUMENT_17": 0,
                "FLAG_DOCUMENT_18": 0,
                "FLAG_DOCUMENT_19": 0,
                "FLAG_DOCUMENT_20": 0,
                "FLAG_DOCUMENT_21": 0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0
            }
        }