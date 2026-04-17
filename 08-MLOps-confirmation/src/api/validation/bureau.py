from pydantic import BaseModel, Field
from typing import Optional

# Modèle Pydantic pour bureau.csv
class Bureau(BaseModel):
    """
    Modèle de validation pour les données du bureau de crédit (bureau.csv)
    Contient les crédits précédents des clients auprès d'autres institutions financières
    """
    SK_ID_CURR: int = Field(..., description="ID du client (clé étrangère vers application)")
    SK_ID_BUREAU: int = Field(..., description="ID unique du crédit précédent dans le bureau")
    CREDIT_ACTIVE: str = Field(..., description="Statut du crédit (Closed, Active, Sold, Bad debt)")
    CREDIT_CURRENCY: str = Field(..., description="Devise du crédit")
    DAYS_CREDIT: int = Field(..., description="Nombre de jours avant la demande actuelle où le crédit a été accordé (négatif)")
    CREDIT_DAY_OVERDUE: int = Field(..., description="Nombre de jours de retard sur le crédit du bureau")
    DAYS_CREDIT_ENDDATE: Optional[float] = Field(None, description="Jours restants jusqu'à la fin du crédit (négatif)")
    DAYS_ENDDATE_FACT: Optional[float] = Field(None, description="Jours depuis la fermeture effective du crédit (négatif)")
    AMT_CREDIT_MAX_OVERDUE: Optional[float] = Field(None, description="Montant maximum de retard sur le crédit")
    CNT_CREDIT_PROLONG: int = Field(..., description="Nombre de fois où le crédit a été prolongé")
    AMT_CREDIT_SUM: Optional[float] = Field(None, description="Montant total du crédit actuel")
    AMT_CREDIT_SUM_DEBT: Optional[float] = Field(None, description="Dette actuelle sur le crédit")
    AMT_CREDIT_SUM_LIMIT: Optional[float] = Field(None, description="Limite de crédit actuelle sur la carte de crédit")
    AMT_CREDIT_SUM_OVERDUE: Optional[float] = Field(None, description="Montant actuel en retard sur le crédit")
    CREDIT_TYPE: str = Field(..., description="Type de crédit (Consumer credit, Credit card, Mortgage, etc.)")
    DAYS_CREDIT_UPDATE: int = Field(..., description="Nombre de jours avant la demande où les informations ont été mises à jour")
    AMT_ANNUITY: Optional[float] = Field(None, description="Montant de l'annuité du crédit du bureau")

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 215354,
                "SK_ID_BUREAU": 5714462,
                "CREDIT_ACTIVE": "Closed",
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -497,
                "CREDIT_DAY_OVERDUE": 0,
                "DAYS_CREDIT_ENDDATE": -153.0,
                "DAYS_ENDDATE_FACT": -153.0,
                "AMT_CREDIT_MAX_OVERDUE": None,
                "CNT_CREDIT_PROLONG": 0,
                "AMT_CREDIT_SUM": 91323.0,
                "AMT_CREDIT_SUM_DEBT": 0.0,
                "AMT_CREDIT_SUM_LIMIT": None,
                "AMT_CREDIT_SUM_OVERDUE": 0.0,
                "CREDIT_TYPE": "Consumer credit",
                "DAYS_CREDIT_UPDATE": -131,
                "AMT_ANNUITY": None
            }
        }