from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Literal

class PredictionRequest(BaseModel):
    # Champs numériques avec contraintes de base
    age: int = Field(ge=18, le=70)
    revenu_mensuel: float = Field(gt=0)
    nombre_experiences_precedentes: int = Field(ge=0)
    annee_experience_totale: int = Field(ge=0)
    annees_dans_l_entreprise: int = Field(ge=0)
    annees_dans_le_poste_actuel: int = Field(ge=0)
    note_evaluation_precedente: int = Field(ge=1, le=5)
    niveau_hierarchique_poste: int = Field(ge=1, le=5)
    note_evaluation_actuelle: int = Field(ge=1, le=5)
    nombre_participation_pee: int = Field(ge=0)
    nb_formations_suivies: int = Field(ge=0)
    distance_domicile_travail: int = Field(ge=0)
    niveau_education: int = Field(ge=1, le=5)
    annees_depuis_la_derniere_promotion: int = Field(ge=0)
    annes_sous_responsable_actuel: int = Field(ge=0)

    # Scores de satisfaction (1-5)
    satisfaction_employee_environnement: float = Field(ge=1, le=5)
    satisfaction_employee_nature_travail: float = Field(ge=1, le=5)
    satisfaction_employee_equipe: float = Field(ge=1, le=5)
    satisfaction_employee_equilibre_pro_perso: float = Field(ge=1, le=5)

    # Champs catégoriels avec valeurs possibles
    genre: Literal["M", "F"]
    statut_marital: Literal["Célibataire", "Marié(e)", "Divorcé(e)"]
    departement: Literal["Commercial", "Consulting", "Ressources Humaines"]
    poste: Literal[
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Représentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead"
    ]
    domaine_etude: Literal[
        "Entrepreunariat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale"
    ]
    heure_supplementaires: Literal["Oui", "Non"]
    augementation_salaire_precedente: str = Field(..., pattern=r"^\d{1,2}%$", description="Pourcentage au format 'X%' ou 'XX%'")
    frequence_deplacement: Literal["Jamais", "Occasionnel", "Frequent"]

    @model_validator(mode='after')
    def validate_coherence(self):
        if self.annees_dans_le_poste_actuel > self.annees_dans_l_entreprise:
            raise ValueError("Années dans le poste ne peuvent pas dépasser les années dans l'entreprise.")

        if self.annee_experience_totale < self.annees_dans_l_entreprise:
            raise ValueError("Expérience totale ne peut pas être inférieure aux années dans l'entreprise.")

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35, "genre": "M", "revenu_mensuel": 3500.0,
                "nombre_experiences_precedentes": 2, "annee_experience_totale": 10,
                "annees_dans_l_entreprise": 5, "annees_dans_le_poste_actuel": 3,
                "note_evaluation_precedente": 4, "niveau_hierarchique_poste": 2,
                "note_evaluation_actuelle": 4, "nombre_participation_pee": 2,
                "nb_formations_suivies": 5, "distance_domicile_travail": 15,
                "niveau_education": 3, "annees_depuis_la_derniere_promotion": 2,
                "annes_sous_responsable_actuel": 3,
                "satisfaction_employee_environnement": 4.0,
                "satisfaction_employee_nature_travail": 3.5,
                "satisfaction_employee_equipe": 4.0,
                "satisfaction_employee_equilibre_pro_perso": 2.5,
                "statut_marital": "Marié(e)", "departement": "Consulting", "poste": "Manager",
                "domaine_etude": "Infra & Cloud", "heure_supplementaires": "Oui",
                "augementation_salaire_precedente": "5%",
                "frequence_deplacement": "Occasionnel"
            }
        }
    )