CREATE TABLE sirh_domaine_etude (
    domaine_etude_id INTEGER PRIMARY KEY,
    intitule VARCHAR(50)
);

CREATE TABLE sirh_departement (
    departement_id INTEGER PRIMARY KEY,
    nom VARCHAR(50)
);

CREATE TABLE sirh_poste (
    poste_id INTEGER PRIMARY KEY,
    intitule VARCHAR(50)
);

CREATE TABLE sirh_statut_marital (
    statut_marital_id INTEGER PRIMARY KEY,
    intitule VARCHAR(20)
);

CREATE TABLE sirh_frequence_deplacement (
    frequence_deplacement_id INTEGER PRIMARY KEY,
    intitule VARCHAR(20)
);

CREATE TABLE sirh_employee (
    employee_id INTEGER PRIMARY KEY,
    a_quitte_l_entreprise BOOLEAN,
    age INTEGER,
    genre CHAR(1) CHECK (genre IN ('M','F')), /* 'M' or 'F'*/
    revenu_mensuel INTEGER,
    nombre_experiences_precedentes INTEGER,
    annees_experience_totale INTEGER,
    annees_dans_l_entreprise INTEGER,
    annees_dans_le_poste_actuel INTEGER,
    niveau_education INTEGER,
    heures_supplementaires BOOLEAN,
    statut_marital_id INTEGER,
    departement_id INTEGER,
    poste_id INTEGER,
    domaine_etude_id INTEGER,
    FOREIGN KEY (statut_marital_id) REFERENCES sirh_statut_marital(statut_marital_id),
    FOREIGN KEY (departement_id) REFERENCES sirh_departement(departement_id),
    FOREIGN KEY (poste_id) REFERENCES sirh_poste(poste_id),
    FOREIGN KEY (domaine_etude_id) REFERENCES sirh_domaine_etude(domaine_etude_id)
);

CREATE TABLE sirh_evaluation (
    evaluation_id INTEGER PRIMARY KEY,
    satisfaction_employee_environnement INTEGER,
    note_evaluation_precedente INTEGER,
    niveau_hierarchique_poste INTEGER,
    satisfaction_employee_nature_travail INTEGER,
    satisfaction_employee_equipe INTEGER,
    satisfaction_employee_equilibre_pro_perso INTEGER,
    note_evaluation_actuelle INTEGER,
    augementation_salaire_precedente DECIMAL(4,2),
    employee_id INTEGER,
    FOREIGN KEY (employee_id) REFERENCES sirh_employee(employee_id)
);

CREATE TABLE sirh_sondage (
    sondage_id INTEGER PRIMARY KEY,
    nombre_participation_pee INTEGER,
    nb_formations_suivies INTEGER,
    nombre_employee_sous_responsabilite INTEGER,
    distance_domicile_travail INTEGER,
    annees_depuis_la_derniere_promotion INTEGER,
    annes_sous_responsable_actuel INTEGER,
    frequence_deplacement_id INTEGER,
    employee_id INTEGER,
    FOREIGN KEY (employee_id) REFERENCES sirh_employee(employee_id),
    FOREIGN KEY (frequence_deplacement_id) REFERENCES sirh_frequence_deplacement(frequence_deplacement_id)
);
