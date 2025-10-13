import numpy as np

def make_sample_weights(y, pos_weight=10.0):
    """
    Retourne un vecteur de poids: pos_weight pour y==1, 1.0 pour y==0.
    Utilisable comme `sample_weight` dans .fit(...) ou dans cross_validate via fit_params.
    """
    y = np.asarray(y)
    return np.where(y == 1, pos_weight, 1.0)