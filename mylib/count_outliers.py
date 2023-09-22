import pandas as pd
from scipy import stats

def count_outliers(billets, seuil=1.96):
    """
    Trouve, compte et affiche les outliers dans un DataFrame pour toutes les colonnes.

    Args:
    - billets (DataFrame): Le DataFrame contenant les données.
    - seuil (float): Le seuil pour les scores Z (par défaut 1.96 pour 2 écarts-types).
    """
    num_columns = [col for col in billets.columns if billets[col].dtypes in (int, float)]
    z_scores = stats.zscore(billets[num_columns])

    outliers_sup = billets[num_columns][(z_scores > seuil).any(axis=1)]
    outliers_inf = billets[num_columns][(z_scores < -seuil).any(axis=1)]
    outliers_sup2 = billets[num_columns][(z_scores > seuil).all(axis=1)]
    outliers_inf2 = billets[num_columns][(z_scores < -seuil).all(axis=1)]
    outliers_tot = billets[num_columns][(z_scores > seuil).any(axis=1) | (z_scores < -seuil).any(axis=1)]
    outliers_tot2 = billets[num_columns][(z_scores > seuil).any(axis=1) & (z_scores < -seuil).any(axis=1)]
    outliers_tot_all_col = billets[num_columns][(z_scores > seuil).all(axis=1) | (z_scores < -seuil).all(axis=1)]

    count_sup = len(outliers_sup)
    count_inf = len(outliers_inf)
    count_sup2 = len(outliers_sup2)
    count_inf2 = len(outliers_inf2)
    count_tot = len(outliers_tot)
    count_tot2 = len(outliers_tot2)
    count_tot_all_col = len(outliers_tot_all_col)

    print("\nNombre d'outliers positifs :", count_sup)
    print("Nombre d'outliers négatifs :", count_inf)
    print("\nNombre d'observations ayant des outliers positifs dans chaque variable :", count_sup2)
    print("Nombre d'observations ayant des outliers négatifs dans chaque variable :", count_inf2)
    print("\nNombre total d'observations avec au moins une variable contenant des outliers positifs et/ou négatifs :", count_tot)
    print("\nNombre total d'observations avec au moins une variable contenant des outliers positifs et négatifs :", count_tot2)
    print("Nombre total d'observations ayant des outliers positifs et/ou négatifs dans chaque variable :", count_tot_all_col)
    print(outliers_tot2)
    print(outliers_tot2.index)
  
# Appeler la fonction pour trouver, compter et afficher les outliers
#count_outliers(billets, seuil=1.96)
