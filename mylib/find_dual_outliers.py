import pandas as pd
from scipy import stats
import numpy as np

def find_dual_outliers(billets, seuil=1.96):
    
    """
    Trouve et affiche les observations ayant à la fois des outliers positifs et négatifs, ainsi que les variables correspondantes.

    Args:
    - billets (DataFrame): Le DataFrame contenant les données.
    - seuil (float): Le seuil pour les scores Z (par défaut 1.96 pour 2 écarts-types).
    """
    
    bold = "\033[1m"
    end = "\033[0;0m"
    
    num_columns = [col for col in billets.columns if billets[col].dtypes in [np.number]]
    z_scores = stats.zscore(billets[num_columns])

    outliers_sup = billets[num_columns][(z_scores > seuil)]
    outliers_inf = billets[num_columns][(z_scores < -seuil)]

    # Trouver les observations avec à la fois des outliers positifs et négatifs
    dual_outliers_indices = billets.index[(z_scores > seuil).any(axis=1) & (z_scores < -seuil).any(axis=1)]

    # Ajouter une nouvelle colonne pour stocker les numéros d'index de base
    billets['Original_Index'] = billets.index

    # Réinitialiser l'index pour pouvoir faire le groupby tout en conservant les numéros d'index de base
    billets.reset_index(drop=True, inplace=True)

    # Créer un DataFrame pour afficher les résultats
    results_df = pd.DataFrame(columns=['Original_Index', 'Variable', 'Outlier positif', 'Outlier négatif'])

    for idx in dual_outliers_indices:
        dual_outliers_vars = list(set(outliers_sup.columns) & set(outliers_inf.columns))
        for var in dual_outliers_vars:
            outlier_pos = outliers_sup.at[idx, var] if var in outliers_sup.columns else ''
            outlier_neg = outliers_inf.at[idx, var] if var in outliers_inf.columns else ''
            
            # Ajouter les résultats au DataFrame
            results_df = results_df.append({'Original_Index': billets.at[idx, 'Original_Index'], 'Variable': var, 'Outlier positif': outlier_pos, 'Outlier négatif': outlier_neg}, ignore_index=True)
            results_df = results_df.fillna('')
    # Grouper par l'index original
    grouped_results = results_df.groupby('Original_Index')

    # Afficher le DataFrame des résultats
    for name, group in grouped_results:
        print(bold, "\nObservation #", name, end)
        #group.reset_index(drop=True, inplace=True)
        print(group.drop(columns='Original_Index').to_string(index=False))
        

# Appeler la fonction pour trouver et afficher les observations avec des outliers positifs et négatifs
find_dual_outliers(billets, seuil=1.96)
