import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_outlier_piecharts(data, column, title1='', title2=''):
    # Définir les couleurs
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']  # Ajoutez plus de couleurs si nécessaire

    # Détecter automatiquement le nombre de valeurs uniques dans la colonne 'is_genuine'
    unique_values = data[column].nunique()

    # on détermine les colonnes numériques
    num = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    # On calcule le z-score sur toutes les variables numériques
    outlier_data = []
    for col in num:
        data.sort_values(by=col, inplace=True)
        z_scores = stats.zscore(data[col])
        outlier_indices = data.index[z_scores > 1.96]
        outlier_data.extend(data.loc[outlier_indices, [col, column]].values.tolist())

    # Créer le nouveau DataFrame pour les outliers
    outliers_df = pd.DataFrame(outlier_data, columns=['Variable', column])

    # Créer une figure avec deux sous-graphiques
    _, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Réglage de l'espace horizontal entre les graphiques
    plt.subplots_adjust(wspace=20)

    #======================================================
    # Piechart - répartition des billets True/False
    ax1 = axes[0]
    ax1.set_title(title1 if title1 else f"Répartition de {column}", fontsize=16, pad=20)
    labels = data[column].unique()
    sizes = data[column].value_counts(normalize=True)
    explode = tuple([0.1] + [0] * (unique_values - 1))  # Pour séparer légèrement les segments
    labels = labels[::-1]
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%\ndes billets", colors=colors[:unique_values], startangle=90, textprops={"fontsize": 14})
    for i, text in enumerate(texts):
        text.set_color(colors[i])
    for autotext in autotexts:
        autotext.set_color("white")
    ax1.axis('equal')

    #======================================================
    # Nombre d'outliers parmi les billets True/False
    outliers_counts = [len(outliers_df[outliers_df[column] == label]) for label in labels]

    # Piechart - répartition des outliers parmi les billets True/False
    ax2 = axes[1]
    ax2.set_title(title2 if title2 else f"Répartition des outliers parmi {column}", fontsize=16, pad=20)
    sizes = outliers_counts
    explode = tuple([0.1] + [0] * (unique_values - 1))  # Pour séparer légèrement les segments
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%\ndes outliers", colors=colors[:unique_values], startangle=90, textprops={"fontsize": 14})
    for i, text in enumerate(texts):
        text.set_color(colors[i])
    for autotext in autotexts:
        autotext.set_color("white")
    ax2.axis('equal')

    #======================================================
    # Ajuster la disposition
    plt.tight_layout()

    # Afficher les piecharts
    plt.show()
