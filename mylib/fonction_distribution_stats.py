import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import math

def plot_distribution_stats(data, figsize=(20, 10)):
    num_cols = len(data.columns)
    num_rows = math.ceil(num_cols / 2)

    f, a = plt.subplots(2, num_rows, figsize=figsize)
    
    # Flatten the axis array
    a = a.ravel()

    for i, col in enumerate(data.columns):
        sns.distplot(data[col], ax=a[i], kde=True)

        #===================================
        # Calcul de Skewness, Kurtosis et Shapiro-Wilk
        skewness = data[col].skew()
        kurtosis = data[col].kurtosis()
        _, p_shapiro = shapiro(data[col])

        #===================================
        # Affichage des résultats dans le titre
        a[i].set_title('Skew: {:.2f}\nKurtosis: {:.2f}\nShapiro p-value: {:.3f}\n'.format(skewness, kurtosis, p_shapiro), fontsize=13)

        #===================================
        # Ajouter une droite pour la médiane
        median = data[col].median()
        a[i].axvline(median, color='red', linestyle='dashed', linewidth=1, label='Median')

        # Ajouter une droite pour la moyenne
        mean = data[col].mean()
        a[i].axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')

        # Légende pour les droites
        a[i].legend()

        #===================================
        # Augmenter la taille des titres des axes
        a[i].set_xlabel(col, fontsize=12)
        a[i].set_ylabel('Density', fontsize=12)
    
        #===================================
    
    # Supprimer le dernier sous-graphique vide
    for i in range(num_cols, len(a)):
        f.delaxes(a[i])
    
    # Ajouter un espacement entre les graphiques
    plt.tight_layout(w_pad=2, h_pad=4)
    
    plt.show()
    
    
#====================================================

def plot_distribution_comparison(distri_avant, distri_après, column_name):
    
    # Créez une figure avec deux sous-graphiques côte à côte
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    if distri_avant[column_name].isnull().any():
        # Calcul de skewness, kurtosis et Shapiro-Wilk pour la colonne avant régression
        skewness_before = distri_avant[column_name].skew()
        kurtosis_before = distri_avant[column_name].kurtosis()
        info_text_before = f'Skew: {skewness_before:.2f}\nKurtosis: {kurtosis_before:.2f}\nShapiro p-value: nan'
    else:
        # Calcul de skewness, kurtosis et Shapiro-Wilk pour la colonne avant régression
        skewness_before = distri_avant[column_name].skew()
        kurtosis_before = distri_avant[column_name].kurtosis()
        _, p_shapiro_before = shapiro(distri_avant[column_name])
        info_text_before = f'Skew: {skewness_before:.2f}\nKurtosis: {kurtosis_before:.2f}\nShapiro p-value: {p_shapiro_before:.3f}'

    if distri_après[column_name].isnull().any():
        # Calcul de skewness, kurtosis et Shapiro-Wilk pour la colonne après régression
        skewness_after = distri_après[column_name].skew()
        kurtosis_after = distri_après[column_name].kurtosis()
        info_text_after = f'Skew: {skewness_after:.2f}\nKurtosis: {kurtosis_after:.2f}\nShapiro p-value: nan'
    else:
        # Calcul de skewness, kurtosis et Shapiro-Wilk pour la colonne après régression
        skewness_after = distri_après[column_name].skew()
        kurtosis_after = distri_après[column_name].kurtosis()
        _, p_shapiro_after = shapiro(distri_après[column_name])
        info_text_after = f'Skew: {skewness_after:.2f}\nKurtosis: {kurtosis_after:.2f}\nShapiro p-value: {p_shapiro_after:.3f}'

    # Distribution de la colonne avant régression linéaire
    sns.histplot(distri_avant[column_name], bins=20, kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution de {column_name} avant imputation des NaNs')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Count')

    # Affichage des mesures dans le premier sous-graphique
    axes[0].text(0.63, 0.65, info_text_before, transform=axes[0].transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Ajouter une droite pour la médiane et moyenne dans le premier sous-graphique
    median_before = distri_avant[column_name].median()
    axes[0].axvline(median_before, color='red', linestyle='dashed', linewidth=1, label='Médiane')
    mean_before = distri_avant[column_name].mean()
    axes[0].axvline(mean_before, color='green', linestyle='dashed', linewidth=1, label='Moyenne')

    # Légende pour les droites dans le premier sous-graphique
    axes[0].legend()

    # Distribution de la colonne après régression linéaire
    sns.histplot(distri_après[column_name], bins=20, kde=True, ax=axes[1])
    axes[1].set_title(f'Distribution de {column_name} après imputation des NaNs')
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Count')

    # Affichage des mesures dans le deuxième sous-graphique
    axes[1].text(0.63, 0.65, info_text_after, transform=axes[1].transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Ajouter une droite pour la médiane et moyenne dans le deuxième sous-graphique
    median_after = distri_après[column_name].median()
    axes[1].axvline(median_after, color='red', linestyle='dashed', linewidth=1, label='Médiane')
    mean_after = distri_après[column_name].mean()
    axes[1].axvline(mean_after, color='green', linestyle='dashed', linewidth=1, label='Moyenne')

    # Légende pour les droites dans le deuxième sous-graphique
    axes[1].legend()

    # Afficher les graphiques côte à côte
    plt.tight_layout()
    plt.show()
    
