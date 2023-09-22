import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplots_bool(data, variables, titles, target_variable, figsize=(22, 16)): #18/19, 14
    # Définition des couleurs
    colors = {True: '#1F77B4', False: '#FF7F0E'}
    
    #=============================
    # Création du plot
    num_rows = len(variables) // 2 + len(variables) % 2  # Calcul du nombre de lignes pour les subplots
    fig, ax = plt.subplots(num_rows, 2, figsize=figsize)  # Création de la grille de sous-tracés

    #=============================
    # Création des boxplots avec ajout de l'indicateur de la moyenne
    for i, var in enumerate(variables):
        row, col = i // 2, i % 2
        sns.boxplot(data=data, x=var, y=target_variable, orient="h", ax=ax[row, col], palette=colors)
        ax[row, col].invert_yaxis()
        ax[row, col].set_title(titles[i], fontsize='15') # Utilisation des titres personnalisés

        group_means = data.groupby(target_variable)[var].mean()
        for j, (target_value, mean) in enumerate(group_means.items()):
            ax[row, col].plot(mean, j, 'ro', markersize=10, markeredgecolor='black') #8
            
    #=============================
    # Ajustement du layout
    plt.tight_layout(h_pad=2, w_pad=3) #h_pad=2, w_pad=4
    
    # Affichage des tracés
    plt.show()