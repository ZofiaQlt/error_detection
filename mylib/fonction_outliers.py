import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analyze_outliers(billets, hue=None):
    
    # Add formatting
    bold = "\033[1m"  
    red = '\033[91m'
    end = "\033[0;0m"  
    
    if hue:
        # Obtenir les classes uniques à partir de la colonne hue
        unique_classes = billets[hue].unique()
        # Créer un dictionnaire de palette en utilisant les classes uniques comme clés et en assignant des couleurs uniques à chaque classe
        class_palette = {class_name: sns.color_palette('colorblind', n_colors=len(unique_classes))[i] for i, class_name in enumerate(unique_classes)}
    
    total_outliers = 0  # Initialize the total outliers count

    # Determine numeric columns
    num = []
    for i in billets.columns:
        if billets[i].dtypes == int or billets[i].dtypes == float:
            num.append(i)

    for i in num:
        billets.sort_values(by=i, inplace=True)
        outlier_z = billets[(stats.zscore(billets[i]) > 1.96) | (stats.zscore(billets[i]) < -1.96)]
        nb_outlier_z = len(outlier_z)

        total_outliers += nb_outlier_z  # Add the outliers count for the current variable to the total

    print(f"{bold}{red}Total of outliers in the dataset: {total_outliers}")
    print(f"Percentage of outliers in the dataset: {round(total_outliers / len(billets) * 100, 2)} %\n{end}")
    print("=" * 120, "\n")
    
    total_outliers = pd.DataFrame()  # Initialize DataFrame to store outlier information

    for i in num:
        billets.sort_values(by=i, inplace=True)
        outlier_z = billets[(stats.zscore(billets[i]) > 1.96) | (stats.zscore(billets[i]) < -1.96)]
        outlier_z['Variable'] = i  # Add a column to identify the variable
        total_outliers = total_outliers.append(outlier_z, ignore_index=True)  # Append outlier information
    
    if hue:
        # Create a stacked histogram for outlier distribution by variable and class
        plt.figure(figsize=(12, 6))
        sns.histplot(data=total_outliers, x='Variable', hue=hue, multiple='stack', palette=class_palette)
        plt.xlabel('Variables')
        plt.ylabel("Nombre d'outliers")
        plt.title('Distribution des outliers dans le dataset par classe')
        plt.xticks(rotation=45)
        plt.legend(title=hue, labels=unique_classes)
        plt.tight_layout()
        plt.show()

        print("=" * 120, bold + '\n\nDÉTAIL DES OUTLIERS PAR VARIABLE\n' + end)
        print("=" * 120)
    else:
        # Create a stacked histogram for outlier distribution by variable and class
        plt.figure(figsize=(12, 6))
        sns.histplot(data=total_outliers, x='Variable', hue=hue, multiple='stack')
        plt.xlabel('Variables')
        plt.ylabel("Nombre d'outliers")
        plt.title('Distribution des outliers dans le dataset par classe')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("=" * 120, bold + '\n\nDÉTAIL DES OUTLIERS PAR VARIABLE\n' + end)
        print("=" * 120)

    total_outliers = {}  # Dictionary to store outlier counts by variable

    for i in num:
        billets.sort_values(by=i, inplace=True)
        outlier_z = billets[(stats.zscore(billets[i]) > 1.96) | (stats.zscore(billets[i]) < -1.96)]
        nb_outlier_z = len(outlier_z)
        percentage = round(nb_outlier_z / len(billets) * 100, 2)

        total_outliers[i] = nb_outlier_z
        
        if hue:
            if nb_outlier_z > 0:
                min_value = min(outlier_z[i])
                max_value = max(outlier_z[i])
                mean_value = np.mean(outlier_z[i])
                std_value = np.std(outlier_z[i])
               print(f"{bold}{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - "
                    f"min {min_value} - max {max_value} - mean {mean_value:.2f} - std {std_value:.2f}\n\n{end}")
  print(outlier_z[[i, hue]])
                print("-" * 64)
            else:
                min_value = None
                max_value = None
                mean_value = None
                std_value = None
                print(f"{bold}{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - min {min_value} - max {max_value} - mean {mean_value} - std {std_value}\n\n{end}")
                print(outlier_z[[i, hue]])
        else:
            if nb_outlier_z > 0:
                min_value = min(outlier_z[i])
                max_value = max(outlier_z[i])
                mean_value = np.mean(outlier_z[i])
                std_value = np.std(outlier_z[i])
                print(f"{bold}{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - min {min_value} - max {max_value} - mean {mean_value:.2f} - std {std_value:.2f}\n\n{end}")
                print(outlier_z[i])
                print("-" * 64)
            else:
                min_value = None
                max_value = None
                mean_value = None
                std_value = None
                print(f"{bold}{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - min {min_value} - max {max_value} - mean {mean_value} - std {std_value}\n\n{end}")
                print(outlier_z[i])

        if nb_outlier_z > 0:
            if hue:
                outlier_dist_by_class = outlier_z.groupby(hue)[i].describe().sort_index(ascending=False)
                print(bold + f"Distribution des outliers par classe\n\n{end}{outlier_dist_by_class}")
                print("-" * 64 + '\n')

                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                sns.histplot(data=outlier_z, x=i, hue=hue, multiple='stack', palette=class_palette, ax=axes[0])
                axes[0].set_title(f"Distribution des outliers de '{i}' par classe")
                axes[0].set_xlabel(i)
                axes[0].set_ylabel("Nombre d'outliers")
                axes[0].legend(title=hue, labels=unique_classes)

                sns.boxplot(data=outlier_z, y=i, x=hue, order=unique_classes, palette=class_palette, ax=axes[1])
                axes[1].set_title(f"Boxplots des outliers de '{i}' par classe")
                axes[1].set_xlabel(hue)
                axes[1].set_ylabel(i)

                plt.tight_layout()

                plt.show()
                print('=' * 120)
            else:
                outlier_dist_by_class = outlier_z[i].describe().sort_index(ascending=False)
                print(bold + f"Distribution des outliers par classe\n\n{end}{outlier_dist_by_class}" + end)
                print("-" * 64 + '\n')
                plt.figure(figsize=(12, 6))
                sns.histplot(data=outlier_z, x=i, hue=hue, multiple='stack')
                plt.title(f"Distribution des outliers de '{i}' par classe")
                plt.xlabel(i)
                plt.ylabel("Nombre d'outliers")
            
                plt.show()
                print('=' * 120)
                
        else:
            print('=' * 120)
 

#######################################################################################

def analyze_outliers_bool(billets):
    
    # Paramétrage du style du texte
    bold = "\033[1m"#
    red = '\033[91m'
    end = "\033[0;0m"
    
    # Couleurs à contraste élevé pour améliorer l'accessibilité des graphiques aux personnes malvoyantes
    color_true = '#1F77B4'
    color_false = '#FF7F0E'

    
    #if isinstance(billets, pd.Series):  # Si 'billets' est un DataFrame
    #billets = billets.to_frame()
    
    # on détermine les colonnes numériques
    num = []
    for i in billets.columns:
        if billets[i].dtypes == int or billets[i].dtypes == float:
            num.append(i)
            
    total_outliers = 0  # Initialize the total outliers count

    for i in num:
        #billets.sort_values(by=i, inplace=True)
        z_scores = stats.zscore(billets[i])
        outlier_z = billets[(z_scores > 1.96) | (z_scores < -1.96)]  # Check for outliers beyond 1.96 standard deviations
        nb_outlier_z = len(outlier_z)

        total_outliers += nb_outlier_z  # Add the outliers count for the current variable to the total

    print(bold + red + f"Total of outliers in the dataset: {total_outliers}" + end)
    print(bold + red + f"Percentage of outliers in the dataset: {round(total_outliers / len(billets) * 100, 2)} %\n" + end)
    print("=" * 120, "\n")

    total_outliers = pd.DataFrame()  # Initialize DataFrame to store outlier information

    for i in num:
        billets.sort_values(by=i, inplace=True)
        z_scores = stats.zscore(billets[i])
        outlier_z = billets[(z_scores > 1.96) | (z_scores < -1.96)]  # Check for outliers beyond 1.96 standard deviations
        outlier_z['Variable'] = i  # Add a column to identify the variable
        total_outliers = total_outliers.append(outlier_z, ignore_index=True)  # Append outlier information

    # Create a stacked histogram for outlier distribution by variable and class
    plt.figure(figsize=(12, 6))
    sns.histplot(data=total_outliers, x='Variable', hue='is_genuine', multiple='stack', palette={True: color_true, False: color_false})
    plt.xlabel('Variables')
    plt.ylabel("Nombre d'outliers")
    plt.title('Distribution des outliers dans le dataset par classe True/False')
    plt.xticks(rotation=45)
    plt.legend(title='Class', labels=['True', 'False'])
    plt.tight_layout()
    plt.show()

    print("=" * 120, bold + '\n\nDÉTAIL DES OUTLIERS PAR VARIABLE\n' + end)
    print("=" * 120)

    # Loop through each variable again to display individual counts
    outlier_count = {}  # Dictionary to store outlier counts by variable
    for i in num:
        billets.sort_values(by=i, inplace=True)
        z_scores = stats.zscore(billets[i])
        outlier_z = billets[(z_scores > 1.96) | (z_scores < -1.96)]  # Check for outliers beyond 1.96 standard deviations
        nb_outlier_z = len(outlier_z)
        percentage = round(nb_outlier_z / len(billets) * 100, 2)

        outlier_count[i] = nb_outlier_z
        
        #========================================

        # Affichage des outliers avec résumé statistique
        if nb_outlier_z > 0:
            min_value = min(outlier_z[i])
            max_value = max(outlier_z[i])
            mean_value = np.mean(outlier_z[i])
            std_value = np.std(outlier_z[i])
            print(bold + f"\n{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - min {min_value} - max {max_value} - mean {mean_value:.2f} - std {std_value:.2f}\n\n" + end + f"{outlier_z[[i, 'is_genuine']]} \n\n" + "-" * 64)
        else:
            min_value = None
            max_value = None
            mean_value = None
            std_value = None
            print(bold + f"\n{i} : {nb_outlier_z} outliers - {percentage} % of the dataset - min {min_value} - max {max_value} - mean {mean_value} - std {std_value}\n\n" + end + f"{outlier_z[[i, 'is_genuine']]} \n\n")

        #========================================

        # Calcul des statistiques supplémentaires
        if nb_outlier_z > 0:
            # Distribution des outliers par classe
            outlier_dist_by_class = outlier_z.groupby('is_genuine')[i].describe().sort_index(ascending=False)
            print(bold + f"Distribution des outliers par classe True/False\n\n" + end + f"{outlier_dist_by_class}")
            print("-" * 64 + '\n')

            #========================================

            # Création de la grille de sous-graphiques
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            #========================================
     
            # Distribution des outliers
            sns.histplot(data=outlier_z, x=i, hue='is_genuine', multiple='stack', palette={True: color_true, False: color_false}, ax=axes[0])
            axes[0].set_title(f"Distribution des outliers de '{i}' par classe")
            axes[0].set_xlabel(i)
            axes[0].set_ylabel('Nombre d\'outliers')
            axes[0].legend(title='Classe', labels=['True', 'False'])

            #========================================

            # Boxplots des outliers par variable
            sns.boxplot(data=outlier_z, y=i, x='is_genuine', order=[True, False], palette={True: color_true, False: color_false}, ax=axes[1])
            axes[1].set_title(f"Boxplots des outliers de '{i}' par classe")
            axes[1].set_xlabel('Classe')
            axes[1].set_ylabel(i)

            #========================================

            # Ajustement de la disposition
            plt.tight_layout()

            # Affichage des tracés
            plt.show()
            print('=' * 120)
        else:
            print('=' * 120)
            
       
            
          

