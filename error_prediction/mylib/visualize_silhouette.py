import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def silhouette_viz(X, range_n_clusters):
    silhouette_scores = [] # Pour stocker les scores de silhouette

    for n_clusters in range_n_clusters:
        # Crée une figure avec 1 ligne et 2 colonnes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        
        # Le 1er sous-graphique est le graphique de silhouette
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialise le regroupeur avec la valeur n_clusters et une graine génératrice aléatoire
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # Le silhouette_score donne la valeur moyenne pour tous les échantillons.
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))

        # Calcule les valeurs de silhouette pour chaque échantillon
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Agrège les valeurs de silhouette pour les échantillons appartenant au
            # cluster i, puis les trie
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Étiqueter les graphiques de silhouette avec les numéros de cluster au milieu
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Calcule le nouveau y_lower pour le prochain graphique
            y_lower = y_upper + 10

        ax1.set_title("Silhouette Plot for Various Clusters")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        # La ligne verticale pour le score de silhouette moyen de toutes les valeurs
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2ème graphique montrant les clusters réels formés
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X.iloc[:, 0],
            X.iloc[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k"
        )

        # Étiquetage des clusters
        centers = clusterer.cluster_centers_
        # Dessine des cercles blancs aux centres des clusters
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("Clustered Data Visualization")
        ax2.set_xlabel("Feature Space for 1st Feature")
        ax2.set_ylabel("Feature Space for 2nd Feature")

        plt.suptitle(
            "Silhouette Analysis for KMeans Clustering with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    # Imprime les scores de silhouette en haut de tous les graphiques
    for n_clusters, silhouette_avg in silhouette_scores:
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

    plt.show()

# Exemple d'utilisation :
# X est votre ensemble de données
# range_n_clusters est une liste des valeurs de n_clusters que vous souhaitez tester
# visualize_silhouette(X, range_n_clusters)
