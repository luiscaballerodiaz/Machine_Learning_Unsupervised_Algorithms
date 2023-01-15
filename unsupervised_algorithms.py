import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_score
from sklearn.decomposition import PCA


class UnsupervisedAlgorithms:
    """Class to operate with a dataset in CSV format"""

    def __init__(self, df_scaled, list_features):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.max_clusters = 10
        self.df_scaled = df_scaled
        self.list_features = list_features

    def clustering_tuning(self, algorithm):
        """Find the optimized number of clusters with a clustering sweep"""
        ini_time = time.time()
        if algorithm.lower() == 'kmeans':
            model = KMeans(n_init=10, random_state=0)
        elif algorithm.lower() == 'agglomerative':
            model = AgglomerativeClustering(linkage='ward')
        else:
            print('Algorithm NOT correct')
            return None
        inertia = []
        silhouette = []
        for k in range(1, self.max_clusters + 1):
            setattr(model, 'n_clusters', k)
            cluster_class = model.fit_predict(self.df_scaled)
            total_inertia = 0
            cluster_centers = np.array([self.df_scaled[cluster_class == cluster].mean(axis=0) for cluster in range(k)])
            for i in range(self.df_scaled.shape[1]):
                for j in range(self.df_scaled.shape[0]):
                    total_inertia += (self.df_scaled[j, i] - cluster_centers[cluster_class[j], i]) ** 2
            inertia.append(total_inertia)
            if k > 1:
                silhouette.append(silhouette_score(self.df_scaled, cluster_class))
        print(algorithm + ' tuning time: ' + str(time.time() - ini_time) + ' seconds\n')
        return inertia, silhouette

    def dbscan_tuning(self, eps_ini, eps_end, eps_incr, min_samples_ini, min_samples_end, min_samples_incr):
        """Find the optimized eps and min_samples for DBSCAN algorithm"""
        ini_time = time.time()
        distances, indices = NearestNeighbors(n_neighbors=2).fit(self.df_scaled).kneighbors(self.df_scaled)
        distances = np.sort(distances[:, 1])
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.plot(distances, color='b', linewidth=2)
        plt.title('Distance to the closest datapoint', fontsize=24)
        plt.xlabel('Number of datapoints', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.grid()
        plt.savefig('DBSCAN datapoints distance.png', bbox_inches='tight')
        plt.clf()
        eps_vector = np.round(np.arange(eps_ini, eps_end + eps_incr, eps_incr), 1)
        min_samples_vector = range(min_samples_ini, min_samples_end + 1, min_samples_incr)
        silhouette_matrix = np.zeros([len(eps_vector), len(min_samples_vector)])
        clusters = np.zeros([len(eps_vector), len(min_samples_vector)])
        for eps, i in zip(eps_vector, range(len(eps_vector))):
            for min_samples, j in zip(min_samples_vector, range(len(min_samples_vector))):
                model = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_class = model.fit_predict(self.df_scaled)
                if max(cluster_class) < 1:
                    silhouette_matrix[i, j] = -1
                else:
                    silhouette_matrix[i, j] = silhouette_score(
                        self.df_scaled, cluster_class)
                clusters[i, j] = max(cluster_class) + 1
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(silhouette_matrix, cmap=plt.cm.PuBuGn)
        plt.colorbar()
        ax.set_xlabel('min_samples parameter', fontsize=14)
        ax.set_ylabel('eps parameter', fontsize=14)
        ax.set_title('DBSCAN parameter sweep by Silhouette score / Number of clusters', fontsize=24)
        ax.set_xticks(np.arange(0.5, len(min_samples_vector) + 0.5), labels=min_samples_vector, fontsize=14)
        ax.set_yticks(np.arange(0.5, len(eps_vector) + 0.5), labels=eps_vector, fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(min_samples_vector)):
            for j in range(len(eps_vector)):
                if round(silhouette_matrix[j, i], 2) == -1:
                    string = ''
                else:
                    string = str(round(silhouette_matrix[j, i], 2))
                ax.text(i + 0.5, j + 0.5, string + ' / ' + str(clusters[j, i]),
                        ha="center", va="center", color="k", fontweight='bold', fontsize=10)
        plt.savefig('DBSCAN parameter sweep.png', bbox_inches='tight')
        plt.clf()
        print('DBSCAN tuning time: ' + str(time.time() - ini_time) + ' seconds\n')

    def apply_clustering(self, algorithm, **parameters):
        """Apply the machine learning algorithm and plot the clusters in the two PCA components"""
        params = {}
        for key, value in parameters.items():
            params[key] = value
        if algorithm.lower() == 'kmeans':
            model = KMeans(n_clusters=params['n_clusters'], n_init=params['n_init'], random_state=0)
        elif algorithm.lower() == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
            self.create_dendrogram(params['n_clusters'])
        elif algorithm.lower() == 'dbscan':
            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        else:
            print('Algorithm NOT correct')
            return None
        cluster_class = model.fit_predict(self.df_scaled)
        n_clusters = max(cluster_class) + 1
        cluster_centers = np.array([self.df_scaled[cluster_class == cluster].mean(axis=0)
                                    for cluster in range(n_clusters)])
        print(algorithm + ' cluster class type: {} and shape: {}'.format(type(cluster_class), cluster_class.shape))
        print(algorithm + ' cluster centers type: {} and shape: {}\n'.format(type(cluster_centers),
                                                                             cluster_centers.shape))
        if n_clusters > 0:
            self.apply_pca_plot_clusters(algorithm, parameters, cluster_class, n_clusters, cluster_centers, ncomps=2)
        return cluster_class

    def create_dendrogram(self, n_clusters):
        """Create dendrogram plot"""
        linkage_array = ward(self.df_scaled)
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        dendrogram(linkage_array, p=5, truncate_mode='level')
        plt.title('Dendrogram for ' + str(n_clusters) + ' clusters', fontsize=20, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=14)
        plt.ylabel('Cluster Distance', fontsize=14)
        plt.grid()
        plt.savefig('Agglomerative dendrogram ' + str(n_clusters) + ' clusters.png')

    def apply_pca_plot_clusters(self, algorithm, parameters, cluster_class, n_clusters, cluster_centers, ncomps=2):
        """Apply PCA algorithm in the data and plot meaningful graphs"""
        pca = PCA(n_components=ncomps)
        try:
            self.df_scaled = self.df_scaled.to_numpy()
        except (Exception,):
            pass
        df_pca = pca.fit_transform(self.df_scaled)
        print(algorithm + " data PCA type: {} and shape: {}".format(type(df_pca), df_pca.shape))
        cluster_centers_pca = pca.transform(cluster_centers)
        print(algorithm + " cluster centers PCA type: {} and shape: {}\n".format(type(cluster_centers_pca),
                                                                                 cluster_centers_pca.shape))
        if ncomps >= 2:
            self.plot_clusters_pca(algorithm, parameters, df_pca, cluster_class, n_clusters, cluster_centers_pca)
        self.plot_pca_breakdown(self.list_features, pca)
        pca = PCA(n_components=self.df_scaled.shape[1])
        pca.fit_transform(self.df_scaled)
        self.plot_pca_scree(pca)

    def plot_clusters_pca(self, algorithm, parameters, df_pca, cluster_class, n_clusters, cluster_centers_pca):
        """Plot first vs second PCA component"""
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        if -1 in cluster_class:
            plt.scatter(df_pca[cluster_class == -1, 0], df_pca[cluster_class == -1, 1], s=10, marker='o', lw=0,
                        color='k', label='noise')
        for n in range(n_clusters):
            lab = 'points cluster' + str(n)
            labc = 'center cluster' + str(n)
            plt.scatter(df_pca[cluster_class == n, 0], df_pca[cluster_class == n, 1], s=10, marker='o', lw=0,
                        color=colors[n % len(colors)], label=lab)
            plt.scatter(cluster_centers_pca[n, 0], cluster_centers_pca[n, 1], s=100, marker='^',
                        edgecolor='k', lw=3, color=colors[n % len(colors)], label=labc)
        plt.title(algorithm + ' with params ' + str(parameters) + ' and ' + str(n_clusters) + ' clusters',
                  fontsize=20, fontweight='bold')
        plt.xlabel('First PCA', fontsize=14)
        plt.ylabel('Second PCA', fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig(algorithm + ' ' + str(n_clusters) + ' clusters plot.png', bbox_inches='tight')
        plt.clf()

    def plot_pca_breakdown(self, list_features, pca):
        """Plot the PCA breakdown per each feature"""
        _, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        plt.pcolormesh(pca.components_, cmap=plt.cm.cool)
        plt.colorbar()
        pca_yrange = [x + 0.5 for x in range(pca.components_.shape[0])]
        pca_xrange = [x + 0.5 for x in range(pca.components_.shape[1])]
        plt.xticks(pca_xrange, list_features, rotation=60, ha='center')
        ax.xaxis.tick_top()
        str_ypca = []
        for i in range(pca.components_.shape[0]):
            str_ypca.append('Component ' + str(i + 1))
        plt.yticks(pca_yrange, str_ypca)
        plt.xlabel("Feature", weight='bold', fontsize=14)
        plt.ylabel("Principal components", weight='bold', fontsize=14)
        plt.savefig('PCA breakdown.png', bbox_inches='tight')
        plt.clf()

    def plot_pca_scree(self, pca):
        """Plot the scree plot of the PCA to understand the covered variance"""
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        label1 = ax1.plot(range(1, len(pca.components_) + 1), pca.explained_variance_ratio_,
                          'ro-', linewidth=2, label='Individual PCA variance')
        label2 = ax2.plot(range(1, len(pca.components_) + 1), np.cumsum(pca.explained_variance_ratio_),
                          'b^-', linewidth=2, label='Cumulative PCA variance')
        plt.title('Scree Plot', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Principal Components', fontsize=14)
        ax1.set_ylabel('Proportion of Variance Explained', fontsize=14, color='r')
        ax2.set_ylabel('Cumulative Proportion of Variance Explained', fontsize=14, color='b')
        la = label1 + label2
        lb = [la[0].get_label(), la[1].get_label()]
        ax1.legend(la, lb, loc='upper center')
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        plt.savefig('PCA scree plot.png', bbox_inches='tight')
        plt.clf()
