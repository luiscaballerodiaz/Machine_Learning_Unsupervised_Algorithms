import pandas as pd
import numpy as np
import math
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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class DataUnclassedCSV:
    """Class to operate with a dataset in CSV format"""

    def __init__(self, name):
        self.name = name
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.percentile = 0.02
        self.max_clusters = 10
        self.df_scaled = None
        self.list_features = None

    def read_csv(self):
        """"Read csv and convert it to dataframe"""
        dataset = pd.read_csv(self.name)
        print("Full source data from CSV type: {} and shape: {}\n".format(type(dataset), dataset.shape))
        return dataset

    def unclassed_boxplot(self, dataset, plot_name, max_features_row):
        """Plot boxplot based on input dataset"""
        dfcopy = dataset.copy()
        max_vector = np.zeros([dataset.shape[1]])
        for i in range(dataset.shape[1]):
            max_vector[i] = dataset.iloc[:, i].max()
        columns = []
        for i in range(dataset.shape[1]):
            index_max = np.argmax(max_vector)
            columns.append(dataset.columns.values[index_max])
            max_vector[index_max] = 0
        dfcopy = dfcopy.reindex(columns=columns)
        dfcopy.replace(np.nan, 0, inplace=True)
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / max_features_row), 1,
                                 figsize=(self.fig_width, self.fig_height))
        ax = axes.ravel()
        for i in range(len(ax)):
            ax[i].boxplot(dfcopy.iloc[:, (i * max_features_row):min(((i + 1) * max_features_row), dataset.shape[1])])
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            if ((i + 1) * max_features_row) > dataset.shape[1]:
                xrange = range(1, dataset.shape[1] - (i * max_features_row) + 1)
            else:
                xrange = range(1, max_features_row + 1)
            ax[i].set_xticks(xrange,
                             dfcopy.keys()[(i * max_features_row):min(((i + 1) * max_features_row), dataset.shape[1])],
                             rotation=10, ha='center')
            ax[i].set_ylabel('Feature magnitude', fontsize=8)
        ax[0].set_title(plot_name, fontsize=24, fontweight='bold')
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def unclassed_histogram(self, dataset, plot_name, ncolumns):
        """Plot histogram based on input dataset"""
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1,  ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        for i in range(dataset.shape[1]):
            ax[i].hist(dataset.iloc[:, i], histtype='stepfilled', bins=25, alpha=0.25, color="#0000FF", lw=0)
            ax[i].set_title(dataset.keys()[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel('Frequency', fontsize=8)
            ax[i].set_xlabel('Feature magnitude', fontsize=8)
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.clf()

    def data_scrubbing(self, dataset, max_filter=False, min_filter=False, max_threshold=1, min_threshold=0,
                       columns_to_remove='', concept1='', concept2='', encodings=''):
        """Scrub data from input dataset by removing the introduced columns, duplicates, empty and wrong values,
        and apply one hot encoding for categorical features"""
        # Remove non-meaningful columns
        if columns_to_remove:
            dataset.drop(columns_to_remove, axis=1, inplace=True)
        print("Scrubbed data after eliminating non-meaningful columns type: {} and shape: {}".format(type(dataset),
                                                                                                     dataset.shape))
        # One hot encoding
        if encodings:
            for encoding in encodings:
                dataset[encoding] = dataset[encoding].astype(str)
            dataset = pd.get_dummies(dataset, columns=encodings)
        print("Scrubbed data after one hot encoding type: {} and shape: {}".format(type(dataset), dataset.shape))
        # Remove duplicates
        dataset.drop_duplicates(keep='first', inplace=True)
        print("Scrubbed data after eliminating duplicates type: {} and shape: {}".format(type(dataset), dataset.shape))
        # Remove outliers
        df_qmin = dataset.quantile(self.percentile)
        df_qmax = dataset.quantile(1 - self.percentile)
        for i in range(len(dataset.keys())):
            if min(dataset.iloc[:, i]) >= min_threshold and max(dataset.iloc[:, i]) <= max_threshold:
                continue
            else:
                if max_filter:
                    dataset = dataset.loc[dataset[dataset.keys()[i]] <= df_qmax[i], :]
                if min_filter:
                    dataset = dataset.loc[dataset[dataset.keys()[i]] >= df_qmin[i], :]
        print("Scrubbed data after eliminating outliers type: {} and shape: {}".format(type(dataset), dataset.shape))
        # Remove empty rows
        dataset.replace('', np.nan, inplace=True)
        dataset.dropna(axis=0, how='any', inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        print("Scrubbed data after eliminating empty datasets type: {} and shape: {}".format(type(dataset),
                                                                                             dataset.shape))
        # Remove wrong rows if concept1 is higher than concept2
        if concept1 and concept2:
            index_to_drop = []
            for i in range(dataset.shape[0]):
                if dataset.iloc[i, dataset.columns.get_loc(
                        concept1)] > dataset.iloc[i, dataset.columns.get_loc(concept2)]:
                    index_to_drop.append(i)
            dataset.drop(index_to_drop, inplace=True)
            dataset.reset_index(drop=True, inplace=True)
        print("Scrubbed data after eliminating non-consistent datasets type: {} and shape: {}\n".format(type(dataset),
                                                                                                        dataset.shape))
        return dataset

    def data_scaling(self, dataset, algorithm):
        """Scaling data to normalization or standardization"""
        if algorithm.lower() == 'norm':
            scaler = MinMaxScaler()
        elif algorithm.lower() == 'standard':
            scaler = StandardScaler()
        else:
            print('Dataset NOT scaled type: {} and shape: {}\n'.format(type(dataset), dataset.shape))
            return None
        self.list_features = dataset.keys()
        scaler.fit(dataset)
        self.df_scaled = scaler.transform(dataset)
        print("Dataset scaled type: {} and shape: {}\n".format(type(self.df_scaled), self.df_scaled.shape))

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

    def plot_inertia_silhouette_tuning(self, algorithm, inertia, silhouette):
        """Plot the cluster sweep plot vs inertia to tune the optimum number of clusters"""
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        inertia = np.array(inertia)
        silhouette = np.array(silhouette)
        nplots = inertia.shape[0]
        if (len(algorithm) != nplots) or (silhouette.shape[0] != nplots):
            print('Wrong dimensionality in the input data\n')
            return None
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        plot_feat = []
        for i in range(nplots):
            plot_feat.append(ax1.plot(range(1, self.max_clusters + 1), inertia[i, :], color=colors[i % len(colors)],
                                      marker='o', markersize=10, linewidth=2, label=algorithm[i] + ' inertia'))
            plot_feat.append(ax2.plot(range(2, self.max_clusters + 1), silhouette[i, :], marker='^', markersize=10,
                                      color=colors[(i+5) % len(colors)], linewidth=2,
                                      label=algorithm[i] + ' silhouette score'))
        plt.title('Cluster sweep tuning', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Number of clusters', fontsize=14)
        ax1.set_ylabel('Inertia (marker=o)', fontsize=14)
        ax2.set_ylabel('Silhouette score (marker=^)', fontsize=14)
        la = []
        for i in range(nplots * 2):
            la += plot_feat[i]
        lb = [la[0].get_label(), la[1].get_label(), la[2].get_label(), la[3].get_label()]
        ax1.legend(la, lb, loc='upper center')
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        plt.savefig('Cluster sweep tuning.png', bbox_inches='tight')
        plt.clf()

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

    def plot_cluster_features(self, algorithm, dataset, cluster_class, ncolumns):
        """Plot clusters features in a bar plot"""
        n_clusters = max(cluster_class) + 1
        fig, axes = plt.subplots(math.ceil(dataset.shape[1] / ncolumns), ncolumns,
                                 figsize=(self.fig_width, self.fig_height))
        spare_axes = ncolumns - dataset.shape[1] % ncolumns
        if spare_axes == ncolumns:
            spare_axes = 0
        for axis in range(ncolumns - 1, ncolumns - 1 - spare_axes, -1):
            fig.delaxes(axes[math.ceil(dataset.shape[1] / ncolumns) - 1, axis])
        ax = axes.ravel()
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        for i in range(dataset.shape[1]):
            for cluster in range(n_clusters):
                ax[i].bar(1 + cluster * self.bar_width, dataset.iloc[cluster_class == cluster, i].mean(),
                          color=colors[cluster % len(colors)], width=self.bar_width, edgecolor='black',
                          label='cluster' + str(cluster))
            ax[i].set_title(self.list_features[i], fontsize=10, y=1.0, pad=-14, fontweight='bold')
            ax[i].grid(visible=True)
            ax[i].tick_params(axis='both', labelsize=8)
            ax[i].set_ylabel('Feature magnitude', fontsize=8)
        ax[0].legend()
        plt.savefig(algorithm + ' cluster features analysis.png', bbox_inches='tight')
        plt.clf()
