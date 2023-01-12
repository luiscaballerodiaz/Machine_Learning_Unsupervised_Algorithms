import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
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
                       columns_to_remove='', concept1='', concept2='', encodings='', algorithm='standard'):
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
        dataset_scaled = self.data_scaling(dataset, algorithm)
        return dataset, dataset_scaled

    @staticmethod
    def data_scaling(dataset, algorithm):
        """Scaling data to normalization or standardization"""
        if algorithm.lower() == 'norm':
            scaler = MinMaxScaler()
        elif algorithm.lower() == 'standard':
            scaler = StandardScaler()
        else:
            print('Dataset NOT scaled type: {} and shape: {}\n'.format(type(dataset), dataset.shape))
            return None
        scaler.fit(dataset)
        dataset_scaled = scaler.transform(dataset)
        print("Dataset scaled type: {} and shape: {}\n".format(type(dataset_scaled), dataset_scaled.shape))
        return dataset_scaled

    def kmeans_clustering_tuning(self, df, max_clusters=25):
        """Find the optimized number of clusters using a kmeans clustering sweep"""
        model = KMeans(n_init=10, random_state=0)
        inertia = []
        silhouette = []
        for k in range(1, max_clusters + 1):
            setattr(model, 'n_clusters', k)
            clusters_class = model.fit_predict(df)
            inertia.append(model.inertia_)
            if k > 1:
                silhouette.append(silhouette_score(df, clusters_class))
        self.plot_sweep_cluster(inertia, silhouette, max_clusters)

    def agglomerative_clustering_tuning(self, df):
        linkage_array = ward(df)
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        dendrogram(linkage_array, p=4, truncate_mode='level')
        plt.savefig('Agglomerative dendrogram.png')

    def plot_sweep_cluster(self, inertia, silhouette, max_clusters):
        """Plot the cluster sweep plot vs inertia to tune the optimum number of clusters"""
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        label1 = ax1.plot(range(1, max_clusters + 1), inertia, 'ro-', linewidth=2, label='Inertia')
        label2 = ax2.plot(range(2, max_clusters + 1), silhouette, 'b^-', linewidth=2, label='Silhouette score')
        plt.title('Cluster sweep tuning', fontsize=20, fontweight='bold')
        ax1.set_xlabel('Number of clusters', fontsize=14)
        ax1.set_ylabel('Inertia', fontsize=14, color='r')
        ax2.set_ylabel('Silhouette score', fontsize=14, color='b')
        la = label1 + label2
        lb = [la[0].get_label(), la[1].get_label()]
        ax1.legend(la, lb, loc='upper center')
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        plt.savefig('Cluster sweep tuning.png', bbox_inches='tight')
        plt.clf()

    def apply_kmeans_clustering(self, df, list_features, n_clusters=3):
        """Apply the kmeans algorithm and plot the clusters in the two PCA components"""
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        cluster_class = model.fit_predict(df)
        cluster_centers = model.cluster_centers_
        print('KMeans ClusteringAlgorithm...\n')
        print("Cluster class type: {} and shape: {}".format(type(cluster_class), cluster_class.shape))
        print("Cluster Centers type: {} and shape: {}\n".format(type(cluster_centers), cluster_centers.shape))
        self.apply_pca_plot_clusters('KMeans', 1, df, cluster_class, n_clusters, list_features,
                                     cluster_centers, ncomps=2)

    def apply_agglomerative_clustering(self, df, list_features, n_clusters=3):
        """Apply the agglomerative clustering algorithm  and plot the clusters in the two PCA components"""
        model = AgglomerativeClustering(n_clusters=3)
        cluster_class = model.fit_predict(df)
        print('Agglomerative Clustering Algorithm...\n')
        print("Cluster class type: {} and shape: {}\n".format(type(cluster_class), cluster_class.shape))
        self.apply_pca_plot_clusters('Agglomerative', 0, df, cluster_class, n_clusters, list_features, ncomps=2)

    def apply_pca_plot_clusters(self, algorithm, pca_plots, df, cluster_class, n_clusters, list_features,
                                cluster_centers=None, ncomps=2):
        """Apply PCA algorithm in the data and plot meaningful graphs"""
        pca = PCA(n_components=ncomps)
        try:
            df = df.to_numpy()
        except (Exception,):
            pass
        df_pca = pca.fit_transform(df)
        print("Data PCA type: {} and shape: {}".format(type(df_pca), df_pca.shape))
        try:
            cluster_centers_pca = pca.transform(cluster_centers)
            print("Cluster Centers PCA type: {} and shape: {}\n".format(type(cluster_centers_pca),
                                                                        cluster_centers_pca.shape))
        except (Exception,):
            cluster_centers_pca = None
            print('Cluster Centers PCA not applying\n')
        if ncomps >= 2:
            self.plot_clusters_pca(algorithm, df_pca, cluster_class, n_clusters, cluster_centers_pca)
        if pca_plots == 1:
            self.plot_pca_breakdown(list_features, pca)
            pca = PCA(n_components=df.shape[1])
            pca.fit_transform(df)
            self.plot_pca_scree(pca)

    def plot_clusters_pca(self, algorithm, df_pca, cluster_class, n_clusters, cluster_centers_pca):
        """Plot first vs second PCA component"""
        plt.subplots(figsize=(self.fig_width, self.fig_height))
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        for n in range(n_clusters):
            lab = 'points cluster' + str(n)
            labc = 'center cluster' + str(n)
            plt.scatter(df_pca[cluster_class == n, 0], df_pca[cluster_class == n, 1], s=10, marker='o', lw=0,
                        color=colors[n], label=lab)
            try:
                plt.scatter(cluster_centers_pca[n, 0], cluster_centers_pca[n, 1], s=100, marker='^',
                            edgecolor='k', lw=3, color=colors[n], label=labc)
            except (Exception,):
                pass
        plt.title(algorithm + ' ' + str(n_clusters) + ' Clusters for Credit Card Customer',
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
