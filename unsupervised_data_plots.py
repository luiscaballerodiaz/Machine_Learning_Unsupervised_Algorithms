import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy as np


class NoClassDataVisualization:
    """Class to create plots from unclassified input data"""

    def __init__(self, list_features):
        self.fig_width = 20
        self.fig_height = 10
        self.bar_width = 0.25
        self.percentile = 0.02
        self.list_features = list_features

    def boxplot(self, dataset, plot_name, max_features_row):
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

    def histogram(self, dataset, plot_name, ncolumns):
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

    def plot_inertia_silhouette_tuning(self, algorithm, inertia, silhouette):
        """Plot the cluster sweep plot vs inertia to tune the optimum number of clusters"""
        fig, ax1 = plt.subplots(figsize=(self.fig_width, self.fig_height))
        ax2 = ax1.twinx()
        inertia = np.array(inertia)
        silhouette = np.array(silhouette)
        nplots = inertia.shape[0]
        max_clusters = inertia.shape[1]
        if (len(algorithm) != nplots) or (silhouette.shape[0] != nplots):
            print('Wrong dimensionality in the input data\n')
            return None
        cmap = cm.get_cmap('tab10')
        colors = cmap.colors
        plot_feat = []
        for i in range(nplots):
            plot_feat.append(ax1.plot(range(1, max_clusters + 1), inertia[i, :], color=colors[i % len(colors)],
                                      marker='o', markersize=10, linewidth=2, label=algorithm[i] + ' inertia'))
            plot_feat.append(ax2.plot(range(2, max_clusters + 1), silhouette[i, :], marker='^', markersize=10,
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
