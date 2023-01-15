from data_preprocessing import DataPreprocessing
from unsupervised_data_plots import NoClassDataVisualization
from unsupervised_algorithms import UnsupervisedAlgorithms
import pandas as pd


sourcedf = pd.read_csv('CC GENERAL.csv')
print("Full source data from CSV type: {} and shape: {}\n".format(type(sourcedf), sourcedf.shape))

preprocessing = DataPreprocessing(sourcedf)
df_unscaled = preprocessing.data_scrubbing(columns_to_remove='CUST_ID', max_filter=True, max_threshold=100)
list_features = df_unscaled.keys()
df_scaled = preprocessing.data_scaling(algorithm='standard')

data_analysis = NoClassDataVisualization(list_features)
data_analysis.boxplot(dataset=sourcedf.iloc[:, 1:], plot_name='Original boxplot', max_features_row=9)
data_analysis.histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Original histogram', ncolumns=3)
data_analysis.boxplot(dataset=df_unscaled, plot_name='Scrubbed boxplot', max_features_row=9)
data_analysis.histogram(dataset=df_unscaled, plot_name='Scrubbed histogram', ncolumns=3)

unsupervised = UnsupervisedAlgorithms(df_scaled, list_features)

kmeans_inertia, kmeans_silhouette = unsupervised.clustering_tuning('KMeans')
agg_inertia, agg_silhouette = unsupervised.clustering_tuning('Agglomerative')
unsupervised.dbscan_tuning(eps_ini=0.4, eps_end=1.8, eps_incr=0.2, min_samples_ini=len(list_features),
                           min_samples_end=len(list_features)*2, min_samples_incr=2)

kmeans_assignment = unsupervised.apply_clustering('KMeans', n_clusters=3, n_init=10)
agg_assignment = unsupervised.apply_clustering('Agglomerative', n_clusters=3, linkage='ward')
dbscan_assignment = unsupervised.apply_clustering('DBSCAN', min_samples=19, eps=1.4)

data_analysis.plot_inertia_silhouette_tuning(algorithm=['KMeans', 'Agglomerative'],
                                             inertia=[kmeans_inertia, agg_inertia],
                                             silhouette=[kmeans_silhouette, agg_silhouette])
data_analysis.plot_cluster_features(algorithm='KMeans', dataset=df_unscaled, cluster_class=kmeans_assignment,
                                    ncolumns=5)
