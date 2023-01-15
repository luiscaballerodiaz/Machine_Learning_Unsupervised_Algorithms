from unclassed_datacsv import DataUnclassedCSV


datacsv = DataUnclassedCSV('CC GENERAL.csv')
sourcedf = datacsv.read_csv()

datacsv.unclassed_boxplot(dataset=sourcedf.iloc[:, 1:], plot_name='Original boxplot', max_features_row=9)
datacsv.unclassed_histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Original histogram', ncolumns=3)

df = datacsv.data_scrubbing(dataset=sourcedf, columns_to_remove='CUST_ID', max_filter=True, max_threshold=100)
datacsv.data_scaling(dataset=df, algorithm='standard')

datacsv.unclassed_boxplot(dataset=df, plot_name='Scrubbed boxplot', max_features_row=9)
datacsv.unclassed_histogram(dataset=df, plot_name='Scrubbed histogram', ncolumns=3)

kmeans_inertia, kmeans_silhouette = datacsv.clustering_tuning('KMeans')
agg_inertia, agg_silhouette = datacsv.clustering_tuning('Agglomerative')
datacsv.plot_inertia_silhouette_tuning(algorithm=['KMeans', 'Agglomerative'], inertia=[kmeans_inertia, agg_inertia],
                                       silhouette=[kmeans_silhouette, agg_silhouette])

datacsv.dbscan_tuning(eps_ini=0.4, eps_end=1.8, eps_incr=0.2, min_samples_ini=df.shape[1],
                      min_samples_end=df.shape[1]*2, min_samples_incr=2)

kmeans_assignment = datacsv.apply_clustering('KMeans', n_clusters=3, n_init=10)
agg_assignment = datacsv.apply_clustering('Agglomerative', n_clusters=3, linkage='ward')
dbscan_assignment = datacsv.apply_clustering('DBSCAN', min_samples=19, eps=1.4)

datacsv.plot_cluster_features(algorithm='KMeans', dataset=df, cluster_class=kmeans_assignment, ncolumns=5)
