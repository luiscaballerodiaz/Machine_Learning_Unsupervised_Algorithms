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

datacsv.plot_tuning(nplots=2, algorithm=['KMeans', 'Agglomerative'], inertia=[kmeans_inertia, agg_inertia],
                    silhouette=[kmeans_silhouette, agg_silhouette])

kmeans_assignment = datacsv.apply_clustering('KMeans', n_clusters=3)
agg_assignment = datacsv.apply_clustering('Agglomerative', n_clusters=3)
