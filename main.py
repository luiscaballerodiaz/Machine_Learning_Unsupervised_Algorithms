from unclassed_datacsv import DataUnclassedCSV


max_clusters = 20
datacsv = DataUnclassedCSV('CC GENERAL.csv')
sourcedf = datacsv.read_csv()

datacsv.unclassed_boxplot(dataset=sourcedf.iloc[:, 1:], plot_name='Original boxplot', max_features_row=9)
datacsv.unclassed_histogram(dataset=sourcedf.iloc[:, 1:], plot_name='Original histogram', ncolumns=3)

df, df_scaled = datacsv.data_scrubbing(dataset=sourcedf, columns_to_remove='CUST_ID', max_filter=True,
                                       max_threshold=100, algorithm='standard')

datacsv.unclassed_boxplot(dataset=df, plot_name='Scrubbed boxplot', max_features_row=9)
datacsv.unclassed_histogram(dataset=df, plot_name='Scrubbed histogram', ncolumns=3)

#datacsv.kmeans_clustering_tuning(df_scaled, max_clusters=max_clusters)
datacsv.apply_kmeans_clustering(df=df_scaled, list_features=df.keys(), n_clusters=3)

#datacsv.agglomerative_clustering_tuning(df=df_scaled, max_clusters=max_clusters)
datacsv.apply_agglomerative_clustering(df=df_scaled, list_features=df.keys(), n_clusters=3)

