from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class DataPreprocessing:
    """Class to prepare dataset to apply machine learning algorithms"""

    def __init__(self, dataset):
        self.df = dataset
        self.percentile = 0.02

    def data_scrubbing(self, max_filter=False, min_filter=False, max_threshold=1, min_threshold=0,
                       columns_to_remove='', concept1='', concept2='', encodings=''):
        """Scrub data from input dataset by removing the introduced columns, duplicates, empty and wrong values,
        and apply one hot encoding for categorical features"""
        self.remove_columns(columns_to_remove)
        self.encodings(encodings)
        self.remove_duplicates()
        self.remove_outliers(max_filter, min_filter, max_threshold, min_threshold)
        self.remove_empty_rows()
        self.remove_wrong_rows(concept1, concept2)
        return self.df

    def remove_columns(self, columns_to_remove):
        """Remove non-meaningful columns"""
        if columns_to_remove:
            self.df.drop(columns_to_remove, axis=1, inplace=True)
        print("Scrubbed data after eliminating non-meaningful columns type: {} and shape: {}".format(type(self.df),
                                                                                                     self.df.shape))

    def encodings(self, encodings):
        """One hot encoding"""
        if encodings:
            for encoding in encodings:
                self.df[encoding] = self.df[encoding].astype(str)
            self.df = pd.get_dummies(self.df, columns=encodings)
        print("Scrubbed data after one hot encoding type: {} and shape: {}".format(type(self.df), self.df.shape))

    def remove_duplicates(self):
        """Remove duplicates"""
        self.df.drop_duplicates(keep='first', inplace=True)
        print("Scrubbed data after eliminating duplicates type: {} and shape: {}".format(type(self.df), self.df.shape))

    def remove_outliers(self, max_filter, min_filter, max_threshold, min_threshold):
        """Remove outliers"""
        df_qmin = self.df.quantile(self.percentile)
        df_qmax = self.df.quantile(1 - self.percentile)
        for i in range(len(self.df.keys())):
            if min(self.df.iloc[:, i]) >= min_threshold and max(self.df.iloc[:, i]) <= max_threshold:
                continue
            else:
                if max_filter:
                    self.df = self.df.loc[self.df[self.df.keys()[i]] <= df_qmax[i], :]
                if min_filter:
                    self.df = self.df.loc[self.df[self.df.keys()[i]] >= df_qmin[i], :]
        print("Scrubbed data after eliminating outliers type: {} and shape: {}".format(type(self.df), self.df.shape))

    def remove_empty_rows(self):
        """Remove empty rows"""
        self.df.replace('', np.nan, inplace=True)
        self.df.dropna(axis=0, how='any', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("Scrubbed data after eliminating empty datasets type: {} and shape: {}".format(type(self.df),
                                                                                             self.df.shape))

    def remove_wrong_rows(self, concept1, concept2):
        """Remove wrong rows if concept1 is higher than concept2"""
        if concept1 and concept2:
            index_to_drop = []
            for i in range(self.df.shape[0]):
                if self.df.iloc[i, self.df.columns.get_loc(
                        concept1)] > self.df.iloc[i, self.df.columns.get_loc(concept2)]:
                    index_to_drop.append(i)
            self.df.drop(index_to_drop, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        print("Scrubbed data after eliminating non-consistent datasets type: {} and shape: {}\n".format(type(self.df),
                                                                                                        self.df.shape))

    def data_scaling(self, algorithm):
        """Scaling data to normalization or standardization"""
        if algorithm.lower() == 'norm':
            scaler = MinMaxScaler()
        elif algorithm.lower() == 'standard':
            scaler = StandardScaler()
        else:
            print('Wrong scaling algorithm')
            return None
        scaler.fit(self.df)
        df_scaled = scaler.transform(self.df)
        print("Dataset scaled type: {} and shape: {}\n".format(type(df_scaled), df_scaled.shape))
        return df_scaled
