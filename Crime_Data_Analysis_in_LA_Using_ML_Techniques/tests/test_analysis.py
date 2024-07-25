# PCA ANALYSIS - 
# CLUSTERING WITH K-MEANS -
# CLASSIFICATION WITH LOGISTIC REGRESSION - 
# FORECASTING WITH PROPHET
# test_analysis.py

import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from analysis import df_combined, df_pca, X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster, X_train_sex, X_test_sex, y_train_sex, y_test_sex, classifiers

class TestAnalysis(unittest.TestCase):

    def test_scaler(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_combined)
        self.assertEqual(scaled_data.shape, df_combined.shape)
        
    def test_pca(self):
        pca = PCA(n_components=4)
        pca_data = pca.fit_transform(StandardScaler().fit_transform(df_combined))
        self.assertEqual(pca_data.shape[1], 4)
        
    def test_kmeans(self):
        kmeans = KMeans(n_clusters=6, random_state=42)
        clusters = kmeans.fit_predict(df_pca)
        self.assertEqual(len(set(clusters)), 6)

    def test_classification_report_cluster(self):
        for name, clf in classifiers.items():
            clf.fit(X_train_cluster, y_train_cluster)
            predictions = clf.predict(X_test_cluster)
            report = classification_report(y_test_cluster, predictions, output_dict=True)
            self.assertGreater(report['accuracy'], 0.5)
    
    def test_classification_report_sex(self):
        for name, clf in classifiers.items():
            clf.fit(X_train_sex, y_train_sex)
            predictions = clf.predict(X_test_sex)
            report = classification_report(y_test_sex, predictions, output_dict=True)
            self.assertGreater(report['accuracy'], 0.5)

if __name__ == '__main__':
    unittest.main()

 def test_forecast_area(self):
        df_area = crime_counts_pivot[['Hollywood']].reset_index()
        df_area.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(df_area)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        self.assertIn('yhat', forecast.columns)
