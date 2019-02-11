#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 21:01:48 2018

@author: ashvinsrinivasan
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from visualization import Visualisation
class Unsupervised:
    def __init__(self):
        pass
        
class KMeansclustering(Unsupervised):
    
    def __init__(self, df, clusterSizes = None):
        Unsupervised.__init__(self)
        self.df = df
        if clusterSizes == None:
            self.scores = []
            clusterSizes = [2,3,4,5,6,7,8,9,10]
            for clusterSize in clusterSizes:
                self.kmeans = KMeans(n_clusters= clusterSize)
                self.fit()
                self.labels, self.centroids, self.silhouette_avg  = self.predict()
                self.scores.append(self.silhouette_avg)
            visuals = Visualisation()
            visuals.silhouetteScores(self.scores,clusterSizes)
        else:
            self.kmeans = KMeans(n_clusters= clusterSizes)
            self.fit()
            self.labels, self.centroids, _  = self.predict()
                
            
                
    def fit(self):
        self.kmeans = self.kmeans.fit(self.df)
        
    def predict(self):
        labels = self.kmeans.predict(self.df)
        centroids = self.kmeans.cluster_centers_
        print(labels)
        print(centroids)
        silhouette_avg = silhouette_score(self.df, labels)
        print(silhouette_avg)
        return labels, centroids, silhouette_avg 
        
        