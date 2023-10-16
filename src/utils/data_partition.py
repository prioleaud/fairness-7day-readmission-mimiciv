#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: data_partition.py
"""
# import packages 
import pandas as pd 
from sklearn.cluster import KMeans
import pickle

# functions 
def demo_subsets(df,demographics={}):
    subsets = {}
    for key, values in demographics.items():
        subset = []
        for val in values:
            sub = df[df[key] == val]
            subset.append(sub)
        subset_dict = {key: subset}
        subsets.update(subset_dict)
    return subsets

def kmeans_subsets(df,columns,num_clusters=[],rnd_seeds=[]):
    
    df_saved = df.copy()
    df = df[columns]
    
    subsets = {}
    for idx in range(len(num_clusters)):
        df_copy = df_saved.copy()
        
        km = KMeans(n_clusters=num_clusters[idx],random_state=rnd_seeds[idx])
        km.fit(df)
        
        filename = 'models/kmeans_numclusters_' + str(num_clusters[idx]) + '_' + str(rnd_seeds[idx]) + '_model.pickle'
        pickle.dump(km, open(filename, 'wb'))
        
        labels = km.predict(df)
        cluster_label = str(num_clusters[idx]) + '_' + str(rnd_seeds[idx]) + '_cluster_labels'
        df_copy[cluster_label] = labels
        clusters = df_copy[cluster_label].unique().tolist()
        
        
        subset = []
        for cluster in clusters:
            sub = df_copy[df_copy[cluster_label] == cluster]
            subset.append(sub)
        subset_dict = {str(num_clusters[idx]) + '_' + str(rnd_seeds[idx]): subset}   
        subsets.update(subset_dict)
    return subsets
