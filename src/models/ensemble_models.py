#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       ensemble_models.py
Description:    This script contains class object for ensemble model based on
                social identities of patients and ensemble model using 
                KMeans clusters
Author:         Diandra Prioleau Ojo
Date Created:   December 14, 2022
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

# sampling technique 
from imblearn.over_sampling import SMOTE

# import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# import post-hoc calibration algorithms 
from sklearn.isotonic import IsotonicRegression

from sklearn.calibration import CalibratedClassifierCV

class sociodemo_ensemble_model(BaseEstimator,ClassifierMixin):
    def __init__(self):
        self.female = RandomForestClassifier(n_estimators=300,criterion='entropy',max_features=4,random_state=12)
        self.male = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=8,random_state=12)
        self.hispanic = RandomForestClassifier(n_estimators=300,criterion='gini',max_features=6,random_state=12)
        self.black = RandomForestClassifier(n_estimators=300,criterion='gini',max_features=6,random_state=12)
        self.white = GradientBoostingClassifier(n_estimators=100,min_samples_split=0.5,min_samples_leaf=0.1,max_features=8,random_state=12)
        self.age17_64 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=10,random_state=12)
        self.age64_91 = LogisticRegression(max_iter=100,penalty='l2',solver='liblinear',random_state=12)
  
        
    def fit(self, X_subsets,columns,label=None,oversample=False):
        if label == None:
            TypeError('Class label not given.')
            
        skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
            
        for key, X in X_subsets.items():
            X_train = X[columns].to_numpy()
            y_train = X[label].to_numpy()
            
            if oversample:
                smote = SMOTE(sampling_strategy=1.0,random_state=12)
                X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
                
            if key == 'F':
                    
                clf = self.female
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.female = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.female = cclf.fit(X_train,y_train)
                    
            elif key == 'M':
                clf = self.male
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.male = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.male = cclf.fit(X_train,y_train)
                    
            elif key == 'HISPANIC/LATINO':
                clf = self.hispanic
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.hispanic = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.hispanic = cclf.fit(X_train,y_train)
                    
            elif key == 'BLACK/AFRICAN AMERICAN':
                clf = self.black
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.black = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.black = cclf.fit(X_train,y_train)
                    
            elif key == 'WHITE':
                clf = self.white
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.white = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.white = cclf.fit(X_train,y_train)
                    
            elif key == '(17, 64]':
                clf = self.age17_64
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.age17_64 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.age17_64 = cclf.fit(X_train,y_train)
                    
            elif key == '(64, 91]':
                clf = self.age64_91
                cclf = clf
                #cclf = CalibratedClassifierCV(base_estimator=clf,
                #                  method='isotonic',
                #                  cv=skf)
                if oversample:
                    self.age64_91 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.age64_91 = cclf.fit(X_train,y_train)
                    
            else:
                TypeError('Model for demographic group is not available.')
    
        
    def predict_proba(self, X,columns,y=None):
        def predict_based_on_gender(X,columns,y=None):
            predictions = []
            for ind in X.index:
                x = X.iloc[ind]
                if x.gender == 'F':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.female.predict_proba(x)[:,1]
                    thresh = 0.145125
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.gender == 'M':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.male.predict_proba(x)[:,1]
                    thresh = 0.017256
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Gender not Female or Male.')
            return predictions
        
        def predict_based_on_ethnicity(X,columns,y=None):
            predictions = []
            for ind in X.index:
                x = X.iloc[ind]
                if x.ethnicity == 'HISPANIC/LATINO':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.hispanic.predict_proba(x)[:,1]
                    thresh = 0.018207
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.ethnicity == 'BLACK/AFRICAN AMERICAN':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.black.predict_proba(x)[:,1]
                    thresh = 0.049823
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.ethnicity == 'WHITE':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.white.predict_proba(x)[:,1]
                    thresh = 0.174159
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Ethnicity not represented.')
            return predictions
        
        def predict_based_on_age(X,columns,y=None):
            predictions = []
            for ind in X.index:
                x = X.iloc[ind]
                if x.age_binned == '(17, 64]':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.age17_64.predict_proba(x)[:,1]
                    thresh = 0.041449
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.age_binned == '(64, 91]':
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.age64_91.predict_proba(x)[:,1]
                    thresh = 0.451766
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Age group not represented.')
            return predictions
        
        
        predictions = pd.DataFrame()
        
        predictions[['predict_proba_based_on_gender','predict_proba_based_on_gender_binary']] = predict_based_on_gender(X,columns)
        predictions[['predict_proba_based_on_ethnicity','predict_proba_based_on_ethnicity_binary']] = predict_based_on_ethnicity(X,columns)
        predictions[['predict_proba_based_on_age','predict_proba_based_on_age_binary']] = predict_based_on_age(X,columns)
        
        predictions['predict_proba_averaged'] = predictions[['predict_proba_based_on_gender',
                                                             'predict_proba_based_on_ethnicity',
                                                             'predict_proba_based_on_age']].mean(axis=1)
        

        predictions['predict_binary_majority'] = predictions[['predict_proba_based_on_gender_binary',
                                                             'predict_proba_based_on_ethnicity_binary',
                                                             'predict_proba_based_on_age_binary']].mode(axis=1)[0]
        return predictions
        

class kmeans_ensemble_model(BaseEstimator,ClassifierMixin):
    def __init__(self):
        self.cluster1of2_12 = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=10,random_state=12)
        self.cluster2of2_12 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=8,random_state=12)
        self.cluster1of2_24 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=12,random_state=12)
        self.cluster2of2_24 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=8,random_state=12)
        self.cluster1of3 = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=4,random_state=12)
        self.cluster2of3 = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=10,random_state=12)
        self.cluster3of3 = RandomForestClassifier(n_estimators=400,criterion='entropy',max_features=2,random_state=12)
                
    def fit(self, X_subsets,columns,label=None,oversample=False):
        if label == None:
            TypeError('Class label not given.')
        
        skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
        
        for key, X in X_subsets.items():
            X_train = X[columns].to_numpy()
                
            y_train = X[label].to_numpy()
            
            if oversample:
                smote = SMOTE(sampling_strategy=1.0,random_state=12)
                X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)
                
            if key == 'cluster1of2_12':
                clf = self.cluster1of2_12 
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.cluster1of2_12 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster1of2_12 = cclf.fit(X_train,y_train)
            elif key == 'cluster2of2_12':
                clf = self.cluster2of2_12
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.cluster2of2_12 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster2of2_12 = cclf.fit(X_train,y_train)
            elif key == 'cluster1of2_24':
                clf = self.cluster1of2_24
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.cluster1of2_24 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster1of2_24 = cclf.fit(X_train,y_train)
            elif key == 'cluster2of2_24':
                clf = self.cluster2of2_24
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.cluster2of2_24 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster2of2_24 = cclf.fit(X_train,y_train)
            elif key == 'cluster1of3':
                clf = self.cluster1of3
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='sigmoid',
                                  cv=skf)
                if oversample:
                    self.cluster1of3 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster1of3 = cclf.fit(X_train,y_train)
            elif key == 'cluster2of3':
                clf = self.cluster2of3
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.cluster2of3 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster2of3 = cclf.fit(X_train,y_train)
            elif key == 'cluster3of3':
                clf = self.cluster3of3
                cclf = CalibratedClassifierCV(base_estimator=clf,
                                  method='isotonic',
                                  cv=skf)
                if oversample:
                    self.cluster3of3 = cclf.fit(X_train_smote,y_train_smote)
                else:
                    self.cluster3of3 = cclf.fit(X_train,y_train)
            else:
                TypeError('Model for cluster group is not available.')
    
    
        
    def predict_proba(self, X,columns,label,recalibrate=True,y=None):
        def predict_based_on_gender(X,columns,y=None):
            predictions = []
            size = len(columns)
            
            filename = 'models/kmeans_numclusters_2_12_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))
            X['cluster_label'] = loaded_model.predict(X[columns])
            
            for ind in X.index:
                x = X.iloc[ind]
                if x.cluster_label == 0:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster1of2_12.predict_proba(x)[:,1]
                    thresh = 0.116056
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.cluster_label == 1:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster2of2_12.predict_proba(x)[:,1]
                    thresh = 0.031127
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Gender not Female or Male.')
            return predictions
        
        def predict_based_on_ethnicity(X,columns,y=None):
            predictions = []
            
            filename = 'models/kmeans_numclusters_3_36_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))
            X['cluster_label'] = loaded_model.predict(X[columns])
            
            for ind in X.index:
                x = X.iloc[ind]
                if x.cluster_label == 0:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster1of3.predict_proba(x)[:,1]
                    thresh = 0.038972
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.cluster_label == 1:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster2of3.predict_proba(x)[:,1]
                    thresh = 0.031861
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.cluster_label == 2:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster3of3.predict_proba(x)[:,1]
                    thresh = 0.120326
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Ethnicity not represented.')
            return predictions
        
        def predict_based_on_age(X,columns,y=None):
            predictions = []
            
            filename = 'models/kmeans_numclusters_2_24_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))
            X['cluster_label'] = loaded_model.predict(X[columns])
            
            for ind in X.index:
                x = X.iloc[ind]
                if x.cluster_label == 0:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster1of2_24.predict_proba(x)[:,1]
                    thresh = 0.140491
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                elif x.cluster_label == 1:
                    x = np.array(x[columns]).reshape(1,-1)
                    prediction = self.cluster2of2_24.predict_proba(x)[:,1]
                    thresh = 0.013412
                    pred_binary = prediction >= thresh
                    predictions.append([prediction,pred_binary])
                else:
                    TypeError('Age group not represented.')
            return predictions
        
        
        predictions = pd.DataFrame()
        
        predictions[['predict_proba_based_on_clusters2_12','predict_proba_based_on_clusters2_12_binary']] = predict_based_on_gender(X,columns)
        predictions[['predict_proba_based_on_clusters3','predict_proba_based_on_clusters3_binary']] = predict_based_on_ethnicity(X,columns)
        predictions[['predict_proba_based_on_clusters2_24','predict_proba_based_on_clusters2_24_binary']] = predict_based_on_age(X,columns)
        
                
        predictions['predict_proba_averaged'] = predictions[['predict_proba_based_on_clusters2_12',
                                                             'predict_proba_based_on_clusters3',
                                                             'predict_proba_based_on_clusters2_24']].mean(axis=1)
        

        predictions['predict_binary_majority'] = predictions[['predict_proba_based_on_clusters2_12_binary',
                                                             'predict_proba_based_on_clusters3_binary',
                                                             'predict_proba_based_on_clusters2_24_binary']].mode(axis=1)[0]

        return predictions
        
        
                
        