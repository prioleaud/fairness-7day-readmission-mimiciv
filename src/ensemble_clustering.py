#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       ensemble_clustering.py
Description:    Fairness evaluation of ensemble model with partitions  
                based on social identities for 7-day hospital readmission 
                prediction with T2DM diabetes cohort from MIMIC-IV database 
Author:         Diandra Prioleau Ojo
Date Created:   March 1, 2023
"""

import pandas as pd 
import numpy as np
from utils.model_selection import cross_validation,parameter_tuning
from utils.data_partition import kmeans_subsets
from  scipy.stats import uniform

from hypopt import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.metrics import Youden_J_thresh, F_score_thresh,output_results

# import plots
from sklearn.calibration import calibration_curve
from utils.plots import plot_roc_curve,plot_calibration_curve,plot_pr_curve,plot_calibration_by_subgroup

#%%
# import train, validation, and test sets 
train = pd.read_csv('../data/train_data.csv')
validation = pd.read_csv('../data/validation_data.csv')
test = pd.read_csv('../data/test_data.csv')

train_df = pd.concat([train,validation])

keep_features = train_df.columns.tolist()[-94:113]
keep_features.extend(train_df.columns.tolist()[1:12])

#%%
# min-max normalization 
scaler = MinMaxScaler()
scaler.fit(train[keep_features])

# normalize train data
train_scaled = scaler.transform(train_df[keep_features])

train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=train_df.index)
train_scaled = pd.concat([train_df.loc[:,~train_df.columns.isin(keep_features)],train_scaled],axis=1)

#X = train_scaled[keep_features].to_numpy()
#y = train_scaled['label'].to_numpy()


# normalize test data 
test_scaled = scaler.transform(test[keep_features])
test_scaled = pd.DataFrame(test_scaled, columns=keep_features,index=test.index)
test_scaled = pd.concat([test.loc[:,~test.columns.isin(keep_features)],test_scaled],axis=1)

X_test = test_scaled[keep_features].to_numpy()
y_test = test_scaled['label'].to_numpy()

#%%
# create subsets using KMeans clustering
n_clusters = [2,2,3]
random_seeds = [12,24,36]
cluster_subsets = kmeans_subsets(train_df,keep_features,n_clusters,random_seeds)

seed_count = 0
for key, values in cluster_subsets.items():
    val_count = 1
    for val in values:
        val = val
        val.to_csv('../data/' + str(key) + '_subset_' + str(val_count) + '_data.csv')
        val_count += 1
    seed_count += 1
 

def run_CV(X,y,subgroup_name):

    classifiers = {
                    'KNN': KNeighborsClassifier(),
                    'RF': RandomForestClassifier(random_state=12),
                   'GNB': GaussianNB(),
                   'GB': GradientBoostingClassifier(random_state=12)}
    
    calibration_methods = ['isotonic','sigmoid']
    
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    for method in calibration_methods:
        for key in classifiers:
            estimator = CalibratedClassifierCV(classifiers[key],method=method,cv=skf)
            model = Pipeline([
                        ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
            
            
            save_results_to_filename = '../results/ensemble_clustering_cv/'+ str(subgroup_name) + '_' + str(key) + '_' + str(method) + '.xlsx'
            cross_validation(model,X,y,save_results_to_filename)
#%%
seed_count = 0
for key, values in cluster_subsets.items():

    val_count = 1
    
    for val in values:
        
        group_name = val[key + '_cluster_labels'].unique() + 1
        
        val.to_csv('../data/' + str(key) + '_' + str(random_seeds[seed_count])  + '_subset_' + str(group_name) + '_data.csv')
        
        
        y = val['label']
        X = val.drop(columns=['label'])
        
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,
                                                    stratify=y,
                                                    random_state=12)
        
        X_train = X_train[keep_features].to_numpy()
        y_train = y_train.to_numpy()
        
        
        run_CV(X_train,y_train,str(key) + '_' + str(group_name) )
        
        lr = LogisticRegression(random_state=12)
        model = Pipeline([
                            ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                            ('model', lr)])
        
        
        save_results_to_filename = '../results/ensemble_clustering_cv/' + str(key) + '_' + str(group_name) + '_' + 'LR.xlsx'
        cross_validation(model,X_train, y_train, save_results_to_filename)
        val_count += 1
    seed_count += 1
        
#%%


# Randomized search w/ Random Forest on validation set for  independently 

parameters = {
                '1': ['2_12','isotonic'],
                '2': ['2_12', 'sigmoid'],                
                }


for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_' + str(key) + '_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,
                                                    stratify=y,
                                                    random_state=12)
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = RandomForestClassifier(random_state=12)
    calibration_method = values[1]
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
            'model__base_estimator__n_estimators': (100,200,300,400,500),
            'model__base_estimator__criterion': ('gini', 'entropy'),
            'model__base_estimator__max_features': (2,4,6,8,10,12)
            } 
    
    save_results_to_file = '../results/ensemble_clustering_tuning/RF_' + str(subgroup_type) + '_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)

#%%

parameters = {
                '1': ['2_24', 'sigmoid'],
                '2': ['2_24', 'isotonic'],
                
                }
for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_' + str(key) + '_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,
                                                    stratify=y,
                                                    random_state=12)
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = RandomForestClassifier(random_state=12)
    calibration_method = values[1]
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
            'model__base_estimator__n_estimators': (100,200,300,400,500),
            'model__base_estimator__criterion': ('gini', 'entropy'),
            'model__base_estimator__max_features': (2,4,6,8,10,12)
            } 
    
    save_results_to_file = '../results/ensemble_clustering_tuning/RF_' + str(subgroup_type) + '_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)
#%%
parameters = {
                '1': ['3_36', 'sigmoid'],
                '2': ['3_36', 'isotonic'],
                '3': ['3_36', 'isotonic']
                
                }
for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_' + str(key) + '_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,
                                                    stratify=y,
                                                    random_state=12)
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = RandomForestClassifier(random_state=12)
    calibration_method = values[1]
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
            'model__base_estimator__n_estimators': (100,200,300,400,500),
            'model__base_estimator__criterion': ('gini', 'entropy'),
            'model__base_estimator__max_features': (2,4,6,8,10,12)
            } 
    
    save_results_to_file = '../results/ensemble_clustering_tuning/RF_' + str(subgroup_type) + '_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)
    
#%%


def return_train_test_split(subgroup_type,subgroup_name):
    filename = '../data/' + str(subgroup_type) +'_subset_' + str(subgroup_name) + '_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,
                                                    stratify=y,
                                                    random_state=12)
    
    return X_train, X_valid, y_train, y_valid
    
# determine optimal thresh for Female model
X_train, X_valid, y_train, y_valid = return_train_test_split('2_12', '1')
classifier = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=10,random_state=12)
skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# determine optimal thresh for Male model
X_train, X_valid, y_train, y_valid = return_train_test_split('2_12', '2')
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=8,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='sigmoid',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# Hispanic
X_train, X_valid, y_train, y_valid = return_train_test_split('2_24', '1')
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=12,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='sigmoid',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# Black
X_train, X_valid, y_train, y_valid = return_train_test_split('2_24', '2')
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=8,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# White
X_train, X_valid, y_train, y_valid = return_train_test_split('3_36', '1')
classifier = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=4,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='sigmoid',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# 17-64
X_train, X_valid, y_train, y_valid = return_train_test_split('3_36', '2')
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=10,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

# 64 - 91
X_train, X_valid, y_train, y_valid = return_train_test_split('3_36', '3')
classifier = RandomForestClassifier(n_estimators=400,criterion='entropy',max_features=2,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

#%%

from models.ensemble_models import kmeans_ensemble_model

# min-max normalization 
scaler = MinMaxScaler()
scaler.fit(train_df[keep_features])

# normalize train data
train_scaled = scaler.transform(train_df[keep_features])

train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=train_df.index)
train_scaled = pd.concat([train_df.loc[:,~train_df.columns.isin(keep_features)],train_scaled],axis=1)

X_train = train_scaled[keep_features].to_numpy()
y_train = train_scaled['label'].to_numpy()

# normalize test data 
test_scaled = scaler.transform(test[keep_features])
test_scaled = pd.DataFrame(test_scaled, columns=keep_features,index=test.index)
test_scaled = pd.concat([test.loc[:,~test.columns.isin(keep_features)],test_scaled],axis=1)

X_test = test_scaled[keep_features].to_numpy()
y_test = test_scaled['label'].to_numpy()

# evaluate ensemble method based on social identities with test data

true_label = 'label'
cluster_subsets = {'cluster1of2_12': cluster_subsets['2_12'][0],
                'cluster2of2_12': cluster_subsets['2_12'][1],
                'cluster1of3': cluster_subsets['3_36'][0],
                'cluster2of3': cluster_subsets['3_36'][1],
                'cluster3of3': cluster_subsets['3_36'][2],
                'cluster1of2_24': cluster_subsets['2_24'][0],
                'cluster2of2_24': cluster_subsets['2_24'][1],

                }

model = kmeans_ensemble_model()
model.fit(cluster_subsets,keep_features,true_label,oversample=True)
proba_pred = model.predict_proba(test_scaled,keep_features,true_label)

#%%

# evaluate the model
# AUC
y_test = test_scaled[true_label]
y_pred = proba_pred['predict_proba_averaged']

auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Sociodemographic-based Ensemble Model',
               'Receiver operating characteristic curve: test set',
               '../results/ensemble_clustering_test/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Sociodemographic-based Ensemble Model',
               'Precision-recall curve: test set',
               '../results/ensemble_clustering_test/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'Sociodemographic-based Ensemble Model',
                       'Calibration Plot: test set',
                       '../results/ensemble_clustering_test/calibration_curve.png')

#%%

results = []

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_clustering_test/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])

#%%

subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred.to_list(),columns=['pred_proba'])],axis=1)
thresh = J_optimal_thresh_test

ethnic_groups = test_results_df.ethnicity.unique()
    
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]
    
    result = output_results(sub_df['label'],sub_df['pred_proba'],thresh)
    subgrp_results.append([grp] + result)

gender_groups = test_results_df.gender.unique()

for grp in gender_groups:
    sub_df = test_results_df[test_results_df.gender == grp]
    
    result = output_results(sub_df['label'],sub_df['pred_proba'],thresh)
    subgrp_results.append([grp] + result)

age_groups = test_results_df.age_binned.unique()

for grp in age_groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]
    
    result = output_results(sub_df['label'],sub_df['pred_proba'],thresh)
    subgrp_results.append([grp] + result)

groups = test_results_df.groupby(['ethnicity','gender'])

for name, subgroup in groups:
     
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)

groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)

groups = test_results_df.groupby(['gender','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)

groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)
    
subgrp_results_df = pd.DataFrame(subgrp_results,
                                 columns=['Subgroup','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])
subgrp_results_df.to_excel('../results/ensemble_clustering_test/subgroup_results.xlsx')

#%%
# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/ensemble_clustering_test/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/ensemble_clustering_test/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/ensemble_clustering_test/calibration_plot_age.png')


# plot calibration curve by ethnicity
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/ensemble_clustering_test/calibration_plot_ethnicity_gender.png')



