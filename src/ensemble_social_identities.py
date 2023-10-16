#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       ensemble_social_identities.py
Description:    Fairness evaluation of ensemble model with partitions  
                based on social identities for 7-day hospital readmission 
                prediction with T2DM diabetes cohort from MIMIC-IV database 
Date Created:   February 25, 2023
"""

import pandas as pd 
import numpy as np
from utils.model_selection import cross_validation,parameter_tuning
from utils.data_partition import demo_subsets
from  scipy.stats import uniform

from sklearn.model_selection import train_test_split
from hypopt import GridSearch
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
from utils.plots import plot_roc_curve,plot_calibration_curve,plot_pr_curve,plot_calibration_by_subgroup,calibration_barplot
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
scaler.fit(train_df[keep_features])

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
# create subsets (i.e., clustering - KMeans vs DEMO subsets)
demographics = {'ethnicity': train_scaled.ethnicity.unique().tolist(),
                'age_binned': train_scaled.age_binned.unique().tolist(),
                'gender': train_scaled.gender.unique().tolist()}
demo_subsets = demo_subsets(train_scaled,demographics)

 
#%%
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
                        #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
            
            
            save_results_to_filename = '../results/ensemble_subgroup_cv/'+ str(subgroup_name) + '_' + str(key) + '_' + str(method) + '.xlsx'
            cross_validation(model,X,y,save_results_to_filename)

#%%

# cross validation 
for key, values in demo_subsets.items():
    val_count = 1
    for val in values:
        
        group_name = str(val[key].unique())
        
        if "/" in group_name:
            group_name = group_name.replace("/","_")
        val.to_csv('../data/' + str(key) + '_subset_' + str(group_name) + '_data.csv')
        
        
        y = val['label']
        X = val.drop(columns=['label'])
        
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)
        '''
        # min-max normalization 
        scaler = MinMaxScaler()
        scaler.fit(X_train[keep_features])

        # normalize train data
        train_scaled = scaler.transform(X_train[keep_features])
        
        train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
        train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)
        '''
        X_train = X_train[keep_features].to_numpy()
        y_train = y_train.to_numpy()
        
        
        run_CV(X_train,y_train,str(group_name))
        
        lr = LogisticRegression(random_state=12)
        model = Pipeline([
                            #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                            ('model', lr)])
        
        
        save_results_to_filename = '../results/ensemble_subgroup_cv/' + str(group_name) + '_' + 'LR.xlsx'
        cross_validation(model,X_train, y_train, save_results_to_filename)
        val_count += 1
#%%
scoring = {"AUC": "roc_auc",
           "AP":"average_precision",
           }

#%%
# Randomized search w/ Random Forest on validation set for the White, and subgroups, independently 

parameters = {
                'F': ['gender','isotonic'],
                'M': ['gender', 'sigmoid'],
                'BLACK_AFRICAN AMERICAN': ['ethnicity', 'isotonic']
                
                }

for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_[\'' + str(key) + '\']_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)

    '''
    # min-max normalization 
    scaler = MinMaxScaler()
    scaler.fit(X_train[keep_features])

    # normalize train data
    train_scaled = scaler.transform(X_train[keep_features])
    
    train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
    train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)
    
    X_train = train_scaled[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    valid_scaled = scaler.transform(X_valid[keep_features])
    valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=X_valid.index)
    valid_scaled = pd.concat([X_valid.loc[:,~X_valid.columns.isin(keep_features)],valid_scaled],axis=1)
    
    X_valid = valid_scaled[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    '''
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = RandomForestClassifier(random_state=12)
    calibration_method = values[1]
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
            'model__base_estimator__n_estimators': (100,200,300,400,500),
            'model__base_estimator__criterion': ('gini', 'entropy'),
            'model__base_estimator__max_features': (2,4,6,8,10,12)
            } 
    
    save_results_to_file = '../results/ensemble_subgroup_tuning/RF_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)

#%%

# Randomized search w/ Gradient Boosting on validation set for the White subgroup 

parameters = {
                '(64, 91]': ['age_binned','isotonic']
                }

for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_[\'' + str(key) + '\']_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)

    '''
    # min-max normalization 
    scaler = MinMaxScaler()
    scaler.fit(X_train[keep_features])

    # normalize train data
    train_scaled = scaler.transform(X_train[keep_features])
    
    train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
    train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)
    
    X_train = train_scaled[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    
    
    valid_scaled = scaler.transform(X_valid[keep_features])
    valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=X_valid.index)
    valid_scaled = pd.concat([X_valid.loc[:,~X_valid.columns.isin(keep_features)],valid_scaled],axis=1)
    
    X_valid = valid_scaled[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    '''

    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = GradientBoostingClassifier(random_state=12)
    calibration_method = values[1]
    
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
            'model__base_estimator__n_estimators': (100,200,300,400,500),
            'model__base_estimator__min_samples_split': (np.linspace(0.1, 1.0, 10, endpoint=True)),
            'model__base_estimator__min_samples_leaf': (np.linspace(0.1, 1.0, 10, endpoint=True)),
            'model__base_estimator__max_features': (2,4,6,8,10,12)
            } 
    
    save_results_to_file = '../results/ensemble_subgroup_tuning/GB_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)
#%%

# Randomized search w/ KNN on validation set for the Hispanic ethnic subgroup and 65 to 91 age group 

parameters = {
                'HISPANIC_LATINO': ['ethnicity','isotonic']
                
                }

for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_[\'' + str(key) + '\']_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)

    '''# min-max normalization 
    scaler = MinMaxScaler()
    scaler.fit(X_train[keep_features])

    # normalize train data
    train_scaled = scaler.transform(X_train[keep_features])
    
    train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
    train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)
    
    X_train = train_scaled[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    
    valid_scaled = scaler.transform(X_valid[keep_features])
    valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=X_valid.index)
    valid_scaled = pd.concat([X_valid.loc[:,~X_valid.columns.isin(keep_features)],valid_scaled],axis=1)
    
    X_valid = valid_scaled[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    '''
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()

    classifier = KNeighborsClassifier()
    calibration_method = values[1]
    
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
     
    estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
    model = Pipeline([
                        #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', estimator)])
    
    param_grid = {
        'model__base_estimator__n_neighbors': (3,4,5,6,7,8,9,10,11,12,13,14,15),
        'model__base_estimator__metric': ('minkowski', 'euclidean','manhattan'),
        'model__base_estimator__weights': ('uniform','distance')
        } 
    
    save_results_to_file = '../results/ensemble_subgroup_tuning/KNN_' + str(key) + '_' + str(calibration_method) + '.xlsx'
    #parameter_tuning(model, param_grid, X_train, y_train, X_valid, y_valid, save_results_to_file)  
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)
    

    
#%% 

# Randomized search w/ Logsitic Regression on validation set for the 65 to 91 age group 

parameters = {
                '(17, 64]': ['age_binned'], 
                'WHITE': ['ethnicity']
                }


for key, values in parameters.items():
    
    subgroup_type = values[0]
    
    filename = '../data/' + str(subgroup_type) +'_subset_[\'' + str(key) + '\']_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)

    '''
    # min-max normalization 
    scaler = MinMaxScaler()
    scaler.fit(X_train[keep_features])

    # normalize train data
    train_scaled = scaler.transform(X_train[keep_features])
    
    train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
    train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)
    
    X_train = train_scaled[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    
    
    valid_scaled = scaler.transform(X_valid[keep_features])
    valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=X_valid.index)
    valid_scaled = pd.concat([X_valid.loc[:,~X_valid.columns.isin(keep_features)],valid_scaled],axis=1)
    
    X_valid = valid_scaled[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    '''
    
    X_train = X_train[keep_features].to_numpy()
    y_train = y_train.to_numpy()
    
    X_valid = X_valid[keep_features].to_numpy()
    y_valid = y_valid.to_numpy()
    
    classifier = LogisticRegression(random_state=12)
    model = Pipeline([
                        #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                        ('model', classifier)])
    param_grid = {
            'model__penalty': ('l1','l2'),
            'model__solver': ('liblinear', 'saga'),
            'model__max_iter': (100, 200,300,400,500,600,700,800,900,1000),
            } 
    save_results_to_file = '../results/ensemble_subgroup_tuning/LR_' + str(key) + '.xlsx'
    #parameter_tuning(model, param_grid, X, y, X_valid, y_valid, save_results_to_file)
    
    gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
    gs.fit(X_train,y_train,X_valid,y_valid,scoring='roc_auc')
    
    df_params = pd.DataFrame(gs.params)
    df_scores = pd.DataFrame(gs.scores)
    
    results = pd.concat([df_params,df_scores],axis=1)
    results['optimal_thresh'] = Youden_J_thresh(y_valid,gs.predict_proba(X_valid)[:,1])
    results.to_csv(save_results_to_file)
#%%

def return_train_test_split(subgroup_type,subgroup_name):
    filename = '../data/' + str(subgroup_type) +'_subset_[\'' + str(subgroup_name) + '\']_data.csv'
    subgrp_df = pd.read_csv(filename)
    
    y = subgrp_df.label
    X = subgrp_df.drop(columns=['label'])
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)
    
    return X_train, X_valid, y_train, y_valid
#%%    
# determine optimal thresh for Female model
X_train, X_valid, y_train, y_valid = return_train_test_split('gender', 'F')
classifier = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features=2,random_state=12)
skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

#%%
# determine optimal thresh for Male model
X_train, X_valid, y_train, y_valid = return_train_test_split('gender', 'M')
classifier = RandomForestClassifier(n_estimators=200,criterion='entropy',max_features=2,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='sigmoid',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)
#%%
# Hispanic
X_train, X_valid, y_train, y_valid = return_train_test_split('ethnicity', 'HISPANIC_LATINO')
classifier = KNeighborsClassifier(metric='minkowski',n_neighbors=14,weights='uniform')
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)
#%%
# Black
X_train, X_valid, y_train, y_valid = return_train_test_split('ethnicity', 'BLACK_AFRICAN AMERICAN')
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',max_features=10,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)
#%%
# White
X_train, X_valid, y_train, y_valid = return_train_test_split('ethnicity', 'WHITE')
classifier = LogisticRegression(max_iter=100,penalty='l1',solver='liblinear',random_state=12)
#estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', classifier)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)
#%%
# 17-64
X_train, X_valid, y_train, y_valid = return_train_test_split('age_binned', '(17, 64]')
classifier = LogisticRegression(max_iter=100,penalty='l2',solver='liblinear',random_state=12)
#estimator = CalibratedClassifierCV(classifier,method='sigmoid',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', classifier)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)
#%%
# 64 - 91
X_train, X_valid, y_train, y_valid = return_train_test_split('age_binned', '(64, 91]')
classifier = GradientBoostingClassifier(n_estimators=100,min_samples_split=0.3,min_samples_leaf=0.3,max_features=4,random_state=12)
estimator = CalibratedClassifierCV(classifier,method='isotonic',cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
model.fit(X_train[keep_features],y_train)
y_pred = model.predict_proba(X_valid[keep_features])[:,1]
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

#%%
from models.ensemble_models import sociodemo_ensemble_model

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
sociodemo_subsets = {'F': train_scaled[train_scaled.gender == 'F'],
                'M': train_scaled[train_scaled.gender == 'M'],
                'WHITE': train_scaled[train_scaled.ethnicity == 'WHITE'],
                'HISPANIC/LATINO': train_scaled[train_scaled.ethnicity == 'HISPANIC/LATINO'],
                'BLACK/AFRICAN AMERICAN': train_scaled[train_scaled.ethnicity == 'BLACK/AFRICAN AMERICAN'],
                '(17, 64]': train_scaled[train_scaled.age_binned == '(17, 64]'],
                '(64, 91]': train_scaled[train_scaled.age_binned == '(64, 91]'],

                }

model = sociodemo_ensemble_model()
model.fit(sociodemo_subsets,keep_features,true_label,oversample=False)
proba_pred = model.predict_proba(test_scaled,keep_features,true_label)

#%%

# evaluate the model
# AUC
y_test = test_scaled[true_label]
y_pred = proba_pred['predict_proba_averaged'].astype('float64')

auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Sociodemographic-based Ensemble Model',
               'Receiver operating characteristic curve: test set',
               '../results/ensemble_subgroup_test/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Sociodemographic-based Ensemble Model',
               'Precision-recall curve: test set',
               '../results/ensemble_subgroup_test/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Sociodemographic-based Ensemble Model',
                       'Calibration Plot: test set',
                       '../results/ensemble_subgroup_test/calibration_curve.png')

plot_calibration_barplot(y_test,y_pred,5,'7-day Readmission Sociodemographic-based Ensemble Model',
                       '../results/ensemble_clustering_test/calibration_barplot.png')


#%%

results = []

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])

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
                                 columns=['Subgroup','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])
subgrp_results_df.to_excel('../results/ensemble_subgroup_test/subgroup_results.xlsx')

#%%
# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_age.png')


# plot calibration curve by ethnicity
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_ethnicity_gender.png')

#%%
# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Sociodemographic-based Ensemble Model',
                             filename='../results/ensemble_subgroup_test/calibration_plot_ethnicity_gender_age.png')


#%%
ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])

#%%
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/ensemble_subgroup_test/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

