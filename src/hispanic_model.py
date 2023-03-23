#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       hispanic_model.py
Description:    Fairness evaluation of 7-day hospital readmission prediction 
                with T2DM diabetes cohort of Black/African American patients
                from MIMIC-IV database
Author:         Diandra Prioleau Ojo
Date Created:   March 21,2023
"""

# import packges
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from hypopt import GridSearch
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

# function for nested cross validation 
def cross_validation(model,X,y,save_to_filename):

    scores = []
    
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
    
    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx, :], X[test_idx, :] 
        y_train, y_test = y[train_idx], y[test_idx]
        
        # fit randomized search 
        model.fit(X_train,y_train)
    
        # predict 
        y_pred = model.predict_proba(X_test)[:,1]
        
        # evaluate model
        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        
        # report progress
        print('>AUC=%.3f' % (auc))
        print('>AP=%.3f' % (ap))
        
        scores.append([auc,ap])
    # summarize the estimated performance of the model
    cv_results = pd.DataFrame(scores,columns=['test_AUC','test_AP'])
    cv_results.to_excel(save_to_filename)
    
    print('AUC: %.3f (%.3f)' % (cv_results['test_AUC'].mean(), cv_results['test_AUC'].std()))
    print('AP: %.3f (%.3f)' % (cv_results['test_AP'].mean(), cv_results['test_AP'].std()))
         
#%%

# import train, validation, and test sets 
train = pd.read_csv('../data/train_data.csv')
validation = pd.read_csv('../data/validation_data.csv')
test = pd.read_csv('../data/test_data.csv')

data = pd.concat([train,validation,test])
data = data[data['ethnicity'] == 'HISPANIC/LATINO']

y = data['label']
X = data.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['gender','age_binned']], y],axis=1),
                                                    random_state=12)

X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.3,
                                                    stratify=pd.concat([X_train[['gender','age_binned']], y_train],axis=1),
                                                    random_state=12)

keep_features = train.columns.tolist()[-94:113]
keep_features.extend(train.columns.tolist()[1:12])

#%%
# min-max normalization 
scaler = MinMaxScaler()
scaler.fit(X_train[keep_features])

# normalize train data
train_scaled = scaler.transform(X_train[keep_features])

train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=X_train.index)
train_scaled = pd.concat([X_train.loc[:,~X_train.columns.isin(keep_features)],train_scaled],axis=1)

X = train_scaled[keep_features].to_numpy()
y = y_train.to_numpy()

#%%
# normalize validation data 
valid_scaled = scaler.transform(X_valid[keep_features])
valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=X_valid.index)
valid_scaled = pd.concat([X_valid.loc[:,~X_valid.columns.isin(keep_features)],valid_scaled],axis=1)

X_valid = valid_scaled[keep_features].to_numpy()
y_valid = y_valid.to_numpy()

#%%
from  scipy.stats import uniform

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
        
        
        save_results_to_filename = '../results/model_hispanic_cv/' + str(key) + '_' + str(method) + '.xlsx'
        cross_validation(model,X,y,save_results_to_filename)
        
#%%
lr = LogisticRegression(random_state=12)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', lr)])


save_results_to_filename = '../results/model_hispanic_cv/LR.xlsx'
cross_validation(model,X, y, save_results_to_filename)

#%%

def parameter_tuning(model,parameter_grid,X_train,y_train,X_valid,y_valid,filename):
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)

    search = RandomizedSearchCV(model, parameter_grid, cv=skf, scoring=scoring,n_iter=50,
                         refit="AUC",random_state=12)


    result = search.fit(X, y)
    
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    
    # evaluate model on the validation set
    y_pred = best_model.predict_proba(X_valid)[:,1]
    
    # evaluate the model
    auc = roc_auc_score(y_valid, y_pred)
    ap = average_precision_score(y_valid, y_pred,average='samples')
    
    # report progress
    print('>auc=%.3f, ap=%.3f, est=%.3f, cfg=%s' % (auc, ap, result.best_score_, result.best_params_))
        
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(filename)
#%%

# Randomized search w/ Logistic Regression on validation set

best_classifier = LogisticRegression(random_state=12)

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
# randomized search 
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', best_classifier)])

param_grid = {
        'model__penalty': ('l1','l2'),
        'model__solver': ('liblinear', 'saga'),
        'model__max_iter': (100, 200,300,400,500,600,700,800,900,1000),
        } 

save_results_to_file = '../results/model_hispanic_tuning/LR.xlsx'

gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
gs.fit(X,y,X_valid,y_valid,scoring='roc_auc')

df_params = pd.DataFrame(gs.params)
df_scores = pd.DataFrame(gs.scores)

results = pd.concat([df_params,df_scores],axis=1)
results.to_csv(save_results_to_file)


#%%
from sklearn.metrics import roc_curve,precision_recall_curve

# compute Youden J statistic to determine optimal threshold 
def Youden_J_stat(y_true,y_pred):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # determine optimal threshold
    J = tpr - fpr
    idx = np.argmax(J)
    best_thresh = thresholds[idx]
    print('Best Threshold=%f' % (best_thresh))
    return best_thresh

def F_score_thresh(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    print(precision)
    print(recall)
    # compute f score
    fscore = (2 * precision * recall) / (precision + recall)
    fscore = np.nan_to_num(fscore,nan=0)
    # locate the index of the largest f score
    idx = np.argmax(fscore)
    best_thresh = thresholds[idx]
    print('Best Threshold=%f, F-Score=%.3f' % (best_thresh, fscore[idx]))
    return best_thresh
#%%
# optimal threshold 
best_classifier = LogisticRegression(max_iter=100,solver='liblinear',penalty='l1',random_state=12)

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', best_classifier)])

model.fit(X,y)

y_pred = model.predict_proba(X_valid)[:,1]

J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)

F_optimal_thresh = F_score_thresh(y_valid, y_pred)
#%%
# combine train and validation sets 

full_train = pd.concat([train_scaled,valid_scaled])
scaler.fit(full_train[keep_features])

full_train_scaled = scaler.transform(full_train[keep_features])

full_train_scaled = pd.DataFrame(full_train_scaled, columns=keep_features,index=full_train.index)
full_train_scaled = pd.concat([full_train.loc[:,~full_train.columns.isin(keep_features)],full_train_scaled],axis=1)

X_train = full_train_scaled[keep_features].to_numpy()

y_train_list = y_train.tolist()
y_valid_list = y_valid.tolist()

y_train = y_train_list + y_valid_list
y_train = np.array(y_train)

# normalize test data 
test_scaled = scaler.transform(X_test[keep_features])
test_scaled = pd.DataFrame(test_scaled, columns=keep_features,index=X_test.index)
test_scaled = pd.concat([X_test.loc[:,~X_test.columns.isin(keep_features)],test_scaled],axis=1)
test_scaled = test_scaled.reset_index(drop=True)

X_test = test_scaled[keep_features].to_numpy()
y_test = y_test.to_numpy()
test_scaled['label'] = y_test

#%%
# re-train model with train and validation sets
model.fit(X_train,y_train)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Logistic Regression',
               'Receiver operating characteristic curve: test set',
               '../results/model_hispanic_test/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Logistic Regression',
               'Precision-recall curve: test set',
               '../results/model_hispanic_test/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'Logistic Regression',
                       'Calibration Plot: test set',
                       '../results/model_hispanic_test/calibration_curve.png')


#%%
from sklearn.metrics import precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import roc_curve,precision_recall_curve


results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

#%%
J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/model_hispanic_test/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])

#%%

subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
thresh = F_optimal_thresh_test

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
subgrp_results_df.to_excel('../results/model_hispanic_test/subgroup_results.xlsx')

#%%
# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Logistic Regression model',
                             filename='../results/model_hispanic_test/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Logistic Regression model',
                             filename='../results/model_hispanic_test/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Logistic Regression model',
                             filename='../results/model_hispanic_test/calibration_plot_age.png')


# plot calibration curve by ethnicity
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Logistic Regression model',
                             filename='../results/model_hispanic_test/calibration_plot_ethnicity_gender.png')




#%%

# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[1][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[1][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[1][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/model_hispanic_test/lr_feature_importances.png'
plt.savefig(filename)
