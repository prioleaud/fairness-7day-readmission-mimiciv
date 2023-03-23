#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       preliminary_study.py
Description:    Fairness evaluation of 7-day hospital readmission prediction 
                with T2DM diabetes cohort from MIMIC-IV database 
Author:         Diandra Prioleau Ojo
Date Created:   February 15, 2023
"""

# import packges
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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

#%%
keep_features = train.columns.tolist()[-94:113]
keep_features.extend(train.columns.tolist()[1:12])

#%%
# min-max normalization 
scaler = MinMaxScaler()
scaler.fit(train[keep_features])

# normalize train data
train_scaled = scaler.transform(train[keep_features])

train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=train.index)
train_scaled = pd.concat([train.loc[:,~train.columns.isin(keep_features)],train_scaled],axis=1)

X = train_scaled[keep_features].to_numpy()
y = train_scaled['label'].to_numpy()

#%%
# normalize validation data 
valid_scaled = scaler.transform(validation[keep_features])
valid_scaled = pd.DataFrame(valid_scaled, columns=keep_features,index=validation.index)
valid_scaled = pd.concat([validation.loc[:,~validation.columns.isin(keep_features)],valid_scaled],axis=1)

X_valid = valid_scaled[keep_features].to_numpy()
y_valid = valid_scaled['label'].to_numpy()

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
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])
        
        
        save_results_to_filename = '../results/preliminary_cv/' + str(key) + '_' + str(method) + '_no_sampling.xlsx'
        cross_validation(model,X,y,save_results_to_filename)
        
#%%
lr = LogisticRegression(random_state=12)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', lr)])


save_results_to_filename = '../results/preliminary_cv/LR_no_sampling.xlsx'
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

# Randomized search w/ Random Forest on validation set

best_classifier = RandomForestClassifier(random_state=12)
best_calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
# randomized search 
estimator = CalibratedClassifierCV(best_classifier,method=best_calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

param_grid = {
        'model__base_estimator__n_estimators': (100,200,300,400,500),
        'model__base_estimator__criterion': ('gini', 'entropy'),
        'model__base_estimator__max_features': (2,4,6,8,10,12)
        } 

save_results_to_file = '../results/preliminary_tuning/RF_isotonic_no_sampling.xlsx'

gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
gs.fit(X,y,X_valid,y_valid,scoring='roc_auc')

df_params = pd.DataFrame(gs.params)
df_scores = pd.DataFrame(gs.scores)

results = pd.concat([df_params,df_scores],axis=1)
results.to_csv(save_results_to_file)

#%%

classifier = LogisticRegression(random_state=12)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', classifier)])
param_grid = {
        'model__penalty': ('l1','l2'),
        'model__solver': ('liblinear', 'saga'),
        'model__max_iter': (100, 200,300,400,500,600,700,800,900,1000),
        } 
save_results_to_file = '../results/preliminary_tuning/LR.xlsx'
#parameter_tuning(model, param_grid, X, y, X_valid, y_valid, save_results_to_file)

gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
gs.fit(X,y,X_valid,y_valid,scoring='roc_auc')

df_params = pd.DataFrame(gs.params)
df_scores = pd.DataFrame(gs.scores)

results = pd.concat([df_params,df_scores],axis=1)
results.to_csv(save_results_to_file)

#%%
# Randomized search w/ KNN on validation set

classifier = KNeighborsClassifier()
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

param_grid = {
        'model__base_estimator__n_neighbors': (3,4,5,6,7,8,9,10,11,12,13,14,15),
        'model__base_estimator__metric': ('minkowski', 'euclidean','manhattan'),
        'model__base_estimator__weights': ('uniform','distance')
        } 

save_results_to_file = '../results/preliminary_tuning/KNN_sigmoid.xlsx'

gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
gs.fit(X,y,X_valid,y_valid,scoring='roc_auc')

df_params = pd.DataFrame(gs.params)
df_scores = pd.DataFrame(gs.scores)

results = pd.concat([df_params,df_scores],axis=1)
results.to_csv(save_results_to_file)
#%%
# Randomized search w/ Gradient Boosting on validation set

classifier = GradientBoostingClassifier(random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

param_grid = {
        'model__base_estimator__n_estimators': (100,200,300,400,500),
        'model__base_estimator__min_samples_split': (np.linspace(0.1, 1.0, 10, endpoint=True)),
        'model__base_estimator__min_samples_leaf': (np.linspace(0.1, 1.0, 10, endpoint=True)),
        'model__base_estimator__max_features': (2,4,6,8,10,12)
        } 

save_results_to_file = '../results/preliminary_tuning/GB_sigmoid.xlsx'
#parameter_tuning(model, param_grid, X, y, X_valid, y_valid, save_results_to_file)

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
best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

model.fit(X,y)

y_pred = model.predict_proba(X_valid)[:,1]

J_optimal_thresh = Youden_J_stat(y_valid, y_pred)

F_optimal_thresh = F_score_thresh(y_valid, y_pred)
#%%
# combine train and validation sets 

full_train = pd.concat([train,validation])
scaler.fit(full_train[keep_features])

full_train_scaled = scaler.transform(full_train[keep_features])

full_train_scaled = pd.DataFrame(full_train_scaled, columns=keep_features,index=full_train.index)
full_train_scaled = pd.concat([full_train.loc[:,~full_train.columns.isin(keep_features)],full_train_scaled],axis=1)

X_train = full_train_scaled[keep_features].to_numpy()
y_train = full_train_scaled['label'].to_numpy()

# normalize test data 
test_scaled = scaler.transform(test[keep_features])
test_scaled = pd.DataFrame(test_scaled, columns=keep_features,index=test.index)
test_scaled = pd.concat([test.loc[:,~test.columns.isin(keep_features)],test_scaled],axis=1)

X_test = test_scaled[keep_features].to_numpy()
y_test = test_scaled['label'].to_numpy()

# re-train model with train and validation sets
model.fit(X_train,y_train)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_test/roc_curve__no_sampling.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration',
               'Precision-recall curve: test set',
               '../results/preliminary_test/precision_recall_curve_no_sampling.png')
#%%
#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration',
                       'Calibration Plot: test set',
                       '../results/preliminary_test/calibration_curve_no_sampling.png')


#%%
from sklearn.metrics import precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import roc_curve,precision_recall_curve


results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

#%%
J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/overall_model_results_no_sampling.xlsx',
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
subgrp_results_df.to_excel('../results/preliminary_test/subgroup_results_no_sampling.xlsx')

#%%
# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_no_sampling.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_test/calibration_plot_gender_no_sampling.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_test/calibration_plot_age_no_sampling.png')


# plot calibration curve by ethnicity
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_gender_no_sampling.png')




#%%

# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_test/rf_feature_importances_no_sampling.png'
plt.savefig(filename)

#%%

'''
Bias Migitaion Experiments
Description:    Using AIF360 mitigaiton methods - Reweighing, Prejudice Remover, and 
                Calibrated Equalized Odds
'''

# import algorithms from AIF360
from aif360.algorithms.preprocessing import Reweighing

# import helper functions from AIF360
from aif360.datasets import BinaryLabelDataset
'''
Apply Reweighing to training daa with Hispanic as unprivileged group and Black/African American
and White as privileged group
'''

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['hispanic_vs_all'] = [0 if ethnicity == 'HISPANIC/LATINO' else 1 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])
#convert_train_to_num_df = convert_train_to_num_df.set_index('hispanic_vs_all')



binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_vs_all'])

unprivileged_groups = [{'hispanic_vs_all': 0}]
privileged_groups = [{'hispanic_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_hispanic/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_hispanic/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_hispanic/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_hispanic/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic/calibration_plot_ethnicity_gender_age.png')



# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_hispanic/rf_feature_importances.png'
plt.savefig(filename)


#%%
'''
Apply Reweighing to training daa with White as unprivileged group and Black/African American
and Hispanic as privileged group
'''

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

unprivileged_groups = [{'white_vs_all': 0}]
privileged_groups = [{'white_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_white/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_white/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_white/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_white/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_gender_age.png')



# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_white/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with Black/African American as unprivileged group and White
and Hispanic as privileged group
'''

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['black_vs_all'] = [0 if ethnicity == 'BLACK/AFRICAN AMERICAN' else 1 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['black_vs_all'])

unprivileged_groups = [{'black_vs_all': 0}]
privileged_groups = [{'black_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_black/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_black/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_black/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_black/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_black/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_black/calibration_plot_ethnicity_gender_age.png')



# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_black/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with Black/African American as privileged group and White
and Hispanic as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['black_vs_all'] = [1 if ethnicity == 'BLACK/AFRICAN AMERICAN' else 0 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['black_vs_all'])

unprivileged_groups = [{'black_vs_all': 0}]
privileged_groups = [{'black_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_hispanic_white/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_hispanic_white/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_hispanic_white/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_white/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_hispanic_white/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_white/calibration_plot_ethnicity_gender_age.png')


# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_hispanic_white/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with White as privileged group and Black/African American
and Hispanic as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['white_vs_all'] = [1 if ethnicity == 'WHITE' else 0 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

unprivileged_groups = [{'white_vs_all': 0}]
privileged_groups = [{'white_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_hispanic_black/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_hispanic_black/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_hispanic_black/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_black/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_hispanic_black/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_hispanic_black/calibration_plot_ethnicity_gender_age.png')



# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_hispanic_black/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with Male as privileged group and Female as unprivileged group
'''

keep_cols = keep_features + ['label','gender']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['female_vs_all'] = [1 if gender == 'M' else 0 for gender in full_train_scaled.gender] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['female_vs_all'])

unprivileged_groups = [{'female_vs_all': 0}]
privileged_groups = [{'female_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_female/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_female/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_female/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_female/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_female/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_ethnicity_gender.png')

# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_female/calibration_plot_ethnicity_gender_age.png')





# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_female/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with Female as privileged group and Male as unprivileged group
'''

keep_cols = keep_features + ['label','gender']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in full_train_scaled.gender] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

unprivileged_groups = [{'male_vs_all': 0}]
privileged_groups = [{'male_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_gender.png')

# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_gender_age.png')




# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_male/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training daa with 18 to 64 as privileged group and 65 to 91 as unprivileged group
'''

keep_cols = keep_features + ['label','age_binned']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in full_train_scaled.age_binned] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

unprivileged_groups = [{'older_vs_all': 0}]
privileged_groups = [{'older_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=500,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'isotonic'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    #('sampling', SMOTE(sampling_strategy=1.0,random_state=12)),
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF w/ isotonic calibration: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF w/ isotonic calibration: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'RF w/ isotonic calibration: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_reweighing_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Random Forest model',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_gender_age.png')


# display feature importance for Random Forest model

feature_names = test_scaled.columns[10:]

feat_importances = 0
for i in model.steps[0][1].calibrated_classifiers_:
    feat_importances = feat_importances + i.base_estimator.feature_importances_
feat_importances  = feat_importances/len(model.steps[0][1].calibrated_classifiers_)

std = np.std([tree.base_estimator.feature_importances_ for tree in model.steps[0][1].calibrated_classifiers_], axis=0)


rf_importances = pd.DataFrame(feat_importances, columns=['value'],index=feature_names)
rf_importances['std'] = std
rf_importances = rf_importances.sort_values(by=['value'],ascending=False)

plt.figure(figsize=(20,30))
fig, ax = plt.subplots()
rf_importances['value'][:50].plot.bar(yerr=rf_importances['std'][:50],ax=ax)
ax.set_title("Feature importances for RF model")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
filename = '../results/preliminary_reweighing_older/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Prejudice Remover to training daa with Hispanic/Latino as privileged group and Black/African American
and White as unprivileged group
'''
from aif360.algorithms.inprocessing import PrejudiceRemover

keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['hipsanic_vs_all'] = [0 if ethnicity == 'HISPANIC/LATINO' else 1 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hipsanic_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['hipsanic_vs_all'] = [0 if ethnicity == 'HISPANIC/LATINO' else 1 for ethnicity in test_scaled.ethnicity] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['ethnicity'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hipsanic_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_hispanic/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_hispanic/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,10,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_hispanic/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_stat(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','FPR','Balanced Accuracy'])



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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_hispanic/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,10,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic/calibration_plot_ethnicity_gender_age.png')

