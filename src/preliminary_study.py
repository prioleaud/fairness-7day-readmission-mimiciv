#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       preliminary_study.py
Description:    Fairness evaluation of 7-day hospital readmission prediction 
                with T2DM diabetes cohort from MIMIC-IV database 
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
from sklearn.model_selection import train_test_split

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
from sklearn.metrics import roc_auc_score, average_precision_score,confusion_matrix
from utils.metrics import Youden_J_thresh, F_score_thresh,output_results
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import precision_score,recall_score,balanced_accuracy_score

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

keep_features = train.columns.tolist()[-94:113]
keep_features.extend(train.columns.tolist()[1:12])

train_df = pd.concat([train,validation])

#%%
# min-max normalization 
scaler = MinMaxScaler()
scaler.fit(train_df[keep_features])

# normalize train data
train_scaled = scaler.transform(train_df[keep_features])

train_scaled = pd.DataFrame(train_scaled, columns=keep_features,index=train_df.index)
train_scaled = pd.concat([train_df.loc[:,~train_df.columns.isin(keep_features)],train_scaled],axis=1)



#%%
y = train_scaled['label']
X = train_scaled.drop(columns=['label'])
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)
X = X_train[keep_features].to_numpy()
y = y_train.to_numpy()
#%%
# normalize test data 
test_scaled = scaler.transform(test[keep_features])
test_scaled = pd.DataFrame(test_scaled, columns=keep_features,index=test.index)
test_scaled = pd.concat([test.loc[:,~test.columns.isin(keep_features)],test_scaled],axis=1)

X_test = test_scaled[keep_features].to_numpy()
y_test = test_scaled['label'].to_numpy()
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
valid_scaled = pd.concat([X_valid,y_valid],axis=1)

X_valid = X_valid[keep_features].to_numpy()
y_valid = y_valid.to_numpy()

#%%

# Randomized search w/ Random Forest on validation set

best_classifier = RandomForestClassifier(random_state=12)
best_calibration_method = 'sigmoid'

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

save_results_to_file = '../results/preliminary_tuning/RF_sigmoid_no_sampling.xlsx'

gs = GridSearch(model = model, param_grid = param_grid,parallelize=False)
gs.fit(X,y,X_valid,y_valid,scoring='roc_auc')

df_params = pd.DataFrame(gs.params)
df_scores = pd.DataFrame(gs.scores)

results = pd.concat([df_params,df_scores],axis=1)
results.to_csv(save_results_to_file)

#%%
# optimal threshold

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

model.fit(X,y)

y_valid_pred = model.predict_proba(X_valid)[:,1]

J_optimal_thresh_valid = Youden_J_thresh(y_valid, y_valid_pred)

F_optimal_thresh_valid = F_score_thresh(y_valid, y_valid_pred)
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
y_test_pred = y_pred 

#%%
# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Random Forest with Platt Scaling',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_test/roc_curve__no_sampling.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Random Forest with Platt Scaling',
               'Precision-recall curve: test set',
               '../results/preliminary_test/precision_recall_curve_no_sampling.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Random Forest with Platt Scaling',
                       'Calibration Plot: test set',
                       '../results/preliminary_test/calibration_curve_no_sampling.png')


plot_calibration_barplot(y_test,y_pred,5,'7-day Readmission Baseline Model',
                       '../results/preliminary_test/calibration_barplot.png')

#%%



results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

thresh_test = J_optimal_thresh_test

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))

#%%
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/overall_model_results_no_sampling.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])

#%%

subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
    #subgrp_results.append([name, subgroup['ethnicity'].unique()[0],subgroup['gender'].unique()[0]] + result)

groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)
    #subgrp_results.append([name, subgroup['ethnicity'].unique()[0],subgroup['age_binned'].unique()[0]] + result)


groups = test_results_df.groupby(['gender','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)
    #subgrp_results.append([name, subgroup['gender'].unique()[0],subgroup['age_binned'].unique()[0]] + result)


groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, subgroup in groups:
    result = output_results(subgroup['label'],subgroup['pred_proba'],thresh)
    subgrp_results.append([name] + result)
    #subgrp_results.append([name, subgroup['ethnicity'].unique()[0],subgroup['gender'].unique()[0],subgroup['age_binned'].unique()[0]] + result)

subgrp_results_df = pd.DataFrame(subgrp_results,
                                 columns=['Subgroup','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])
subgrp_results_df.to_excel('../results/preliminary_test/subgroup_results_no_sampling.xlsx')

#%%
# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_no_sampling.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_gender_no_sampling.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_age_no_sampling.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_gender_no_sampling.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_age_no_sampling.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_gender_age_no_sampling.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Random Forest with Platt Scaling',
                             filename='../results/preliminary_test/calibration_plot_ethnicity_gender_age_no_sampling.png')

#%%

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_test/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

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


#%%
'''
Apply Reweighing to training data with White as unprivileged group and Black/African American
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

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_white/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_white/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_white/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_white/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_white/calibration_plot_ethnicity_gender_age.png')


ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
Apply Reweighing to training data with Female as privileged group and Male as unprivileged group
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

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])


subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_gender.png')

# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Reweighing',
                             filename='../results/preliminary_reweighing_male/calibration_plot_ethnicity_gender_age.png')




# 
ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
Apply Reweighing to training data with 18 to 64 as privileged group and 65 to 91 as unprivileged group
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

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot after Reweighing',
                             filename='../results/preliminary_reweighing_older/calibration_plot_ethnicity_gender_age.png')

#
ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
Apply Reweighing to training data with Hipsanic/Latino and 65 to 91 as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','age_binned']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]')) else 1 for attrs in full_train_scaled[['ethnicity','age_binned']].itertuples(index=False) ]
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

unprivileged_groups = [{'hispanic_older_vs_all': 0}]
privileged_groups = [{'hispanic_older_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_hispanic_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_hispanic_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_hispanic_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_hispanic_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_hispanic_older/calibration_plot_ethnicity_gender_age.png')

#
ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_hispanic_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
filename = '../results/preliminary_reweighing_hispanic_older/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training data with White and male as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','gender']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['white_male_vs_all'] = [0 if ((attrs.ethnicity == 'WHITE') & (attrs.gender == 'M')) else 1 for attrs in full_train_scaled[['ethnicity','gender']].itertuples(index=False) ]
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity','gender'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

unprivileged_groups = [{'white_male_vs_all': 0}]
privileged_groups = [{'white_male_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_white_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_white_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_white_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_white_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_white_male/calibration_plot_ethnicity_gender_age.png')


ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_white_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])



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
filename = '../results/preliminary_reweighing_white_male/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training data with male and 65 to 91 as unprivileged group
'''

keep_cols = keep_features + ['label','gender','age_binned']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['male_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) else 1 for attrs in full_train_scaled[['gender','age_binned']].itertuples(index=False) ]
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

unprivileged_groups = [{'male_older_vs_all': 0}]
privileged_groups = [{'male_older_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_male_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_male_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_male_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_male_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_male_older/calibration_plot_ethnicity_gender_age.png')


#
ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_male_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
filename = '../results/preliminary_reweighing_male_older/rf_feature_importances.png'
plt.savefig(filename)

#%%
'''
Apply Reweighing to training data with 18 to 64 as privileged group and 65 to 91 as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','gender','age_binned']


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['3_vs_all'] = [0 if (((attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) | 
                                                      ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) |
                                                      ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]') & (attrs.gender == 'F'))
                                                      )else 1 for attrs in full_train_scaled[['ethnicity','gender','age_binned']].itertuples(index=False) ]
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity','gender','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

unprivileged_groups = [{'3_vs_all': 0}]
privileged_groups = [{'3_vs_all': 1}]
rw = Reweighing(unprivileged_groups, privileged_groups)
rw.fit(binaryLabelDataset)
df_transformed = rw.transform(binaryLabelDataset)

X_train = df_transformed.features[:,:104]
y_train = df_transformed.labels.ravel()
sample_weights = df_transformed.instance_weights.ravel()

best_classifier = RandomForestClassifier(n_estimators=100,max_features=2,criterion='entropy',random_state=12)
calibration_method = 'sigmoid'

skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
 
estimator = CalibratedClassifierCV(best_classifier,method=calibration_method,cv=skf)
model = Pipeline([
                    ('model', estimator)])

# re-train model with train and validation sets
model.fit(X_train,y_train,model__sample_weight=sample_weights)

# predict outcomes for test set 
y_pred = model.predict_proba(X_test)[:,1]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'RF with Platt Scaling: after Reweighing',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_reweighing_ethnicity_gender_age/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'RF with Platt Scaling: after Reweighing ',
               'Precision-recall curve: test set',
               '../results/preliminary_reweighing_ethnicity_gender_age/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'RF with Platt Scaling: after Reweighing',
                       'Calibration Plot: test set',
                       '../results/preliminary_reweighing_ethnicity_gender_age/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_gender.png')

# plot calibration curve by age
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration after Reweighing',
                             filename='../results/preliminary_reweighing_ethnicity_gender_age/calibration_plot_ethnicity_gender_age.png')

groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_reweighing_ethnicity_gender_age/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


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
filename = '../results/preliminary_reweighing_ethnicity_gender_age/rf_feature_importances.png'
plt.savefig(filename)
#%%
'''
Apply Prejudice Remover 
'''
from aif360.algorithms.inprocessing import PrejudiceRemover

y = train_scaled['label']
X = train_scaled.drop(columns=['label'])
train_X, valid_X, train_y, valid_y = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)
X_train_scaled = pd.concat([train_X,train_y],axis=1)
#%%
'''
Apply Prejudice Remover to training daa with White as unprivileged group and Hispanic/Latino
and Black/African American as privileged group
'''
keep_cols = keep_features + ['label','ethnicity']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in X_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in valid_scaled.ethnicity] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)


#%%

convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in full_train_scaled.ethnicity] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in test_scaled.ethnicity] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['ethnicity'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]



model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset)

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_white/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_white/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_white/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_white/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])



#%%
'''
Apply Prejudice Remover to training data with Female as privileged group and Male as unprivileged group
'''

keep_cols = keep_features + ['label','gender']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in X_train_scaled.gender] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in valid_scaled.gender] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in full_train_scaled.gender] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in test_scaled.gender] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])



#%%
'''
Apply Prejudice Remover to training data with (17, 64] as privileged group and (64, 91] as unprivileged group
'''

keep_cols = keep_features + ['label','age_binned']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in X_train_scaled.age_binned] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['age_binned'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in valid_scaled.age_binned] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 5.0
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in full_train_scaled.age_binned] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in test_scaled.age_binned] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_older/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Prejudice Remover to training data with Hispanic/Latino and (64, 91] as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','age_binned']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.ethnicity == 'HISPANIC/LATINO')) else 1 for attrs in X_train_scaled[['ethnicity','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity','age_binned'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.ethnicity == 'HISPANIC/LATINO')) else 1 for attrs in valid_scaled[['ethnicity','age_binned']].itertuples(index=False)] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['ethnicity','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.ethnicity == 'HISPANIC/LATINO')) else 1 for attrs in full_train_scaled[['ethnicity','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['ethnicity','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.ethnicity == 'HISPANIC/LATINO')) else 1 for attrs in test_scaled[['ethnicity','age_binned']].itertuples(index=False)]  
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['ethnicity','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_hispanic_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_hispanic_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_hispanic_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_hispanic_older/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_hispanic_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Prejudice Remover to training data with (17, 64] as privileged group and (64, 91] as unprivileged group
'''

keep_cols = keep_features + ['label','gender','age_binned']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['male_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) else 1 for attrs in X_train_scaled[['gender','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','age_binned'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['male_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) else 1 for attrs in valid_scaled[['gender','age_binned']].itertuples(index=False)] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['male_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) else 1 for attrs in full_train_scaled[['gender','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['male_older_vs_all'] = [0 if ((attrs.age_binned == '(64, 91]') & (attrs.gender == 'M')) else 1 for attrs in test_scaled[['gender','age_binned']].itertuples(index=False)]   
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_male_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_male_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_male_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_male_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_male_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_male_older/calibration_plot_ethnicity_gender_age.png')

#%%
'''
Apply Prejudice Remover to training data with White and male as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','gender']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['white_male_vs_all'] = [0 if ((attrs.ethnicity == 'WHITE') & (attrs.gender == 'M')) else 1 for attrs in X_train_scaled[['gender','ethnicity']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','ethnicity'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['white_male_vs_all'] = [0 if ((attrs.ethnicity == 'WHITE') & (attrs.gender == 'M')) else 1 for attrs in valid_scaled[['gender','ethnicity']].itertuples(index=False)] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','ethnicity'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['white_male_vs_all'] = [0 if ((attrs.ethnicity == 'WHITE') & (attrs.gender == 'M')) else 1 for attrs in full_train_scaled[['gender','ethnicity']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','ethnicity'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['white_male_vs_all'] = [0 if ((attrs.ethnicity == 'WHITE') & (attrs.gender == 'M')) else 1 for attrs in test_scaled[['gender','ethnicity']].itertuples(index=False)]   
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','ethnicity'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_white_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_white_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_white_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_white_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_white_male/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_white_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Prejudice Remover to training data with (17, 64] as privileged group and (64, 91] as unprivileged group
'''

keep_cols = keep_features + ['label','ethnicity','gender','age_binned']


convert_train_to_num_df = X_train_scaled[keep_cols]
convert_train_to_num_df['3_vs_all'] = [0 if (((attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'F') & (attrs.age_binned == '(64, 91]'))
                                       ) else 1 for attrs in X_train_scaled[['gender','ethnicity','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','ethnicity','age_binned'])


binaryLabelDataset_train = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

binaryLabelDataset_train.features = binaryLabelDataset_train.features[:,:104]
binaryLabelDataset_train.feature_names = binaryLabelDataset_train.feature_names[:104]

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['3_vs_all'] = [0 if (((attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'F') & (attrs.age_binned == '(64, 91]'))
                                       ) else 1 for attrs in valid_scaled[['gender','ethnicity','age_binned']].itertuples(index=False)] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','ethnicity','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

sens_attr = binaryLabelDataset_train.protected_attribute_names[0]

# evaluate the model

aucs = []

eta_value = [0.01,1.0,5.0,10.0,15.0,20.0,50.0] # Parameter for Prejudice Remover
y_valid = valid_scaled['label']
for eta in eta_value:
    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=eta)
    model = model.fit(binaryLabelDataset_train)
    y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
    aucs.append(roc_auc_score(y_valid, y_pred))
        
# AUC by eta 
plt.figure()
plt.plot(eta_value,aucs)
plt.ylim([0,1])
plt.xlabel('eta value (Prejudice Remover parameter)')
plt.ylabel('AUROC')

#%%
best_eta = 0.01
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=best_eta)
model = model.fit(binaryLabelDataset_train)
y_pred = model.predict(binaryLabelDataset_valid).scores[:, 0]
  
J_optimal_thresh = Youden_J_thresh(y_valid, y_pred)  
F_optimal_thresh = F_score_thresh(y_valid, y_pred)

#%%


convert_train_to_num_df = full_train_scaled[keep_cols]
convert_train_to_num_df['3_vs_all'] = [0 if (((attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'F') & (attrs.age_binned == '(64, 91]'))
                                       ) else 1 for attrs in full_train_scaled[['gender','ethnicity','age_binned']].itertuples(index=False)] 
convert_train_to_num_df = convert_train_to_num_df.drop(columns=['gender','ethnicity','age_binned'])


binaryLabelDataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_train_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

sens_attr = binaryLabelDataset.protected_attribute_names[0]
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)

binaryLabelDataset.features = binaryLabelDataset.features[:,:104]
binaryLabelDataset.feature_names = binaryLabelDataset.feature_names[:104]
model = model.fit(binaryLabelDataset)

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['3_vs_all'] = [0 if (((attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) |
                                             ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.gender == 'F') & (attrs.age_binned == '(64, 91]'))
                                       ) else 1 for attrs in test_scaled[['gender','ethnicity','age_binned']].itertuples(index=False)]   
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','ethnicity','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

# predict outcomes for test set 
y_pred = model.predict(binaryLabelDataset_Xtest).scores[:, 0]

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Prejudice Remover',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_prejudiceremover_ethnicity_gender_age/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Prejudice Remover',
               'Precision-recall curve: test set',
               '../results/preliminary_prejudiceremover_ethnicity_gender_age/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Prejudice Remover',
                       'Calibration Plot: test set',
                       '../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Prejudice Remover',
                             filename='../results/preliminary_prejudiceremover_ethnicity_gender_age/calibration_plot_ethnicity_gender_age.png')


ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_prejudiceremover_ethnicity_gender_age/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with (17, 64] as privileged group and (64, 91] as unprivileged group
'''

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
#%%
keep_cols = keep_features + ['label','age_binned']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in valid_scaled.age_binned] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['older_vs_all'] = [1 if age_grp == '(17, 64]' else 0 for age_grp in test_scaled.age_binned] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_older/calibration_plot_ethnicity_gender_age.png')


ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with Male as unprivileged group
'''

keep_cols = keep_features + ['label','gender']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in valid_scaled.gender] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['male_vs_all'] = [1 if gender == 'F' else 0 for gender in test_scaled.gender] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])




#%%
'''
Apply Calibrated Equalized Odds to training data with White as unprivileged group and 
Black/African American and Hispanic/Latino as privileged group
'''

keep_cols = keep_features + ['label','ethnicity']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in valid_scaled.ethnicity] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['ethnicity'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['white_vs_all'] = [0 if ethnicity == 'WHITE' else 1 for ethnicity in test_scaled.ethnicity] 
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['ethnicity'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_white/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_white/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_white/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with Hispanic/Latino and (64, 91] as unprivileged
'''

keep_cols = keep_features + ['label','ethnicity','age_binned']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]')) else 1 for attrs in valid_scaled[['ethnicity','age_binned']].itertuples(index=False) ] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['ethnicity','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['hispanic_older_vs_all'] = [0 if ((attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]')) else 1 for attrs in test_scaled[['ethnicity','age_binned']].itertuples(index=False) ]  
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['ethnicity','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['hispanic_older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_hispanic_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_hispanic_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_hispanic_older/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_hispanic_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with Black/African American and Hispanic/Latino as unprivileged group and 
White as privileged group
'''

keep_cols = keep_features + ['label','gender','age_binned']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['male_older_vs_all'] = [0 if ((attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) else 1 for attrs in valid_scaled[['gender','age_binned']].itertuples(index=False) ] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['male_older_vs_all'] = [0 if ((attrs.gender == 'M') & (attrs.age_binned == '(64, 91]')) else 1 for attrs in test_scaled[['gender','age_binned']].itertuples(index=False) ]  
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['male_older_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_male_older/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_male_older/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_male_older/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_male_older/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_male_older/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with White and male as unprivileged group  
'''

keep_cols = keep_features + ['label','gender','ethnicity']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['white_male_vs_all'] = [0 if ((attrs.gender == 'M') & (attrs.ethnicity == 'WHITE')) else 1 for attrs in valid_scaled[['gender','ethnicity']].itertuples(index=False) ] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','ethnicity'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['white_male_vs_all'] = [0 if ((attrs.gender == 'M') & (attrs.ethnicity == 'WHITE')) else 1 for attrs in test_scaled[['gender','ethnicity']].itertuples(index=False) ]  
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','ethnicity'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['white_male_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_white_male/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_white_male/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_white_male/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_white_male/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_white_male/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])


#%%
'''
Apply Calibrated Equalized Odds to training data with Black/African American and Hispanic/Latino as unprivileged group and 
White as privileged group
'''

keep_cols = keep_features + ['label','gender','ethnicity','age_binned']

convert_valid_to_num_df = valid_scaled[keep_cols]
convert_valid_to_num_df['3_vs_all'] = [0 if (((attrs.gender == 'M') & (attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.age_binned == '(64, 91]')) |
                                                      ((attrs.gender == 'M') & (attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]')) |
                                                      ((attrs.gender == 'F') & (attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]'))
                                                      ) else 1 for attrs in valid_scaled[['gender','ethnicity','age_binned']].itertuples(index=False) ] 
convert_valid_to_num_df = convert_valid_to_num_df.drop(columns=['gender','ethnicity','age_binned'])


binaryLabelDataset_valid = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_valid_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

binaryLabelDataset_valid.features = binaryLabelDataset_valid.features[:,:104]
binaryLabelDataset_valid.feature_names = binaryLabelDataset_valid.feature_names[:104]

dataset_valid_pred = binaryLabelDataset_valid.copy(deepcopy=True)
dataset_valid_pred.scores = y_valid_pred
dataset_valid_pred.labels = (y_valid_pred >= J_optimal_thresh_valid).astype('float64')

# convert test data to BinaryLabelDataset
convert_test_to_num_df = test_scaled[keep_cols]
convert_test_to_num_df['3_vs_all'] = [0 if (((attrs.gender == 'M') & (attrs.ethnicity == 'BLACK/AFRICAN AMERICAN') & (attrs.age_binned == '(64, 91]')) |
                                                     ((attrs.gender == 'M') & (attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]')) |
                                                     ((attrs.gender == 'F') & (attrs.ethnicity == 'HISPANIC/LATINO') & (attrs.age_binned == '(64, 91]'))
                                                     ) else 1 for attrs in test_scaled[['gender','ethnicity','age_binned']].itertuples(index=False) ]  
convert_test_to_num_df = convert_test_to_num_df.drop(columns=['gender','ethnicity','age_binned'])

binaryLabelDataset_Xtest = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=convert_test_to_num_df,
    label_names=['label'],
    protected_attribute_names=['3_vs_all'])

binaryLabelDataset_Xtest.features = binaryLabelDataset_Xtest.features[:,:104]
binaryLabelDataset_Xtest.feature_names = binaryLabelDataset_Xtest.feature_names[:104]

dataset_test_pred = binaryLabelDataset_Xtest.copy(deepcopy=True)
dataset_test_pred.scores = y_test_pred
dataset_test_pred.labels = (y_test_pred >= thresh_test).astype('float64')

rnd_seed = 12

privileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.privileged_protected_attributes[0]}]
unprivileged_groups = [{binaryLabelDataset_valid.protected_attribute_names[0]: binaryLabelDataset_valid.unprivileged_protected_attributes[0]}]

# Learn parameters to equalize odds and apply to create a new dataset
cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint='weighted',
                                     seed=rnd_seed)
cpp = cpp.fit(binaryLabelDataset_valid,dataset_valid_pred)
y_pred = cpp.predict(dataset_test_pred,threshold=thresh_test).scores

# evaluate the model
# AUC
auc = roc_auc_score(y_test, y_pred)
plot_roc_curve(y_test, y_pred,'Calibrated Equalized Odds',
               'Receiver operating characteristic curve: test set',
               '../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/roc_curve.png')

# Average Precision
ap = average_precision_score(y_test, y_pred,average='samples')
plot_pr_curve(y_test,y_pred,'Calibrated Equalized Odds',
               'Precision-recall curve: test set',
               '../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/precision_recall_curve.png')

#Calibration Curve
plot_calibration_curve(y_test,y_pred,5,'Calibrated Equalized Odds',
                       'Calibration Plot: test set',
                       '../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_curve.png')

results = []
results.append(['J thresh - validation set'] + output_results(y_test,y_pred,J_optimal_thresh_valid))
results.append(['F thresh - validation set'] + output_results(y_test,y_pred,F_optimal_thresh_valid))

J_optimal_thresh_test = Youden_J_thresh(y_test, y_pred)

F_optimal_thresh_test = F_score_thresh(y_test, y_pred)

results.append(['J thresh - test set'] + output_results(y_test,y_pred,J_optimal_thresh_test))
results.append(['F thresh - test set'] + output_results(y_test,y_pred,F_optimal_thresh_test))


results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/overall_model_results.xlsx',
                  header=['Threshold Type','AUC','AP','FPR','FNR','ECE','MCE','HLH','HLH_pval','prob_thresh','Precision','Recall','Balanced Accuracy'])



subgrp_results = []
test_results_df = pd.concat([test_scaled,pd.DataFrame(y_pred,columns=['pred_proba'])],axis=1)
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
subgrp_results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/subgroup_results.xlsx')


# plot calibration curve by ethnicity
col = 'ethnicity'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_ethnicity.png')


# plot calibration curve by gender
col = 'gender'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_gender.png')

# plot calibration curve by ethnicity
col = 'age_binned'
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_age.png')


# plot calibration curve by ethnicity and gender
col = ['ethnicity','gender']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_ethnicity_gender.png')


# plot calibration curve by ethnicity and age
col = ['ethnicity','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_ethnicity_age.png')

# plot calibration curve by gender and age
col = ['gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_gender_age.png')

# plot calibration curve by ethnicity, gender and age
col = ['ethnicity','gender','age_binned']
true_label = 'label'
pred_label = 'pred_proba'
plot_calibration_by_subgroup(test_results_df,col,true_label,pred_label,5,
                             plot_title='Calibration Plot - Calibrated Equalized Odds',
                             filename='../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/calibration_plot_ethnicity_gender_age.png')

ethnic_groups = test_results_df.ethnicity.unique()
results = []
for grp in ethnic_groups:
    sub_df = test_results_df[test_results_df.ethnicity == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/ethnicity_calibration.xlsx',
                  header=['Ethnicity','Prob_true','Prob_pred'])
    
groups = test_results_df.gender.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.gender == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/gender_calibration.xlsx',
                  header=['Gender','Prob_true','Prob_pred'])

groups = test_results_df.age_binned.unique()
results = []
for grp in groups:
    sub_df = test_results_df[test_results_df.age_binned == grp]

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(grp),prob_true, prob_pred])
    
    groups = test_results_df.gender.unique()

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/age_calibration.xlsx',
                  header=['Age','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/ethnicity_gender_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/ethnicity_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])
    
results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

results = []
groups = test_results_df.groupby(['ethnicity','gender','age_binned'])

for name, grp in groups:
    sub_df = grp

    prob_true, prob_pred = calibration_curve(sub_df['label'],sub_df['pred_proba'],n_bins=5,strategy='uniform')
    results.append([str(name),prob_true, prob_pred])

results_df = pd.DataFrame(results)
results_df.to_excel('../results/preliminary_calibratedequalizedodds_ethnicity_gender_age/ethnicity_gender_age_calibration.xlsx',
                  header=['Subgroup','Prob_true','Prob_pred'])

