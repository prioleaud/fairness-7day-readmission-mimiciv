#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       model_selection.py
Description:    Create and save plots 
Date Created:   February 25, 2022
"""
# import packages
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# import metrics
from sklearn.metrics import roc_auc_score, average_precision_score

# function for cross validation 
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


def parameter_tuning(model,parameter_grid,X_train,y_train,X_valid,y_valid,filename):
    skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=12)
    
    scoring = {"AUC": "roc_auc",
           "AP":"average_precision",
           }

    search = RandomizedSearchCV(model, parameter_grid, cv=skf, scoring=scoring,n_iter=50,
                         refit="AUC",random_state=12)


    result = search.fit(X_train, y_train)
    
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
