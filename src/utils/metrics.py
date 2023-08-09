#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       metrics.py
Description:    This script includes performing a grid search CV to determine
                the models of the ensemble model based on the best model 
                (best AUC score) for each subset/bag of the ensemble 
Author:         Diandra Prioleau Ojo
Date Created:   November 15, 2022
"""

# import packges
import numpy as np
import pandas as pd 
import scipy.stats as st
from scipy.stats import chi2
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from sklearn.metrics import precision_score,recall_score,balanced_accuracy_score,roc_auc_score,average_precision_score

# compute Youden J statistic to determine optimal threshold 
def Youden_J_thresh(y_true,y_pred):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # determine optimal threshold
    print(thresholds)
    J = tpr - fpr
    print(J)
    idx = np.argmax(J)
    best_thresh = thresholds[idx]
    print('Best Threshold=%f, J-statistic=%.3f' % (best_thresh, J[idx]))
    return best_thresh

def F_score_thresh(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # compute f score
    fscore = (2 * precision * recall) / (precision + recall)
    fscore = np.nan_to_num(fscore,nan=0)
    # locate the index of the largest f score
    idx = np.argmax(fscore)
    best_thresh = thresholds[idx]
    print('Best Threshold=%f, F-Score=%.3f' % (best_thresh, fscore[idx]))
    return best_thresh

def confidence_interval(df,columns):
 
    ci = st.norm.interval(confidence=0.95,
                         loc=np.mean(df[col]),
                         scale=st.sem(df[col]))
    ci_95.append(ci)


def average_recall(precisions,recalls):
    size = len(precisions)-1
    avg_recall = 0
    for i in range(size,0,-1):
        avg_recall += recalls[i]
    avg_recall /= size
    
    return avg_recall
        
def expected_calibration_error(y, y_pred, bins = 10):
    '''
    Compute expcted calibration error using Freedman-Diaconis rule to 
    determine the number of bins 
    reference: https://towardsdatascience.com/pythons-predict-proba-doesn-t-actually-predict-probabilities-and-how-to-fix-it-f582c21d63fc
    '''
    bin_count, bin_edges = np.histogram(y_pred, bins = bins)
    n_bins = len(bin_count) 
    bin_edges[0] -= 1e-8 # because left edge is not included
    bin_id = np.digitize(y_pred, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins) 
    bin_probasum = np.bincount(bin_id, weights = y_pred, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0) 
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0) 
    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(y_pred) 
    return ece

def hosmer_lemeshow(y_true,y_pred_proba,g=10):
    ' groups are divided based on equal increment thresholds for the estimates'
    bins = pd.cut(y_pred_proba,bins=g)
    d = {'y_true': y_true, 'y_pred_proba': y_pred_proba, 'group':bins}
    df = pd.DataFrame(d)
    groups = df.groupby('group')
    
    hlh_stat = 0
    for name,grp in groups:
        observed_pos = len(grp[grp['y_true'] == 1])
        expected_pos = np.sum(grp['y_pred_proba'])

        if expected_pos == 0:
            expected_pos = 1e-8
            
        observed_neg = len(grp[grp['y_true'] == 0])
        expected_neg = np.sum(1-grp['y_pred_proba'])
        
        if expected_neg == 0:
            expected_neg = 1e-8
        
        hlh_stat += (np.square(observed_pos - expected_pos)/expected_pos) + \
                    (np.square(observed_neg - expected_neg)/expected_neg)   
    
    pval = 1 - chi2.cdf(hlh_stat, g - 2)
    return hlh_stat,pval

def max_calibration_error(y,y_pred,bins=10):
    bin_count, bin_edges = np.histogram(y_pred, bins = bins)
    n_bins = len(bin_count) 
    bin_edges[0] -= 1e-8 # because left edge is not included
    bin_id = np.digitize(y_pred, bin_edges, right = True) - 1
    bin_ysum = np.bincount(bin_id, weights = y, minlength = n_bins) 
    bin_probasum = np.bincount(bin_id, weights = y_pred, minlength = n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out = np.zeros(n_bins), where = bin_count > 0) 
    bin_probamean = np.divide(bin_probasum, bin_count, out = np.zeros(n_bins), where = bin_count > 0) 
    mce = np.amax((bin_probamean - bin_ymean)) 
    return mce

def output_results(y_true,y_pred,prob_thresh=None):
    
    try:
        auc = roc_auc_score(y_true,y_pred)
    except:
        auc = 0

    ap = average_precision_score(y_true,y_pred,average='samples')
    
    ece = expected_calibration_error(y_true,y_pred)
    mce = max_calibration_error(y_true,y_pred)
    hlh, pval = hosmer_lemeshow(y_true,y_pred)
    
    if prob_thresh != None:
        pred = y_pred >= prob_thresh
        precision = precision_score(y_true, pred)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        fpr = fp/(fp+tn)
        fnr = fn/(fn+tp)
        recall = recall_score(y_true, pred)
        bal_acc = balanced_accuracy_score(y_true, pred)
    
    return [auc,ap,fpr,fnr,ece,mce,hlh,pval,prob_thresh,precision,recall,bal_acc]