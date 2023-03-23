#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       save_results.py
Description:    
Author:         Diandra Prioleau Ojo
Date Created:   February 7, 2022
"""
import pandas as pd 
import numpy as np

# import metrics
from sklearn.metrics import roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve
from utils.metrics import hosmer_lemeshow,expected_calibration_error,max_calibration_error,average_recall,confidence_interval
from sklearn.metrics import precision_recall_fscore_support

def results_by_single_social_identity(df,true_label,pred_label):
            
    try:
        auc_test = roc_auc_score(df[true_label],df[pred_label])
    except:
        auc_test = 0
    ap_test = average_precision_score(df[true_label],df[pred_label],average='samples')
    
    mean_recall = np.linspace(0,1,100)
    precision, recall, _ = precision_recall_curve(df[true_label],df[pred_label])
    
    avg_recall = average_recall(precision,recall)
    
    ece = expected_calibration_error(df[true_label],df[pred_label])
    mce = max_calibration_error(df[true_label],df[pred_label])
    hlh, pval = hosmer_lemeshow(df[true_label],df[pred_label])
    
    
    return [auc_test,ap_test,avg_recall,ece,mce,hlh,pval]
    
