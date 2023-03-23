#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       preliminary_study.py
Description:    Create and save plots 
Author:         Diandra Prioleau Ojo
Date Created:   February 22, 2022
"""
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.calibration import calibration_curve
from matplotlib.lines import Line2D

def plot_roc_curve(y_true,y_pred,model_name,plot_title="Receiver operating characteristic curve"
                   ,filename=None):
    mean_fpr = np.linspace(0,1,100)
    fpr, tpr, t = roc_curve(y_true,y_pred)
    tpr = np.interp(mean_fpr, fpr, tpr)
    auc_score = roc_auc_score(y_true,y_pred)
    
    fig, ax = plt.subplots()
    ax.plot(
                mean_fpr,
                tpr,
                color='b',
                label=r"%s Overall ROC (AUC = %0.2f)" % (str(model_name),auc_score),
                lw=2,
                alpha=0.8)
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel='FPR',
        ylabel='TPR',
        title=str(plot_title),
        )
    ax.legend(loc="lower right")
    
    if filename != None:
        plt.savefig(filename)
    plt.show()
    
def plot_pr_curve(y_true,y_pred,model_name,plot_title="Precision-recall curve"
                   ,filename=None):
    mean_recall = np.linspace(0,1,100)
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    precision = np.interp(mean_recall,precision,recall)
    ap = average_precision_score(y_true,y_pred,average='samples')
 
    
    fig, ax = plt.subplots()
    ax.plot(
                mean_recall,
                precision,
                color='r',
                label=r"%s Overall AP = %0.2f" % (str(model_name),ap),
                lw=2,
                alpha=0.8)
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel='Recall',
        ylabel='Precision',
        title="Precision-recall curve: test set",
        )
    ax.legend(loc="lower right")
    plt.savefig(filename)
    plt.show()

def plot_calibration_curve(y_true,y_pred,num_bins,model_name,plot_title='Calibration Plot',
                           filename=None):
    prob_true, prob_pred = calibration_curve(y_true, y_pred,n_bins=num_bins)
    fig, ax = plt.subplots()
    
    ax.plot(prob_pred,
            prob_true,
            marker='o',
            linewidth=1,
            label=str(model_name))
   
    line = Line2D([0,1],[0,1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    
    ax.set(
        xlim=[-0.05,1.05],
        ylim=[-0.05,1.05],
        xlabel='Predicted probability',
        ylabel='True probability',
        title=str(plot_title))
    
    ax.legend(loc="lower right")
    
    if filename != None:
        plt.savefig(filename)
        
    plt.show()
    
def plot_calibration_by_subgroup(df,subgroup_cols,true_label,pred_label,num_bins,
                                 plot_title='Calibration Plot',filename=None):
    fig,ax = plt.subplots()

    groups = df.groupby(subgroup_cols) 
    for name, grp in groups:
        y_true = grp[true_label]
        y_proba = grp[pred_label]
        
        prob_true, prob_pred = calibration_curve(y_true, y_proba,n_bins=num_bins)
        
        label = str(name)
        ax.plot(
                prob_pred,
                prob_true,
                marker='o',
                label=label,
                linewidth=1)  
    
    line = Line2D([0,1],[0,1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    
    ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel='Predicted probability',
            ylabel='True probability',
            title=str(plot_title))  
    
    ax.legend(loc="upper left")
    
    if filename != None:
        plt.savefig(filename)
        
    plt.show()