#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:       generate_cohort.py
Description:    
Author:         Diandra Prioleau Ojo
Date Created:   February 15, 2022
"""

# import packages
import pandas as pd 
import numpy as np
import datetime as dt

# setting path
#sys.path.append('../')
 

def excluded_patients(df,rm_attr_values={}):
    '''
    Removes patients based on specified exclusion criteria based on focus and
    purpose of analysis
    
    df:             dataframe of the dataset
    rm_attr_values: dictionary containing name of attribute/feature as KEY and 
                    list of corresponding values in that attribute for which 
                    rows/instances should be removed as VALUES
                    Ex: dict = {'race': ['ASIAN','WHITE']}
    '''
    temp_df = df.copy()
    
    for key in rm_attr_values.keys():
        temp_df = temp_df.set_index(key)
        values = rm_attr_values[key]
        print(temp_df.index)
        temp_df = temp_df.drop(values)
        print(temp_df.index)
        temp_df = temp_df.reset_index()
        print(len(temp_df))
    return temp_df

def bin_feat_into_groups(df,label,bins=[]):
    return pd.cut(x=df[label], bins=bins)

def compute_length_of_stay(df,admittime_label,dischtime_label):
    # convert to datetime
    df[admittime_label] = pd.to_datetime(df.admittime)
    df[dischtime_label] = pd.to_datetime(df.dischtime)
    
    # compute length of stay (LOS) for each admission
    return (df[dischtime_label] - df[admittime_label]).dt.days

'''Function extracted from
Reference: '''
def convert_icd_9to10(map_df,icd,map_code_col="icd9cm"):
        
    """Function use to apply over the diag DataFrame for ICD9->ICD10 conversion"""
    # If root is true, only map an ICD 9 -> 10 according to the
    # ICD9's root (first 3 digits)
    errors = []
    
    #icd = icd[:3]
    code_cols = map_df.columns
    
    if map_code_col not in code_cols:
        errors.append(f"ICD NOT FOUND: {icd}")
        return np.nan,np.nan

    matches = map_df.loc[map_df[map_code_col] == icd]
    if matches.shape[0] == 0:
        errors.append(f"ICD NOT FOUND: {icd}")
        return np.nan,np.nan
    
    icd10_code = map_df.loc[map_df[map_code_col] == icd].icd10cm.iloc[0]
    icd10_desc = map_df.loc[map_df[map_code_col] == icd].diagnosis_description.iloc[0]
    
    return [icd10_code, icd10_desc]

def icd_9to10(diagnoses_df,code_label,map_df):
    'Code extracted from'
    # Create new column with original codes as default
    col_name = "root_icd10_convert"
    col_desc = "code_description"
    diagnoses_df[col_name] = diagnoses_df["icd_code"].values
    diagnoses_df[col_desc] = ""
    
    # Group identical ICD9 codes, then convert all ICD9 codes within
    # a group to ICD10
    for code, group in diagnoses_df.loc[diagnoses_df.icd_version == 9].groupby(by="icd_code"):
        new_code, code_description = convert_icd_9to10(map_df,code)
        
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            diagnoses_df.at[idx, col_name] = new_code
            diagnoses_df.at[idx, col_desc] = code_description
            
    return diagnoses_df

    
# import DIABETES cohort  
filename = "/Users/diandra/github/MIMIC-IV-Data-Pipeline-main/data/cohort/cohort_non-icu_07_day_readmission_E11.csv"
df = pd.read_csv(filename)

# remove duplicates
df = df.drop_duplicates(subset = ['subject_id', 'hadm_id'],keep = 'last')
#%%
# keep only patients with Black/African American, Hispanic/Latino, or White ethnicity
remove_demo_groups = {'ethnicity':['ASIAN','AMERICAN INDIAN/ALASKA NATIVE','OTHER','UNKNOWN','UNABLE TO OBTAIN']}
df = excluded_patients(df,remove_demo_groups)

# bin age 
bins = [17,64,91]
df['age_binned'] = bin_feat_into_groups(df, 'Age',bins)

# number of prior visits to hospital for diabetes complication or condition
df['num_prior_visits'] = np.zeros(len(df),dtype=int)

# convert admit time to datetime format
df['admittime'] = pd.to_datetime(df.admittime,format="%Y-%m-%d %H:%M:%S")

#%%
for idx in df.index:
    sub_id = df['subject_id'][idx]
    hadm_id = df['hadm_id'][idx]
    admit_time = df['admittime'][idx]
    
    prior_visits =  df[(df.subject_id == sub_id) & (df.hadm_id != hadm_id)
                       & (df.admittime < admit_time)]
    num_prior_visits = len(prior_visits)
    df.at[idx,'num_prior_visits'] = num_prior_visits
   
#%%
# length of stay (LOS) for each admission
df['los'] = compute_length_of_stay(df, 'admittime', 'dischtime')

df = df[df['los'] >= 1]
#%%
df_copy = df
df_copy.set_index('hadm_id',inplace=True)

cohort_size = len(df)

#%%
# import diagnoses_icd file 
filename = "/Users/diandra/github/MIMIC-IV-Data-Pipeline-main/mimic-iv-1.0/hosp/diagnoses_icd.csv"
diag_df = pd.read_csv(filename)

#%%
df_cols = df.columns.tolist()
df = df.merge(diag_df,on='hadm_id',how='left')

#%%
# import mapping file 
mapping_path = "/Users/diandra/github/MIMIC-IV-Data-Pipeline-main/utils/mappings/ICD9_to_ICD10_mapping.txt"

mapping_df = pd.read_table(mapping_path,sep='\t')


#%%
# map ICD-9 to ICD-10 codes
df = icd_9to10(df,"icd9cm",mapping_df)

#%%

#df = df[df.seq_num <= 5]
df = df.groupby('hadm_id', as_index=True)['root_icd10_convert'].agg(lambda x: '; '.join(el for el in x if pd.notna(el)))

# create variable indicating number of diagnoses for each patient (i.e., co-morbidity)
df_copy['num_diseases'] = df.str.count(';') + 1

# create columns based on diagnoses
diag_cols = df.str.get_dummies('; ')

# output diagnoses 
diag_outdf = diag_cols.sum()

diag_outdf = diag_outdf.reset_index()
diag_outdf = diag_outdf.merge(mapping_df.drop_duplicates('icd10cm'),left_on='index',right_on='icd10cm',how='left')[['index',0,'diagnosis_description']]
diag_outdf.to_csv('../data/generated_diagnosis_counts.csv')

# keep 50 most common diagnoses
col_names = diag_cols.sum().nlargest(50).keys().tolist()
diag_cols = diag_cols[col_names]

df = df_copy.merge(diag_cols,left_index=True,right_index=True,how='left')


df.to_csv('../data/generated_cohort_all_diagnoses.csv')

df_copy = df 

cohort_size = len(df)
#%%
# import labevents file 
filename = "/Users/diandra/github/MIMIC-IV-Data-Pipeline-main/mimic-iv-1.0/hosp/labevents.csv"
lab_df = pd.read_csv(filename)
#%%
# import labitems file 
filename = "/Users/diandra/github/MIMIC-IV-Data-Pipeline-main/mimic-iv-1.0/hosp/d_labitems.csv"
labitem_df = pd.read_csv(filename)

# merge labevents with labitems 
lab_df = lab_df.merge(labitem_df,on='itemid',how='left')

#%%

# merge df with labevents 
df = df.merge(lab_df,on='hadm_id',how='left')


#%% 
df.charttime = pd.to_datetime(df.charttime,format="%Y-%m-%d %H:%M:%S")
df.dischtime = pd.to_datetime(df.dischtime,format="%Y-%m-%d %H:%M:%S")
   
df['less_than_48hrs'] = (df.dischtime - df.charttime)/dt.timedelta(hours=1) <= 48

grpby_item = df.groupby('itemid')

#TODO: maybe reduce since increasing missing values in the latter variables 
#%%
# ouput top 20 items based on number of unique hadm_id
top_100_itemid = df.groupby('itemid').nunique()['hadm_id'].nlargest(100)
top_100_itemid.to_csv('../../reports/outputs/top_100_itemid_based_on_unique_hadmid.csv')

grps = top_100_itemid.keys().tolist()
concat_item_grps = pd.concat([grpby_item.get_group(grp) for grp in grps])

median_num_hadmid_per_item = concat_item_grps.groupby('itemid').apply(lambda grp: grp.groupby('hadm_id').size().median()).nlargest(100)
median_num_hadmid_per_item.to_csv('../../reports/outputs/median_num_hadmid_per_item.csv')

mean_num_hadmid_per_item = concat_item_grps.groupby('itemid').apply(lambda grp: grp.groupby('hadm_id').size().mean()).nlargest(100)
mean_num_hadmid_per_item.to_csv('../../reports/outputs/mean_num_hadmid_per_item.csv')
#%%
grpby_item = df[df.less_than_48hrs == True].groupby('itemid')

#%%
for item in grpby_item.nunique()['hadm_id'].nlargest(20).keys().tolist():
    if np.isnan(item):
        continue
    grp = grpby_item.get_group(item)
    
    #if len(grp) < int(cohort_size*0.09):
    #    continue
    
    
    min_vals = []
    max_vals = []
    #median_vals = []
    #mean_vals = []
    
    
    for admit in df_copy.index.tolist():
        admission = grp[grp['hadm_id'] == admit]
        
        
        #chart_time = pd.to_datetime(df.charttime,format="%Y-%m-%d %H:%M:%S")
        #discharge_time = pd.to_datetime(df.dischtime,format="%Y-%m-%d %H:%M:%S")
        
        #admission['less_than_24hrs'] = (discharge_time - chart_time)/dt.timedelta(hours=1)
        
        # create lab events 48 hours before discharge
        admission = admission[admission['less_than_48hrs'] == True]
        
        if len(admission) == 0 or admission.isnull().valuenum.any():
            agg_min = grp.ref_range_lower.min()
            agg_max = grp.ref_range_upper.max()
        else: 
            agg_min = np.min(admission.valuenum)
            agg_max = np.max(admission.valuenum)
            #agg_med = np.median(admission.valuenum)
            #agg_mean = np.mean(admission.valuenum)


        min_vals.append(agg_min)
        max_vals.append(agg_max)
        #median_vals.append(agg_med)
        #mean_vals.append(agg_mean)
    
    df_copy[str(item)+'_min'] = min_vals
    df_copy[str(item)+'_max'] = max_vals
    #df_copy[str(item)+'_med'] = median_vals
    #df_copy[str(item)+'_mean'] = mean_vals

#%%
df_copy = df_copy.reset_index()

# one-hot encoding of categorical variables 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

ct = make_column_transformer((OneHotEncoder(), ['insurance', 'admission_type']),    
                             remainder='passthrough')

df_transformed = ct.fit_transform(df_copy)
df_transformed = pd.DataFrame(df_transformed, columns=ct.get_feature_names())

y = df_transformed.label
X = df_transformed.drop(columns=['label'])
#%%

# 50-25-25 stratified train-test split 
# Note: stratified by outcome label, race, gender, and age 

# import train-test-split model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,
                                                    stratify=pd.concat([X[['ethnicity','gender','age_binned']], y],axis=1),
                                                    random_state=12)

X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.3,
                                                    stratify=pd.concat([X_train[['ethnicity','gender','age_binned']], y_train],axis=1),
                                                    random_state=12)

# save train set
train = pd.concat([X_train,y_train],axis=1)
train.to_csv('../data/train_data.csv')

# save validation set
validation = pd.concat([X_valid,y_valid],axis=1)
validation.to_csv('../data/validation_data.csv')

# save test set
test = pd.concat([X_test,y_test],axis=1)
test.to_csv('../data/test_data.csv')

#%%
# size of dataset
print('Size of cohort: {}'.format(len(df_copy)))

# unique patients 
print('Number of patients: {}'.format(df_copy.subject_id.nunique()))

# unique admissions 
unique_admit = len(pd.unique(df_copy[ 'hadm_id']))
print('Unique number of admissions: {}'.format(unique_admit))
