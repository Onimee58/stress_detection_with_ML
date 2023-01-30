# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:24:47 2019

@author: rajde
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2
from scipy import stats
import sklearn
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import time
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split,KFold,RandomizedSearchCV,GridSearchCV
from sklearn.utils import shuffle
import warnings
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,mutual_info_regression
warnings.filterwarnings('always')
from collections import Counter
from statistics import mode 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import math  
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import ttest_ind
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

def sine_normalize(dataframe):
    dataframe_values=dataframe.values
    sined_values=[]
    for i in dataframe_values:
        value_sine=math.sin(i)
        sined_values.append(value_sine)
    return sined_values

def select_k_best(X_train,Y_train,k):
    list_index=[]
    selector=SelectKBest(mutual_info_regression,k='all').fit(X_train,Y_train)
    scores=selector.scores_
    scores=scores.tolist()
    sorted_score=sorted(scores,reverse=True)
    duplicates=set([x for x in sorted_score if sorted_score.count(x) > 1])
    for i in duplicates:
        sorted_score.remove(i)
    for i in range(k):
        list_index.append(scores.index(sorted_score[i]))
    return list_index
def generate_train_test(subject_id):
#    subject_id=np.random.choice(subject_id,40,replace=False)
    train_id=np.random.choice(subject_id,30,replace=False)
    train_id=set(train_id)
    subject_id=set(subject_id)
    test_id=subject_id-train_id
    train_id=list(train_id)
    test_id=list(test_id)
    return train_id,test_id
def find_significant_feature(X_train,X_test):
    col=list(X_train)
    pvalue=[]
    for i in col:
        first=X_train[i]
        second=X_test[i]
        values=stats.kruskal(first,second)
        p_value=values[1]
        pvalue.append(p_value)
    return pvalue
def normalize(dataframe):
    minimum=np.min(dataframe)
    maximum=np.max(dataframe)
    dataframe=(dataframe-minimum)/(maximum-minimum)
    return dataframe

def select_k_best_2(X_train,Y_train,k):
    list_index=[]
    feature_name=[]
    column_names=X_train.columns
    for column in column_names:
        corr, _ = pearsonr(X_train[column], Y_train)
        if(corr>=0.02 or corr<=-0.02):
            feature_name.append(column)
    X_train=X_train[feature_name]
    selector=SelectKBest(f_classif,k='all').fit(X_train,Y_train)
    scores=selector.scores_
    scores=scores.tolist()
    sorted_score=sorted(scores,reverse=True)
    duplicates=set([x for x in sorted_score if sorted_score.count(x) > 1])
    for i in duplicates:
        sorted_score.remove(i)
    for i in range(k):
        list_index.append(scores.index(sorted_score[i]))
    return list_index

def select_k_best_3(X_train,Y_train):
    # list_index=[]
    corr_index=[]
    feature_name=[]
    column_names=X_train.columns
    for column in column_names:
        corr, _ = spearmanr(X_train[column], Y_train)
        corr_index.append(abs(corr))
        feature_name.append(column)
    df_feature=pd.DataFrame({'Feature':feature_name,
                             'Corr':corr_index})
    df_feature_sorted=df_feature.sort_values('Corr',ascending=False)
    # print(df_feature_sorted)
    feature_names=df_feature_sorted['Feature']
    feature_names=feature_names.reset_index()
    feature_names=feature_names.drop(['index'],axis=1)
    feature_names=feature_names['Feature'].to_list()
    # for i in range(k):
    #     list_index.append(feature_names[i])
    return feature_names

def return_k_best(feature_names,k):
    list_names=[]
    # print(feature_names)
    for i in range(k):
        list_names.append(feature_names[i])
    return list_names

def feature_select(X_train,Y_train):
    feature_names=[]
    column_names=X_train.columns
    for column in column_names:
        score,p_value=stats.kendalltau(X_train[column],Y_train,method='auto')
        if(p_value<0.05):
            feature_names.append(column)
    return feature_names

def scaler_fit(X_train,X_test):
    values=X_train.values
    scaler_feature = StandardScaler()
    scaler_feature = scaler_feature.fit(values)
    X_train_new=scaler_feature.transform(X_train)
    X_test_new=scaler_feature.transform(X_test) 
    return X_train_new,X_test_new

path_eda_bvp_st_ibi=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_IBI'

allFiles = glob.glob(path_eda_bvp_st_ibi + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set = pd.concat(list_, axis=0, join='outer', ignore_index=False)

# train_id=[1, 4, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 27, 28, 30, 31, 32, 33, 34, 35, 36, 40, 42, 43, 44, 47, 48, 50]
# test_id=[37, 38, 41, 14, 46, 17, 49, 25, 26, 29]
# Some new try
train_id=[1, 4, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 27, 28, 30, 31, 32, 33, 34, 35, 36, 40, 42, 43, 44, 47, 48, 50]
test_id=[37, 38, 41, 14, 46, 17, 49, 25, 26, 29]

eda_feature_list=['MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MAX_PEAK_GSR','MIN_PEAK_GSR','MEAN_PEAK_WIDTH',
                  'MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','RMS_PEAK_WIDTH','MAX_PEAK_WIDTH','MIN_PEAK_WIDTH','MEAN_PEAK_PROM','MEDIAN_PEAK_PROM',
                  'STD_PEAK_PROM','RMS_PEAK_PROM','MAX_PEAK_PROM','MIN_PEAK_PROM']
bvp_feature_list=['MEAN_PPG','STD_PPG','RMS_PPG','RANGE_PPG','MEAN_PPG_WIDTH','MEDIAN_PPG_WIDTH','STD_PPG_WIDTH','RMS_PPG_WIDTH','MAX_PPG_WIDTH',
                  'MIN_PPG_WIDTH','MEAN_PPG_PROM','MEDIAN_PPG_PROM','STD_PPG_PROM','RMS_PPG_PROM','MAX_PPG_PROM','MIN_PPG_PROM','HR']
st_feature_list=['ST_MEAN','ST_SD','ST_MEDIAN','ST_RMS','ST_RANGE','ST_SLOPE','ST_INTERCEPT']
ibi_feature_list=['MEAN_IBI','SD_IBI','MEDIAN_IBI','MAX_IBI','MIN_IBI','RMS_IBI']

feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]
feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]

label_2_train=feature_set_train['Label_2']
label_2_test=feature_set_test['Label_2']

X_train=feature_set_train.drop(['ID','Label_2','Value','Phase'],axis=1)   
X_test=feature_set_test.drop(['ID','Label_2','Value','Phase'],axis=1)  

X_train_eda=X_train[eda_feature_list]
X_train_eda_bvp=X_train[eda_feature_list+bvp_feature_list]
X_train_eda_bvp_ibi=X_train[eda_feature_list+bvp_feature_list+ibi_feature_list]
X_train_eda_bvp_ibi_st=X_train[eda_feature_list+bvp_feature_list+ibi_feature_list+st_feature_list]

X_test_eda=X_test[eda_feature_list]
X_test_eda_bvp=X_test[eda_feature_list+bvp_feature_list]
X_test_eda_bvp_ibi=X_test[eda_feature_list+bvp_feature_list+ibi_feature_list]
X_test_eda_bvp_ibi_st=X_test[eda_feature_list+bvp_feature_list+ibi_feature_list+st_feature_list]

list_feature_eda=feature_select(X_train_eda,label_2_train)
list_feature_eda_bvp=feature_select(X_train_eda_bvp,label_2_train)
list_feature_eda_bvp_ibi=feature_select(X_train_eda_bvp_ibi,label_2_train)
list_feature_eda_bvp_ibi_st=feature_select(X_train_eda_bvp_ibi_st,label_2_train)

X_train_eda_new=X_train_eda[list_feature_eda]
X_train_eda_bvp_new=X_train_eda_bvp[list_feature_eda_bvp]
X_train_eda_bvp_ibi_new=X_train_eda_bvp_ibi[list_feature_eda_bvp_ibi]
X_train_eda_bvp_ibi_st_new=X_train_eda_bvp_ibi_st[list_feature_eda_bvp_ibi_st]

X_test_eda_new=X_test_eda[list_feature_eda]
X_test_eda_bvp_new=X_test_eda_bvp[list_feature_eda_bvp]
X_test_eda_bvp_ibi_new=X_test_eda_bvp_ibi[list_feature_eda_bvp_ibi]
X_test_eda_bvp_ibi_st_new=X_test_eda_bvp_ibi_st[list_feature_eda_bvp_ibi_st]


 


X_train_eda_new,X_test_eda_new=scaler_fit(X_train_eda_new,X_test_eda_new)
X_train_eda_bvp_new,X_test_eda_bvp_new=scaler_fit(X_train_eda_bvp_new,X_test_eda_bvp_new)
X_train_eda_bvp_ibi_new,X_test_eda_bvp_ibi_new=scaler_fit(X_train_eda_bvp_ibi_new,X_test_eda_bvp_ibi_new)
X_train_eda_bvp_ibi_st_new,X_test_eda_bvp_ibi_st_new=scaler_fit(X_train_eda_bvp_ibi_st_new,X_test_eda_bvp_ibi_st_new)

model_eda=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=3)
model_eda_bvp=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=3)
model_eda_bvp_ibi=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=3)
model_eda_bvp_ibi_st=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=3)


fit_eda=model_eda.fit(X_train_eda_new,label_2_train)
fit_eda_bvp=model_eda_bvp.fit(X_train_eda_bvp_new,label_2_train)
fit_eda_bvp_ibi=model_eda_bvp_ibi.fit(X_train_eda_bvp_ibi_new,label_2_train)
fit_eda_bvp_ibi_st=model_eda_bvp_ibi_st.fit(X_train_eda_bvp_ibi_st_new,label_2_train)

# # EDA
# probs_eda = fit_eda.predict_proba(X_test_eda_new)
# preds_eda = probs_eda[:,1]
# fpr_eda, tpr_eda, threshold_eda = metrics.roc_curve(label_2_test, preds_eda)
# roc_auc_eda = metrics.auc(fpr_eda, tpr_eda)
# # EDA_BVP
# probs_eda_bvp = fit_eda_bvp.predict_proba(X_test_eda_bvp_new)
# preds_eda_bvp = probs_eda_bvp[:,1]
# fpr_eda_bvp, tpr_eda_bvp, threshold_eda_bvp = metrics.roc_curve(label_2_test, preds_eda_bvp)
# roc_auc_eda_bvp = metrics.auc(fpr_eda_bvp, tpr_eda_bvp)   
# # EDA_BVP_IBI
# probs_eda_bvp_ibi = fit_eda_bvp_ibi.predict_proba(X_test_eda_bvp_ibi_new)
# preds_eda_bvp_ibi = probs_eda_bvp_ibi[:,1]
# fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, threshold_eda_bvp_ibi = metrics.roc_curve(label_2_test, preds_eda_bvp_ibi)
# roc_auc_eda_bvp_ibi = metrics.auc(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi)   
# # EDA_BVP_IBI_ST
# probs_eda_bvp_ibi_st = fit_eda_bvp_ibi_st.predict_proba(X_test_eda_bvp_ibi_st_new)
# preds_eda_bvp_ibi_st = probs_eda_bvp_ibi_st[:,1]
# fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, threshold_eda_bvp_ibi_st = metrics.roc_curve(label_2_test, preds_eda_bvp_ibi_st)
# roc_auc_eda_bvp_ibi_st = metrics.auc(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr_eda, tpr_eda, label = 'EDA = %0.2f' % roc_auc_eda)
# plt.plot(fpr_eda_bvp, tpr_eda_bvp, label = 'EDA,BVP = %0.2f' % roc_auc_eda_bvp)
# plt.plot(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, label = 'EDA,BVP,IBI = %0.2f' % roc_auc_eda_bvp_ibi)
# plt.plot(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, label = 'EDA,BVP,IBI,ST = %0.2f' % roc_auc_eda_bvp_ibi_st)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
  
    
    
# # EDA
# probs_eda = fit_eda.predict_proba(X_test_eda_new)
# predicted_eda=fit_eda.predict(X_test_eda_new)
# preds_eda = probs_eda[:,1]
# fpr_eda, tpr_eda, threshold_eda = metrics.roc_curve(label_2_test, predicted_eda)
# roc_auc_eda = metrics.auc(fpr_eda, tpr_eda)
# # EDA_BVP
# probs_eda_bvp = fit_eda_bvp.predict_proba(X_test_eda_bvp_new)
# predicted_eda_bvp=fit_eda_bvp.predict(X_test_eda_bvp_new)
# preds_eda_bvp = probs_eda_bvp[:,1]
# fpr_eda_bvp, tpr_eda_bvp, threshold_eda_bvp = metrics.roc_curve(label_2_test, predicted_eda_bvp)
# roc_auc_eda_bvp = metrics.auc(fpr_eda_bvp, tpr_eda_bvp)   
# # EDA_BVP_IBI
# probs_eda_bvp_ibi = fit_eda_bvp_ibi.predict_proba(X_test_eda_bvp_ibi_new)
# predicted_eda_bvp_ibi=fit_eda_bvp_ibi.predict(X_test_eda_bvp_ibi_new)
# preds_eda_bvp_ibi = probs_eda_bvp_ibi[:,1]
# fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, threshold_eda_bvp_ibi = metrics.roc_curve(label_2_test, preds_eda_bvp_ibi)
# roc_auc_eda_bvp_ibi = metrics.auc(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi)   
# # EDA_BVP_IBI_ST
# probs_eda_bvp_ibi_st = fit_eda_bvp_ibi_st.predict_proba(X_test_eda_bvp_ibi_st_new)
# predicted_eda_bvp_ibi_st=fit_eda_bvp_ibi_st.predict(X_test_eda_bvp_ibi_st_new)
# preds_eda_bvp_ibi_st = probs_eda_bvp_ibi_st[:,1]
# fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, threshold_eda_bvp_ibi_st = metrics.roc_curve(label_2_test, predicted_eda_bvp_ibi_st)
# roc_auc_eda_bvp_ibi_st = metrics.auc(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr_eda, tpr_eda, label = 'EDA = %0.2f' % roc_auc_eda)
# plt.plot(fpr_eda_bvp, tpr_eda_bvp, label = 'EDA,BVP = %0.2f' % roc_auc_eda_bvp)
# plt.plot(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, label = 'EDA,BVP,IBI = %0.2f' % roc_auc_eda_bvp_ibi)
# plt.plot(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, label = 'EDA,BVP,IBI,ST = %0.2f' % roc_auc_eda_bvp_ibi_st)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
    
    

# EDA
probs_eda = fit_eda.predict_log_proba(X_test_eda_new)
predicted_eda=fit_eda.predict(X_test_eda_new)
preds_eda = probs_eda[:,1]

preds_eda[np.isinf(preds_eda)] = 0.0
# preds_eda=predicted_eda
fpr_eda, tpr_eda, threshold_eda = metrics.roc_curve(label_2_test, preds_eda, drop_intermediate=False)


roc_auc_eda = metrics.auc(fpr_eda, tpr_eda)
# EDA_BVP
probs_eda_bvp = fit_eda_bvp.predict_log_proba(X_test_eda_bvp_new)
predicted_eda_bvp=fit_eda_bvp.predict(X_test_eda_bvp_new)
preds_eda_bvp = probs_eda_bvp[:,1]
preds_eda_bvp[np.isinf(preds_eda_bvp)] = 0.0
# preds_eda_bvp=predicted_eda_bvp
fpr_eda_bvp, tpr_eda_bvp, threshold_eda_bvp = metrics.roc_curve(label_2_test, preds_eda_bvp, drop_intermediate=False)
roc_auc_eda_bvp = metrics.auc(fpr_eda_bvp, tpr_eda_bvp)   
# EDA_BVP_IBI
probs_eda_bvp_ibi = fit_eda_bvp_ibi.predict_log_proba(X_test_eda_bvp_ibi_new)
predicted_eda_bvp_ibi=fit_eda_bvp_ibi.predict(X_test_eda_bvp_ibi_new)
preds_eda_bvp_ibi = probs_eda_bvp_ibi[:,1]
# preds_eda_bvp_ibi=predicted_eda_bvp_ibi
preds_eda_bvp_ibi[np.isinf(preds_eda_bvp_ibi)] = 0.0
fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, threshold_eda_bvp_ibi = metrics.roc_curve(label_2_test, preds_eda_bvp_ibi, drop_intermediate=False)
roc_auc_eda_bvp_ibi = metrics.auc(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi)   
# EDA_BVP_IBI_ST
probs_eda_bvp_ibi_st = fit_eda_bvp_ibi_st.predict_log_proba(X_test_eda_bvp_ibi_st_new)
predicted_eda_bvp_ibi_st=fit_eda_bvp_ibi_st.predict(X_test_eda_bvp_ibi_st_new)
preds_eda_bvp_ibi_st = probs_eda_bvp_ibi_st[:,1]
# preds_eda_bvp_ibi_st=predicted_eda_bvp_ibi_st
preds_eda_bvp_ibi_st[np.isinf(preds_eda_bvp_ibi_st)] = 0.0
fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, threshold_eda_bvp_ibi_st = metrics.roc_curve(label_2_test, preds_eda_bvp_ibi_st, drop_intermediate=False)
roc_auc_eda_bvp_ibi_st = metrics.auc(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st)

plt.title('Receiver Operating Characteristic')
plt.figure(figsize=(16,8))
plt.plot(fpr_eda,tpr_eda,linewidth=3.0,  label = 'EDA = %0.2f' % roc_auc_eda)
plt.plot(fpr_eda_bvp, tpr_eda_bvp,linewidth=3.0, label = 'EDA,BVP = %0.2f' % roc_auc_eda_bvp)
plt.plot(fpr_eda_bvp_ibi, tpr_eda_bvp_ibi, linewidth=3.0,label = 'EDA,BVP,IBI = %0.2f' % roc_auc_eda_bvp_ibi)
plt.plot(fpr_eda_bvp_ibi_st, tpr_eda_bvp_ibi_st, linewidth=3.0,label = 'EDA,BVP,IBI,ST = %0.2f' % roc_auc_eda_bvp_ibi_st)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
    
    
# metrics.plot_roc_curve(fit_eda, X_test_eda_new,label_2_test)  # doctest: +SKIP
# plt.show()          
    
    
    
    
    
    
    
    
    
    
    