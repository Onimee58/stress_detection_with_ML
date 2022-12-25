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
# def feature_select(X_train,Y_train):
#     feature_names=[]
#     column_names=X_train.columns
#     for column in column_names:
#         score,p_value=pearsonr(X_train[column],Y_train)
#         # print(p_value)
#         if(p_value<0.05):
#             feature_names.append(column)
#     return feature_names

# path e4
#path =r'C:\Users\rajde\Documents\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\processed_30_stress'
# path imotion
# path =r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\processed_30'
# Single sensor
path_eda=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA'
path_bvp=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\BVP'
path_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\ACC'
path_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\ST'
# Dual Sensor
path_eda_bvp=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP'
path_eda_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_ACC'
path_eda_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_ST'
path_bvp_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\BVP_ST'
path_bvp_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\BVP_ACC'
path_acc_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\ACC_ST'
# Tri sensor
path_bvp_st_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\BVP_ST_ACC'
path_eda_bvp_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ACC'
path_eda_bvp_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST'
path_eda_st_acc=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_ST_ACC'
# all
path_eda_bvp_acc_st=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_ACC'
path_eda_bvp_st_ibi=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_IBI'
path_eda_bvp_st_ibi_30=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_IBI_30'
path_eda_bvp_st_ibi_60=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_IBI_60'
path_eda_bvp_st_ibi_120=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP_ST_IBI_120'
# allFiles = glob.glob(path_eda_bvp_st_ibi + "/*.csv")
allFiles = glob.glob(path_eda_bvp_st_ibi + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set = pd.concat(list_, axis=0, join='outer', ignore_index=False)



# test_id=[1,18,19,14]
# # Train
# train_id=[4,5,6,7,10,11,12,13,15,16,17,20,21,22,23]

# test_id=[1,11,19,14]
# # Train
# train_id=[4,5,6,7,10,12,13,15,16,17,18,20,21,22,23]



# test_id=[1,10,19,14]
# # Train
# train_id=[4,5,6,7,11,12,13,15,16,17,18,20,21,22,23]
#
#ids=[1,4,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,20,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
#
#ids=[1,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,20,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]
#train_id=np.random.choice(ids, 36, replace=False)
# test=14, train=31
#test_id=[33,35,4,5,38,7,39,10,44,21,23,26,27,29]
#train_id=[1,6,11,12,13,14,15,16,17,18,19,20,22,25,28,30,31,32,34,36,37,40,41,42,43,45,46,47,48,49,50]
# E4 based generated ids optimal
# train_id=[4, 10, 12, 13, 14, 15,16, 18, 19, 20, 21, 23, 25, 29, 30, 31, 32, 33, 34, 35, 36, 38, 42, 43, 45, 46, 47, 48, 49, 50]
# test_id=[1, 37, 40, 41, 11,44, 17, 22, 26, 27, 28]



# E4 three level
# train_id=[1, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 30, 31, 32, 34, 35, 36, 37, 42, 44, 45, 47, 48] 
# test_id=[33, 38, 40, 41, 43, 46, 49, 50, 20, 26, 29]

# train_id=[4, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 25, 29, 30, 31, 32, 33, 34, 35, 36, 38, 42, 43, 45, 46, 47, 48, 49, 50]
# test_id=[1, 37, 40, 41, 11, 44, 17, 22, 26, 27, 28]

# iMotion generated optimum train test
#train_id=[4, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 25, 26, 28, 29, 31, 33, 34, 36, 38, 40, 41, 43, 44, 46, 47, 48, 49, 50]
#test_id=[32, 1, 35, 37, 42, 12, 45, 19, 23, 27, 30]

# E4 test
# test_id=[33,35,4,38,10,44,21,23,26,27,29]
# train_id=[1,11,12,13,14,15,16,17,18,19,20,22,25,28,30,31,32,34,36,37,40,41,42,43,45,46,47,48,49,50]

# test=9, train=36
#test_id=[32,5,38,39,19,22,23,25,30]
#train_id=[1,4,6,7,10,11,12,13,14,15,16,17,18,19,20,21,26,27,28,29,31,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,50]

# Concept_test
#train_id=[4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28,30,31,32,33,34,37,40,41,42,45,46,47,48,50]
#test_id=[1,35,36,38,43,44,49,25,26,27,29]

# New E4 train_test
# train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
# test_id= [41,10,44,13,14,45, 49, 21, 25,26, 31]

# New E4 new feature
# train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
# test_id= [41,10,44,13,14, 49, 21, 25,26, 31]
# test_id= [31]

#subject_id=[1,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,24,25,26,27,28,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50]

# optimum test train only stress for E4
# train_id=[4, 10, 11, 14, 16, 17, 18, 19, 20, 21, 23, 25, 27, 28, 29, 30, 31, 34, 35, 37, 38, 40, 41, 43, 45, 46, 47, 48, 49, 50]
# test_id=[32, 1, 33, 36, 42, 12, 13, 44, 15, 22, 26]

# optimum train test for iMotion
# train_id=[1, 10, 13, 14, 16, 17, 18, 19, 20, 22, 26, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 49, 50]
# test_id=[35, 4, 11, 12, 44, 15, 48, 21, 23, 25, 27]
# Common
# train_id=[4, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 23, 26, 27, 28, 29, 30, 31, 33, 35, 37, 38, 40, 41, 43, 45, 46, 47, 49, 50]
# test_id=[32, 1, 34, 36, 42, 12, 44, 14, 48, 22, 25]

# with relax new E4
# train_id= [4, 10, 11, 14, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 47, 48, 49, 50]
# test_id = [1, 38, 40, 12, 13, 46, 15, 21, 26, 27, 31]


# new set

# train_ids=[[4, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 41, 42, 47, 48, 49, 50], [1, 4, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 47, 48, 50], [1, 4, 10, 11, 14, 15, 16, 19, 20, 22, 23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 42, 44, 46, 47, 48, 50], [1, 4, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 25, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 43, 44, 47, 48, 49, 50], [1, 4, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 25, 28, 29, 30, 31, 32, 33, 34, 37, 40, 42, 43, 44, 46, 47, 48, 49, 50], [1, 4, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 23, 26, 27, 28, 30, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 48, 49, 50], [1, 4, 10, 11, 12, 14, 16, 17, 18, 19, 20, 23, 25, 27, 28, 29, 30, 32, 34, 35, 36, 37, 40, 42, 43, 44, 47, 48, 49, 50], [1, 4, 10, 11, 12, 16, 18, 20, 21, 22, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 40, 42, 43, 44, 46, 47, 48, 50], [4, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 42, 43, 44, 47, 48, 49, 50], [1, 4, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 30, 32, 33, 34, 35, 36, 38, 40, 42, 43, 44, 47, 49, 50], [1, 4, 10, 11, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 41, 42, 43, 47, 48, 50], [1, 4, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 27, 28, 30, 31, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 47, 50], [1, 4, 10, 11, 12, 14, 15, 16, 18, 20, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 41, 42, 43, 47, 48, 50], [1, 4, 10, 11, 13, 14, 15, 16, 18, 20, 21, 22, 23, 26, 27, 30, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44, 47, 48, 50], [1, 4, 10, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 34, 36, 40, 41, 43, 44, 46, 47, 48, 49, 50]]
# print(len(train_ids))
# test_ids=[[1, 36, 38, 40, 43, 44, 46, 17, 19, 25], [38, 40, 10, 44, 13, 46, 15, 49, 26, 29], [38, 41, 43, 12, 13, 17, 18, 49, 21, 27], [38, 40, 42, 46, 15, 17, 21, 23, 26, 28], [35, 36, 38, 41, 10, 13, 17, 21, 26, 27], [33, 38, 46, 47, 17, 19, 22, 25, 29, 31], [33, 38, 41, 13, 46, 15, 21, 22, 26, 31], [38, 41, 13, 14, 15, 17, 49, 19, 25, 30], [1, 37, 38, 40, 41, 10, 46, 15, 29, 31], [37, 41, 12, 46, 48, 23, 25, 26, 29, 31], [33, 38, 40, 12, 13, 44, 15, 46, 49, 21], [33, 38, 12, 46, 48, 49, 22, 23, 26, 29], [33, 35, 44, 13, 46, 17, 49, 19, 21, 23], [36, 40, 12, 46, 17, 49, 19, 25, 28, 29], [33, 35, 37, 38, 42, 14, 15, 17, 25, 26]]
# print(len(test_ids))

# train_id=train_ids[4]
# test_id=test_ids[4]



train_id=[1, 4, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 27, 28, 30, 31, 32, 33, 34, 35, 36, 40, 42, 43, 44, 47, 48, 50]
test_id=[37, 38, 41, 14, 46, 17, 49, 25, 26, 29]
# test_id=[41]

# train_id=[1, 4, 10, 11, 12, 14, 15, 16, 18, 19, 20, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 47, 48, 50]
# test_id=[13, 46, 17, 49, 21, 22, 23, 25, 26, 29]

eda_feature_list=['MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MAX_PEAK_GSR','MIN_PEAK_GSR','MEAN_PEAK_WIDTH',
                  'MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','RMS_PEAK_WIDTH','MAX_PEAK_WIDTH','MIN_PEAK_WIDTH','MEAN_PEAK_PROM','MEDIAN_PEAK_PROM',
                  'STD_PEAK_PROM','RMS_PEAK_PROM','MAX_PEAK_PROM','MIN_PEAK_PROM']
bvp_feature_list=['MEAN_PPG','STD_PPG','RMS_PPG','RANGE_PPG','MEAN_PPG_WIDTH','MEDIAN_PPG_WIDTH','STD_PPG_WIDTH','RMS_PPG_WIDTH','MAX_PPG_WIDTH',
                  'MIN_PPG_WIDTH','MEAN_PPG_PROM','MEDIAN_PPG_PROM','STD_PPG_PROM','RMS_PPG_PROM','MAX_PPG_PROM','MIN_PPG_PROM','HR']
hrv_feature_list=['HRV','RANGE_HRV','HRV_STD','HRV_RMS']
st_feature_list=['ST_MEAN','ST_SD','ST_MEDIAN','ST_RMS','ST_RANGE','ST_SLOPE','ST_INTERCEPT']
ibi_feature_list=['MEAN_IBI','SD_IBI','MEDIAN_IBI','MAX_IBI','MIN_IBI','RMS_IBI']


eda_new=['MIN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','MAX_PEAK_PROM','STD_PEAK_PROM','MIN_PEAK_PROM',
         'MEAN_PEAK_WIDTH','RMS_PEAK_WIDTH', 'MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','MAX_PEAK_WIDTH']

# list_features_original=eda_feature_list
# print(len(list_features_original))
list_features_original=eda_feature_list+bvp_feature_list+ibi_feature_list+st_feature_list
k=29

feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]



label='Phase'
#label='Label_3'


#
#feature_set_train=pd.concat([stress,nstress],axis=0)


label_2_train=feature_set_train['Label_2']

value_train=feature_set_train['Value']

value_train=normalize(value_train)

#
#label_3_train=feature_set_train['Label_3']
#
#label_train=feature_set_train['VALUE']

# X_train=feature_set_train.drop(['ID','Label_2',label,'Value'],axis=1)
X_train=feature_set_train.drop(['ID','Label_2','Value',label],axis=1)
X_train=X_train[list_features_original]
feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]
label_2_test=feature_set_test['Label_2']
id_test=feature_set_test['ID']
phase_test=feature_set_test[label]
X_test=feature_set_test.drop(['ID','Label_2','Value',label],axis=1)
X_test=X_test[list_features_original]

values=X_train.values
#values = values.reshape((len(values), 1))
#values=normalize(values)
scaler_feature = StandardScaler()
scaler_feature = scaler_feature.fit(values)
X_train_new=scaler_feature.transform(X_train)
X_test_new=scaler_feature.transform(X_test)   

X_train_new=pd.DataFrame.from_records(X_train_new)
X_test_new=pd.DataFrame.from_records(X_test_new)

X_train_new.columns=list_features_original
X_test_new.columns=list_features_original

print(X_train_new.shape)
print(X_test_new.shape)

X_train_eda=X_train_new[eda_feature_list]
X_train_bvp=X_train_new[bvp_feature_list]
X_train_ibi=X_train_new[ibi_feature_list]
X_train_st=X_train_new[st_feature_list]

X_test_eda=X_test_new[eda_feature_list]
X_test_bvp=X_test_new[bvp_feature_list]
X_test_ibi=X_test_new[ibi_feature_list]
X_test_st=X_test_new[st_feature_list]

selected_eda=feature_select(X_train_eda,label_2_train)
selected_bvp=feature_select(X_train_bvp,label_2_train)
selected_ibi=feature_select(X_train_ibi,label_2_train)
selected_st=feature_select(X_train_st,label_2_train)

X_train_eda_new=X_train_eda[selected_eda]
X_train_bvp_new=X_train_bvp[selected_bvp]
X_train_ibi_new=X_train_ibi[selected_ibi]
X_train_st_new=X_train_st[selected_st]

X_test_eda_new=X_test_eda[selected_eda]
X_test_bvp_new=X_test_bvp[selected_bvp]
X_test_ibi_new=X_test_ibi[selected_ibi]
X_test_st_new=X_test_st[selected_st]

cv_matrix_eda=PCA(n_components = X_test_eda_new.shape[1])
cv_matrix_bvp=PCA(n_components = X_test_bvp_new.shape[1])
cv_matrix_ibi=PCA(n_components = X_test_ibi_new.shape[1])
cv_matrix_st=PCA(n_components = X_test_st_new.shape[1])

# cv_matrix_eda=PCA(n_components = 8)
# cv_matrix_bvp=PCA(n_components = 9)
# cv_matrix_ibi=PCA(n_components = 2)
# cv_matrix_st=PCA(n_components = 2)

cv_matrix_eda.fit(X_test_eda_new)
cv_matrix_bvp.fit(X_test_bvp_new)
cv_matrix_ibi.fit(X_test_ibi_new)
cv_matrix_st.fit(X_test_st_new)

variance_eda = cv_matrix_eda.explained_variance_ratio_
variance_bvp = cv_matrix_bvp.explained_variance_ratio_
variance_ibi = cv_matrix_ibi.explained_variance_ratio_
variance_st = cv_matrix_st.explained_variance_ratio_

var_eda=np.cumsum(np.round(cv_matrix_eda.explained_variance_ratio_, decimals=3)*100)
var_bvp=np.cumsum(np.round(cv_matrix_bvp.explained_variance_ratio_, decimals=3)*100)
var_ibi=np.cumsum(np.round(cv_matrix_ibi.explained_variance_ratio_, decimals=3)*100)
var_st=np.cumsum(np.round(cv_matrix_st.explained_variance_ratio_, decimals=3)*100)


plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,110)
plt.style.context('seaborn-whitegrid')

x_axis=[1,2,3,4,5,6,7,8,9,10,11]

x_axis_ibi=[1,2]
x_axis_st=[1,2,3]
# plt.figure(figsize=(12,6))
# plt.plot(var_eda,'red',linewidth=3.0,markevery=[11],marker="o",label='EDA')
# plt.plot(var_bvp,'blue',linewidth=3.0,markevery=[11],marker="x",label='BVP')
# plt.plot(var_ibi,'green',linewidth=3.0,markevery=[2],marker="o",label='IBI')
# plt.plot(var_st,'violet',linewidth=3.0,markevery=[3],marker="x",label='ST')
# plt.legend(loc='lower right') 

plt.figure(figsize=(12,6))
plt.plot(x_axis,var_eda,'red',linewidth=3.0,label='EDA')
plt.plot(x_axis,var_bvp,'blue',linewidth=3.0,label='BVP')
plt.plot(x_axis_ibi,var_ibi,'green',linewidth=3.0,label='IBI')
plt.plot(x_axis_st,var_st,'violet',linewidth=3.0,label='ST')
plt.legend(loc='lower right') 



# X_train_eda_1=cv_matrix_eda.fit_transform(X_train_eda_new)
# X_train_bvp_1=cv_matrix_bvp.fit_transform(X_train_bvp_new)
# X_train_ibi_1=cv_matrix_ibi.fit_transform(X_train_ibi_new)
# X_train_st_1=cv_matrix_st.fit_transform(X_train_st_new)

# X_test_eda_1=cv_matrix_eda.fit_transform(X_test_eda_new)
# X_test_bvp_1=cv_matrix_bvp.fit_transform(X_test_bvp_new)
# X_test_ibi_1=cv_matrix_ibi.fit_transform(X_test_ibi_new)
# X_test_st_1=cv_matrix_st.fit_transform(X_test_st_new)

# X_train_eda_1=pd.DataFrame.from_records(X_train_eda_1)
# X_train_bvp_1=pd.DataFrame.from_records(X_train_bvp_1)
# X_train_ibi_1=pd.DataFrame.from_records(X_train_ibi_1)
# X_train_st_1=pd.DataFrame.from_records(X_train_st_1)

# X_train_new_1=pd.concat([X_train_eda_1,X_train_bvp_1,X_train_ibi_1,X_train_st_1],axis=1)

# X_test_eda_1=pd.DataFrame.from_records(X_test_eda_1)
# X_test_bvp_1=pd.DataFrame.from_records(X_test_bvp_1)
# X_test_ibi_1=pd.DataFrame.from_records(X_test_ibi_1)
# X_test_st_1=pd.DataFrame.from_records(X_test_st_1)

# X_test_new_1=pd.concat([X_test_eda_1,X_test_bvp_1,X_test_ibi_1,X_test_st_1],axis=1)





# # covar_matrix = PCA(n_components = list_shape[1])
# covar_matrix = PCA(n_components = X_train_st_new.shape[1])

# covar_matrix.fit(X_train_st_new)
# variance = covar_matrix.explained_variance_ratio_

# var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)

# print(var)

# plt.ylabel('% Variance Explained')
# plt.xlabel('# of Features')
# plt.title('PCA Analysis')
# plt.ylim(30,100.5)
# plt.style.context('seaborn-whitegrid')

# plt.plot(var)
    

# # X_train_new=pd.DataFrame.from_records(X_train_new)


# # print(pd.DataFrame(covar_matrix.components_,columns=X_train_new.columns))














