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
import statistics as st
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
    selector=SelectKBest(f_classif,k=k).fit(X_train,Y_train)
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
        if(corr>=0.03 or corr<=-0.03):
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

allFiles = glob.glob(path_eda_bvp + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set = pd.concat(list_, axis=0, join='outer', ignore_index=False)

# New E4 train_test
train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
test_id= [41, 10, 44, 13, 14, 45, 49, 21, 25, 26, 31]


feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]
labels=feature_set_train['Label_2']
feature_set_train_new=feature_set_train.drop(['ID','Label_2','Phase','Value'],axis=1)

stressed=feature_set_train.loc[feature_set_train['Label_2'] == 1]

not_stressed=feature_set_train.loc[feature_set_train['Label_2'] == 0]


feature_names=select_k_best_3(feature_set_train_new,labels)


maximum_list=[]
maximum_index_list=[]
for i in range(10):
    stressed_1=stressed.sample(n=50)
    non_stressed_1=not_stressed.sample(n=50)

    valid_set=pd.concat([stressed_1,non_stressed_1],axis=0,ignore_index=False)


    train_set = feature_set_train[feature_set_train.index.isin(valid_set.index) == False]


    valid_y=valid_set['Label_2']
    valid_x=valid_set.drop(['ID','Label_2','Phase','Value'],axis=1)
    
    train_y=train_set['Label_2']
    train_x=train_set.drop(['ID','Label_2','Phase','Value'],axis=1)
    model_1=RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=30, random_state=3)
    macro_f1=[]
    for k in range(1,39):
        list_features=[]
        list_features=return_k_best(feature_names,k)
        train_x_new=train_x[list_features]
        valid_x_new=valid_x[list_features]
        values=train_x_new.values
        scaler_feature = StandardScaler()
        scaler_feature = scaler_feature.fit(values)
        train_x_new=scaler_feature.transform(train_x_new)  
        valid_x_new=scaler_feature.transform(valid_x_new)
        train_x=pd.DataFrame.from_records(train_x)
        valid_x=pd.DataFrame.from_records(valid_x)
        fit_1=model_1.fit(train_x_new,train_y)
        predicted_1=fit_1.predict(valid_x_new)
        macro_f1.append(f1_score(valid_y, predicted_1, average='macro'))
    maximum_list.append(max(macro_f1))
    maximum_index_list.append(macro_f1.index(max(macro_f1)))

print(np.max(maximum_list))
print(np.min(maximum_list))
# print(st.mode(maximum_index_list))
print(np.min(maximum_index_list))

# print(macro_f1)

print('maximum accuracy',max(maximum_list))
print('minimum accuracy',min(maximum_list))
print('mean accuracy',np.mean(maximum_list))
print('mean index',np.mean(maximum_index_list))
# print('most common index',st.mode(maximum_index_list))
print('minimum index',np.min(maximum_index_list))
print('maximum index',max(maximum_index_list))








