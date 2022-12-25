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
train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
test_id= [41,10,44,13,14, 49, 21, 25,26, 31]
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

eda_feature_list=['MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MAX_PEAK_GSR','MIN_PEAK_GSR','MEAN_PEAK_WIDTH',
                  'MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','RMS_PEAK_WIDTH','MAX_PEAK_WIDTH','MIN_PEAK_WIDTH','MEAN_PEAK_PROM','MEDIAN_PEAK_PROM',
                  'STD_PEAK_PROM','RMS_PEAK_PROM','MAX_PEAK_PROM','MIN_PEAK_PROM']
bvp_feature_list=['MEAN_PPG','STD_PPG','RMS_PPG','RANGE_PPG','MEAN_PPG_WIDTH','MEDIAN_PPG_WIDTH','STD_PPG_WIDTH','RMS_PPG_WIDTH','MAX_PPG_WIDTH',
                  'MIN_PPG_WIDTH','MEAN_PPG_PROM','MEDIAN_PPG_PROM','STD_PPG_PROM','RMS_PPG_PROM','MAX_PPG_PROM','MIN_PPG_PROM','HR']
hrv_feature_list=['HRV','RANGE_HRV','HRV_STD','HRV_RMS']
st_feature_list=['ST_MEAN','ST_SD','ST_MEDIAN','ST_RMS','ST_RANGE','ST_SLOPE','ST_INTERCEPT','ST_LB_SLOPE','ST_UB_SLOPE']
ibi_feature_list=['MEAN_IBI','SD_IBI','MEDIAN_IBI','MAX_IBI','MIN_IBI','RMS_IBI','SLOPE_IBI','INTERCEPT_IBI','LB_SLOPE_IBI','UB_SLOPE_IBI']

# list_features_original=eda_feature_list+bvp_feature_list+st_feature_list
list_features_original=eda_feature_list+bvp_feature_list+st_feature_list
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
# print(X_train)
#list_features=select_k_best(X_train,label_2_train,5)
#print(list_features)


#print(len(X_train))
# Test

feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]


label_2_test=feature_set_test['Label_2']
id_test=feature_set_test['ID']
phase_test=feature_set_test[label]
# value_test=feature_set_test['Value']
#label_3_test=feature_set_test['Label_3']
#
#label_test=feature_set_test['VALUE']
# X_test=feature_set_test.drop(['ID','Label_2',label,'Value'],axis=1)
X_test=feature_set_test.drop(['ID','Label_2','Value',label],axis=1)
X_test=X_test[list_features_original]
#X_test=X_test[list_features]

# Standardize
#list_features=find_significant_feature(X_train,X_test)
#print(list_features) 

values=X_train.values
#values = values.reshape((len(values), 1))
#values=normalize(values)
scaler_feature = StandardScaler()
scaler_feature = scaler_feature.fit(values)
X_train_new=scaler_feature.transform(X_train)
X_test_new=scaler_feature.transform(X_test)   

X_train_new=pd.DataFrame.from_records(X_train_new)

#X_train_new=X_train_new[list_features]

#print(X_train_new)
# feature_names=select_k_best_3(X_train_new,label_2_train)
# list_features=return_k_best(feature_names,k)
# print(list_features)

list_features=select_k_best_2(X_train_new,label_2_train,k)
# list_features=select_k_best_3(X_train_new,label_2_train)
print(len(list_features))
print(list_features)

X_train_new=X_train_new[list_features]
X_test_new=pd.DataFrame.from_records(X_test_new)

X_test_new=X_test_new[list_features]

print(len(X_train_new)/len(X_test_new))
#print(X_train_new)

# pca = PCA(n_components=34,svd_solver='arpack')

# pca.fit(X_train_new)
# X_train_new=pca.transform(X_train_new)
# X_test_new=pca.transform(X_test_new)
# print(pca.explained_variance_ratio_)

#train_X, valid_X, train_y, valid_y = train_test_split(X_train_new, label_2_train, test_size=0.2, random_state=42)
# E4
model_1=RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=50, random_state=3)
model_2 = KNeighborsClassifier(n_neighbors=25)
model_3 = LogisticRegression(random_state=3, solver='lbfgs',multi_class='multinomial',max_iter =100000)
model_4=svm.SVC(random_state=3,C=10,kernel='poly',gamma='auto')
#model_6=XGBClassifier(learning_rate=0.001,n_estimators=200)
model_5 = AdaBoostClassifier(n_estimators=200, base_estimator=RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=50, random_state=2),random_state=2)
#model_7 = LinearDiscriminantAnalysis()
# iMotion
#model_1=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=2)


# train_X, valid_X, train_y, valid_y = train_test_split(X_train_new, label_2_train, test_size=0.2, random_state=42)


fit_1=model_1.fit(X_train_new,label_2_train)


#print(fit_1.oob_score_)
fit_2=model_2.fit(X_train_new,label_2_train)
fit_3=model_3.fit(X_train_new,label_2_train)
fit_4=model_4.fit(X_train_new,label_2_train)

# scores = cross_val_score(model_1, X_train_new, label_2_train, cv=10)
# print(scores)
# Validation

# predicted_1=fit_1.predict(valid_X)
# predicted_2=fit_2.predict(valid_X)
# predicted_3=fit_3.predict(valid_X)
# predicted_4=fit_4.predict(valid_X)
# #predicted_5=fit_5.predict(X_test_new)

# print('micro average random forest',f1_score(valid_y, predicted_1, average='micro'))
# print('macro average random forest',f1_score(valid_y, predicted_1, average='macro'))
# print('micro average knn',f1_score(valid_y, predicted_2, average='micro'))
# print('macro average knn',f1_score(valid_y, predicted_2, average='macro'))
# print('micro average logistic',f1_score(valid_y, predicted_3, average='micro'))
# print('macro average logistic',f1_score(valid_y, predicted_3, average='macro'))
# print('micro average svm',f1_score(valid_y, predicted_4, average='micro'))
# print('macro average svm',f1_score(valid_y, predicted_4, average='macro'))

# Testing

predicted_1_test=fit_1.predict(X_test_new)
predicted_2_test=fit_2.predict(X_test_new)
predicted_3_test=fit_3.predict(X_test_new)
predicted_4_test=fit_4.predict(X_test_new)
#predicted_5=fit_5.predict(X_test_new)


# print(roc_auc_score(label_2_test, predicted_1_test))
# print(roc_auc_score(label_2_test, predicted_2_test))
# print(roc_auc_score(label_2_test, predicted_3_test))
# print(roc_auc_score(label_2_test, predicted_4_test))
#accuracy_feature=accuracy_score(label_2_test,predicted_1)
#print(accuracy_feature)
metric=sklearn.metrics.classification_report(label_2_test,predicted_1_test)
print(metric)


print('micro average random forest',f1_score(label_2_test, predicted_1_test, average='micro'))
print('macro average random forest',f1_score(label_2_test, predicted_1_test, average='macro'))
print('micro average knn',f1_score(label_2_test, predicted_2_test, average='micro'))
print('macro average knn',f1_score(label_2_test, predicted_2_test, average='macro'))
print('micro average logistic',f1_score(label_2_test, predicted_3_test, average='micro'))
print('macro average logistic',f1_score(label_2_test, predicted_3_test, average='macro'))
print('micro average svm',f1_score(label_2_test, predicted_4_test, average='micro'))
print('macro average svm',f1_score(label_2_test, predicted_4_test, average='macro'))






#model_2 = KNeighborsClassifier(n_neighbors=25)
#model_3 = LogisticRegression(random_state=2, solver='lbfgs',multi_class='multinomial',max_iter =10000)
#model_4=svm.SVC(kernel='poly',gamma='auto')
#model_6=XGBClassifier(learning_rate=0.001,n_estimators=200)
#model_5 = AdaBoostClassifier(n_estimators=100, base_estimator=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=30, random_state=2),random_state=2)
#model_7 = LinearDiscriminantAnalysis()


# fit_1=model_1.fit(X_train_new,label_2_train)
# #print(fit_1.oob_score_)
# fit_2=model_2.fit(X_train_new,label_2_train)
# fit_3=model_3.fit(X_train_new,label_2_train)
# fit_4=model_4.fit(X_train_new,label_2_train)
# #fit_5=model_5.fit(X_train_new,label_2_train)



# predicted_1=fit_1.predict(X_test_new)
# predicted_2=fit_2.predict(X_test_new)
# predicted_3=fit_3.predict(X_test_new)
# predicted_4=fit_4.predict(X_test_new)
# #predicted_5=fit_5.predict(X_test_new)

# print('Accuracy for random forest is',accuracy_score(label_2_test,predicted_1))
# print('Accuracy for knn is',accuracy_score(label_2_test,predicted_2))
# print('Accuracy for lr is',accuracy_score(label_2_test,predicted_3))
# print('Accuracy for svm is',accuracy_score(label_2_test,predicted_4))
# #print('Accuracy for knn is',accuracy_score(label_2_test,predicted_2))
# #print('Accuracy for LR is',accuracy_score(label_2_test,predicted_3))
# #print('Accuracy for SVM is',accuracy_score(label_2_test,predicted_4))
# #print('Accuracy for XGBoost is',accuracy_score(label_2_test,predicted_5))
# #predictions = fit_1.predict_proba(X_test_new)

# #prediction_cascaded=cascaded_model(X_train_new,label_2_train,X_test_new)
# print(roc_auc_score(label_2_test, predicted_1))
# print(roc_auc_score(label_2_test, predicted_2))
# print(roc_auc_score(label_2_test, predicted_3))
# print(roc_auc_score(label_2_test, predicted_4))
# #accuracy_feature=accuracy_score(label_2_test,predicted_1)
# #print(accuracy_feature)
# metric=sklearn.metrics.classification_report(label_2_test,predicted_3)
# print(metric)


# print('micro average random forest',f1_score(label_2_test, predicted_1, average='micro'))
# print('macro average random forest',f1_score(label_2_test, predicted_1, average='macro'))
# print('micro average knn',f1_score(label_2_test, predicted_2, average='micro'))
# print('macro average knn',f1_score(label_2_test, predicted_2, average='macro'))
# print('micro average logistic',f1_score(label_2_test, predicted_3, average='micro'))
# print('macro average logistic',f1_score(label_2_test, predicted_3, average='macro'))
# print('micro average svm',f1_score(label_2_test, predicted_4, average='micro'))
# print('macro average svm',f1_score(label_2_test, predicted_4, average='macro'))



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
subjects = ['Test-1', 'Test-2', 'Test-3', 'Test-4', 'Test-5','Test-6','Test-7','Test-8','Test-9','Test-10']
weighted_f1 = [0.75,0.95,0.98,1.00,1.00,0.99,1.00,0.97,0.89,0.86]
ax.bar(subjects,weighted_f1)
plt.xlabel('Test Subjects')
plt.ylabel('Weighted F1-score')
plt.show()



    


















