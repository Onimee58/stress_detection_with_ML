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
import sklearn.metrics as metrics
import os
from scipy import signal
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
path_e4=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\E4_iM_stress\init_processed_e4\window_90'
path_iM=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\E4_iM_stress\init_processed\window_90'

allFiles_e4 = glob.glob(path_e4 + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles_e4:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set_e4 = pd.concat(list_, axis=0, join='outer', ignore_index=False)


allFiles_iM = glob.glob(path_iM + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles_iM:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set_iM = pd.concat(list_, axis=0, join='outer', ignore_index=False)

feature_set_train=feature_set_iM
feature_set_test=feature_set_e4


#
#feature_set_train=pd.concat([stress,nstress],axis=0)


label_2_train=feature_set_train['Label_2']

# value_train=feature_set_train['Value']

# value_train=normalize(value_train)

#
#label_3_train=feature_set_train['Label_3']
#
#label_train=feature_set_train['VALUE']

# X_train=feature_set_train.drop(['ID','Label_2',label,'Value'],axis=1)
# X_train=feature_set_train.drop(['ID','Label_2','Value','Phase'],axis=1)
X_train=feature_set_train.drop(['ID','Label_2','Phase'],axis=1)


list_features=feature_select(X_train,label_2_train)
X_train=X_train[list_features]




label_2_test=feature_set_test['Label_2']
id_test=feature_set_test['ID']
# value_test=feature_set_test['Value']
#label_3_test=feature_set_test['Label_3']
#
#label_test=feature_set_test['VALUE']
# X_test=feature_set_test.drop(['ID','Label_2',label,'Value'],axis=1)
# X_test=feature_set_test.drop(['ID','Label_2','Phase'],axis=1)
X_test=feature_set_test.drop(['ID','Label_2','Value','Phase'],axis=1)
X_test=X_test[list_features]
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


print(len(X_train_new)/len(X_test_new))
# #print(X_train_new)


# E4
model_1=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None, random_state=3)
# model_1=RandomForestClassifier(n_estimators=500,criterion='gini',max_depth=30, random_state=3)
model_2 = KNeighborsClassifier(n_neighbors=18)
model_3 = LogisticRegression(penalty='l2',C=10,random_state=3, solver='lbfgs',multi_class='auto',max_iter =5000)
model_4=svm.SVC(random_state=3,C=10,kernel='poly',gamma='auto')
#model_6=XGBClassifier(learning_rate=0.001,n_estimators=200)
model_5 = AdaBoostClassifier(n_estimators=20, base_estimator=RandomForestClassifier(n_estimators=20,criterion='gini',max_depth=7, random_state=3),random_state=3)
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

plt.plot(label_2_test,'red')
plt.show()
plt.plot(predicted_1_test)
plt.show()

print(roc_auc_score(label_2_test, predicted_1_test))
print(roc_auc_score(label_2_test, predicted_2_test))
print(roc_auc_score(label_2_test, predicted_3_test))
print(roc_auc_score(label_2_test, predicted_4_test))
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



# ROC curve

probs = fit_1.predict_proba(X_test_new)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(label_2_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()





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



# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# subjects = ['Test-1', 'Test-2', 'Test-3', 'Test-4', 'Test-5','Test-6','Test-7','Test-8','Test-9','Test-10']
# weighted_f1 = [0.75,0.95,0.98,1.00,1.00,0.99,1.00,0.97,0.89,0.86]
# ax.bar(subjects,weighted_f1)
# plt.xlabel('Test Subjects')
# plt.ylabel('Weighted F1-score')
# plt.show()


true_values=label_2_test
# print(label_2_test)
predicted_values=predicted_1_test

print(len(true_values))
print(len(predicted_values))


plt.figure(figsize=(16,2))
# plt.plot(true_values,'red',linewidth=3.0,label='True Label')
# plt.step(np.arange(0, len(true_values)), true_values,'red')
plt.step(np.arange(0, len(predicted_values)), predicted_values,'black')
# plt.plot(true_values, signal.square(2 * np.pi * 5 * true_values))
# plt.plot(probs,'blue',linewidth=3.0,label='Predicted Label')
# plt.legend(loc='upper center')
plt.show()

true_values=true_values.to_list()
predicted_values=list(predicted_values)
    
count=0
for i in range(0,len(predicted_values)):
    if(predicted_values[i]!=true_values[i]):
        count=count+1
print('misclassification is',count)















