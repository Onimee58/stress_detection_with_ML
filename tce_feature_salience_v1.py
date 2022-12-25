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
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif
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
import os
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import matthews_corrcoef

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

def select_k_best_2(X_train,Y_train,k):
    list_index=[]
    feature_name=[]
    column_names=X_train.columns
    for column in column_names:
        corr, _ = pearsonr(X_train[column], labels)
        if(corr>=0.1 or corr<-0.1):
            feature_name.append(column)
    X_train=X_train[feature_name]
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
        if(corr>=0.05 or corr<-0.05):
            feature_name.append(column)
    X_train=X_train[feature_name]
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


# path e4
#path =r'C:\Users\rajde\Documents\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\processed_30_stress'
# path imotion
path =r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA'

allFiles = glob.glob(path + "/*.csv")
#print(allFiles)
list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)

feature_set = pd.concat(list_, axis=0, join='outer', ignore_index=False)


train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
test_id= [41, 10, 44, 13, 14, 45, 49, 21, 25, 26, 31]

feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]

cortisol_value=feature_set_train['Value']
labels=feature_set_train['Label_2']

# 



feature_set_train=feature_set_train.drop(['Value','Phase','ID','Label_2'],axis=1)
column_names=feature_set_train.columns

correlation_index=[]
feature_name=[]

# for column in column_names:
#     corr, _ = pearsonr(feature_set_train[column], labels)
#     # corr=matthews_corrcoef(feature_set_train[column], cortisol_value)
#     if(corr>=0.1 or corr<-0.1):
#         correlation_index.append(corr)
#         feature_name.append(column)
        
for column in column_names:
    corr, _ = pearsonr(feature_set_train[column], labels)
    # corr=matthews_corrcoef(feature_set_train[column], cortisol_value)
    correlation_index.append(corr)
    feature_name.append(column)
    
dataframe_feature=pd.DataFrame({
    'Feature':feature_name,
    'Pearson':correlation_index})

print(dataframe_feature)
    
# dataframe_feature.to_csv('feature_correlation.csv')


# feature_set_train_new=feature_set_train[feature_name]

# print(feature_set_train_new)


# selected_feature=select_k_best(feature_set_train_new,labels,13)

# print(selected_feature)

# list_selected=select_k_best_2(feature_set_train,labels,13)

# feature_set_train=feature_set_train.values
# feature_set_train=pd.DataFrame.from_records(feature_set_train)

# feature_set_train_new=feature_set_train[list_selected]

# print(feature_set_train_new)






