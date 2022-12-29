# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:35:59 2022

@author: Saif
"""

from final_functions import *

train_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
            31,32,33,34,35,36,37,38,40,41,42,43,44,45,47,48,49,50]
# test_id=[37, 38, 41, 14, 46, 17, 49, 25, 26, 29]
test_id=[46]

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

eda_only = eda_feature_list
eda_and_bvp = eda_feature_list + bvp_feature_list
eda_and_bvp_ibi = eda_and_bvp + ibi_feature_list
eda_and_bvp_ibi_and_st = eda_and_bvp_ibi + st_feature_list

print('Select mode:\n1.  EDA only \n2.  EDA and BVP \n3.  EDA, BVP and IBI \n4.  EDA, BVp, IBI and ST')

feature_name = input()


if feature_name == '1':
    selected_feature = eda_only
elif feature_name == '2':
    selected_feature = eda_and_bvp
elif feature_name == '2':
    selected_feature = eda_and_bvp_ibi  
else:
    selected_feature = eda_and_bvp_ibi_and_st

others = ['ID', 'Label_2', 'Phase', 'Value']

# print(len(list_features_original))

k=29
df = pd.read_csv('dataset/all_features.csv',index_col=None, header=0)
all_features = df.loc[:, ~df.columns.str.contains('^Unnamed')]

feature_set = all_features[others+selected_feature]
feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]
feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]


label='Phase'

label_2_train=feature_set_train['Label_2']
value_train=feature_set_train['Value']
value_train=normalize(value_train)
X_train=feature_set_train.drop(['ID','Label_2','Value',label],axis=1)
list_features=feature_select(X_train,label_2_train)
X_train=X_train[list_features]

label_2_test=feature_set_test['Label_2']
id_test=feature_set_test['ID']
phase_test=feature_set_test[label]
X_test=feature_set_test.drop(['ID','Label_2','Value',label],axis=1)
X_test=X_test[list_features]

values=X_train.values
scaler_feature = StandardScaler()
scaler_feature = scaler_feature.fit(values)
X_train_new=scaler_feature.transform(X_train)
X_test_new=scaler_feature.transform(X_test)   


model_1=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=None, random_state=3)
model_2 = KNeighborsClassifier(n_neighbors=18)
model_3 = LogisticRegression(penalty='l2',C=10,random_state=3, solver='lbfgs',multi_class='auto',max_iter =5000)
model_4=svm.SVC(random_state=3,C=10,kernel='poly',gamma='auto')


fit_1=model_1.fit(X_train_new,label_2_train)
fit_2=model_2.fit(X_train_new,label_2_train)
fit_3=model_3.fit(X_train_new,label_2_train)
fit_4=model_4.fit(X_train_new,label_2_train)


predicted_1_test=fit_1.predict(X_test_new)
predicted_2_test=fit_2.predict(X_test_new)
predicted_3_test=fit_3.predict(X_test_new)
predicted_4_test=fit_4.predict(X_test_new)

models = ['Random Forest', 'KNN', 'Logistic Regression', 'Support Vector Machine']
tests = [predicted_1_test, predicted_2_test, predicted_3_test, predicted_4_test]

for i in range(len(tests)):
    metric=sklearn.metrics.classification_report(label_2_test,tests[i])
    print('---------------' + models[i] + '---------------' )
    print(metric)

print('micro average random forest',f1_score(label_2_test, predicted_1_test, average='micro'))
print('macro average random forest',f1_score(label_2_test, predicted_1_test, average='macro'))
print('micro average knn',f1_score(label_2_test, predicted_2_test, average='micro'))
print('macro average knn',f1_score(label_2_test, predicted_2_test, average='macro'))
print('micro average logistic',f1_score(label_2_test, predicted_3_test, average='micro'))
print('macro average logistic',f1_score(label_2_test, predicted_3_test, average='macro'))
print('micro average svm',f1_score(label_2_test, predicted_4_test, average='micro'))
print('macro average svm',f1_score(label_2_test, predicted_4_test, average='macro'))


















