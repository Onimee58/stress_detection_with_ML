# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:40:33 2023

@author: monim
"""

from final_functions_context import *
from get_cortisol_gt_context import subject_id, all_labels, time_slots, sub_id
from get_cortisol_gt_context import good_subject_id

from tqdm import tqdm


eda_feature_list=['MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MAX_PEAK_GSR','MIN_PEAK_GSR',
                  'MEAN_PEAK_WIDTH','MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','RMS_PEAK_WIDTH','MAX_PEAK_WIDTH',
                  'MIN_PEAK_WIDTH','MEAN_PEAK_PROM','MEDIAN_PEAK_PROM','STD_PEAK_PROM','RMS_PEAK_PROM',
                  'MAX_PEAK_PROM','MIN_PEAK_PROM']

bvp_feature_list=['MEAN_PPG','STD_PPG','RMS_PPG','RANGE_PPG','MEAN_PPG_WIDTH','MEDIAN_PPG_WIDTH','STD_PPG_WIDTH',
                  'RMS_PPG_WIDTH','MAX_PPG_WIDTH','MIN_PPG_WIDTH','MEAN_PPG_PROM','MEDIAN_PPG_PROM','STD_PPG_PROM',
                  'RMS_PPG_PROM','MAX_PPG_PROM','MIN_PPG_PROM','HR', 'AVG_RESP_FREQUENCY','MAX_RESP_FREQUENCY']

hrv_feature_list=['HRV','RANGE_HRV','HRV_STD','HRV_RMS']

st_feature_list=['ST_MEAN','ST_SD','ST_MEDIAN','ST_RMS','ST_RANGE','ST_SLOPE','ST_INTERCEPT']

ibi_feature_list=['MEAN_IBI','SD_IBI','MEDIAN_IBI','MAX_IBI','MIN_IBI','RMS_IBI']

# eda_new=['MIN_PEAK_GSR','STD_PEAK_GSR','RMS_PEAK_GSR','MEAN_PEAK_GSR','MEDIAN_PEAK_GSR','MAX_PEAK_PROM','STD_PEAK_PROM','MIN_PEAK_PROM',
#          'MEAN_PEAK_WIDTH','RMS_PEAK_WIDTH', 'MEDIAN_PEAK_WIDTH','STD_PEAK_WIDTH','MAX_PEAK_WIDTH']

eda_only = eda_feature_list
eda_and_bvp = eda_feature_list + bvp_feature_list
eda_and_bvp_ibi = eda_and_bvp + ibi_feature_list
eda_and_bvp_ibi_and_st = eda_and_bvp_ibi + st_feature_list

print('Select mode:\n1.  EDA only \n2.  EDA and BVP \n3.  EDA, BVP and IBI \n4.  EDA, BVP, IBI and ST')

feature_name = input()


if feature_name == '1':
    selected_feature = eda_only
elif feature_name == '2':
    selected_feature = eda_and_bvp
elif feature_name == '2':
    selected_feature = eda_and_bvp_ibi  
else:
    selected_feature = eda_and_bvp_ibi_and_st

others = ['ID', 'Labels']

# print(len(list_features_original))

k=29

df = pd.read_csv('dataset/all_features_context.csv',index_col=None, header=0)
all_features = df.loc[:, ~df.columns.str.contains('^Unnamed')]

all_accuracy = []
all_precision_recall_fscore_support = []
macro_rf_f1 = []
micro_rf_f1 = []
macro_knn_f1 = []
micro_knn_f1 = []
macro_lr_f1 = []
micro_lr_f1 = []
macro_svm_f1 = []
micro_svm_f1 = []

# all_id = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
                # 31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]
    
all_id = [p for p in subject_id]

model_1 = RandomForestClassifier(n_estimators=40,criterion='entropy',max_features=None,
                                 max_depth=None, random_state=3, n_jobs=-1, warm_start=False, class_weight='balanced_subsample')
model_2 = KNeighborsClassifier(n_neighbors=18)
model_3 = LogisticRegression(penalty='l2',C=10,random_state=3, solver='lbfgs',multi_class='auto',max_iter =5000)
model_4 = svm.SVC(random_state=3,C=10,kernel='poly',gamma='auto')
    

for i in tqdm(range(0, len(all_id))):
    
    # all_id = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
    #             31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]
    
    all_id = [p for p in subject_id]
    
    print('\n\n')
    print('--------------- Trial ' + str(i+1) + ' ---------------' )
    test_id = [all_id.pop(i)]
    # test_id.append(all_id.pop(i-1))
    train_id = all_id
    
    print('train_id: ', end = '')
    print(train_id)
    print('test_id: ', end = '')
    print(test_id)
    
    feature_set = all_features[others+selected_feature]
    feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]
    feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]

    
    label_2_train=feature_set_train['Labels']
    X_train=feature_set_train.drop(['ID','Labels'],axis=1)
    list_features=feature_select(X_train,label_2_train)
    X_train=X_train[list_features]
    
    label_2_test=feature_set_test['Labels']
    id_test=feature_set_test['ID']
    X_test=feature_set_test.drop(['ID','Labels'],axis=1)
    X_test=X_test[list_features]
    
    values=X_train.values
    scaler_feature = StandardScaler()
    scaler_feature = scaler_feature.fit(values)
    X_train_new=scaler_feature.transform(X_train)
    X_test_new=scaler_feature.transform(X_test)   
    
    
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
    
    for j in range(len(tests)):
        metric = sklearn.metrics.classification_report(label_2_test,tests[j])
        all_accuracy.append(sklearn.metrics.accuracy_score(label_2_test,tests[j]))
        all_precision_recall_fscore_support.append(sklearn.metrics.precision_recall_fscore_support(label_2_test,tests[j]))
        print('---------------' + models[j] + '---------------' )
        print(metric)
    
    
    
    
    
    macro_rf_f1.append(f1_score(label_2_test, predicted_1_test, average='micro'))
    micro_rf_f1.append(f1_score(label_2_test, predicted_1_test, average='macro'))
    macro_knn_f1.append(f1_score(label_2_test, predicted_2_test, average='micro'))
    micro_knn_f1.append(f1_score(label_2_test, predicted_2_test, average='macro'))
    macro_lr_f1.append(f1_score(label_2_test, predicted_3_test, average='micro'))
    micro_lr_f1.append(f1_score(label_2_test, predicted_3_test, average='macro'))
    macro_svm_f1.append(f1_score(label_2_test, predicted_4_test, average='micro'))
    micro_svm_f1.append(f1_score(label_2_test, predicted_4_test, average='macro'))


avg_macro_rf_f1 = np.mean(np.array(macro_rf_f1))
avg_micro_rf_f1 = np.mean(np.array(micro_rf_f1))
avg_macro_knn_f1 = np.mean(np.array(macro_knn_f1))
avg_micro_knn_f1 = np.mean(np.array(micro_knn_f1))
avg_macro_lr_f1 = np.mean(np.array(macro_lr_f1))
avg_micro_lr_f1 = np.mean(np.array(micro_lr_f1))
avg_macro_svm_f1 = np.mean(np.array(macro_svm_f1))
avg_micro_svm_f1 = np.mean(np.array(micro_svm_f1))
avg_acc = np.mean(np.array(all_accuracy))

print('avg_macro_rf_f1', avg_macro_rf_f1)
print('avg_micro_rf_f1', avg_micro_rf_f1)
print('avg_macro_knn_f1', avg_macro_knn_f1)
print('avg_micro_knn_f1', avg_micro_knn_f1)
print('avg_macro_lr_f1', avg_macro_lr_f1)
print('avg_micro_lr_f1', avg_micro_lr_f1)
print('avg_macro_svm_f1', avg_macro_svm_f1)
print('avg_micro_svm_f1', avg_micro_svm_f1)
print('avg_acc', avg_acc)












