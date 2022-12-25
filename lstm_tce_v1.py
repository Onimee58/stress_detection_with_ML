# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:33:13 2019

@author: rajde
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D, BatchNormalization, GRU
import keras
from keras import regularizers
from sklearn.feature_selection import SelectKBest,f_classif
from keras.layers.advanced_activations import LeakyReLU
import glob
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import sklearn
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import load_model
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr,spearmanr

def select_k_best(X_train,Y_train,k):
    list_index=[]
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


# E4
path_1 =r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\Stress Detection\Affective Computing\Research\Processed'

path_eda_bvp=r'C:\Users\rajde\Documents\Research_LAB\RESEARCH_WORKS\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed_e4\EDA_BVP'

allFiles = glob.glob(path_eda_bvp + "/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    list_.append(df)
feature_set = pd.concat(list_, axis=0, join='outer', ignore_index=False)

# feature_set=feature_set.sample(frac=1)
# feature_set_e4=feature_set_e4.drop(feature_set_e4.columns[0], axis=1)
# iMotion
# path_2=r'C:\Users\rajde\Documents\RESEARCH_WORKS\On-going Work\ICCE_Extension\init_processed\win_30_stress'
# allFiles = glob.glob(path_2 + "/*.csv")
# list_ = []
# for file_ in allFiles:
#     df = pd.read_csv(file_,index_col=None, header=0)
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     list_.append(df)
# feature_set_iM = pd.concat(list_, axis=0, join='outer', ignore_index=False)

#feature_set=pd.concat([feature_set_e4,feature_set_iM],axis=0)

#train, test = train_test_split(df, test_size=0.2)
# test=14, train=31
#test_id=[33,35,4,5,38,7,39,10,44,21,23,26,27,29]
#train_id=[1,6,11,12,13,14,15,16,17,18,19,20,22,25,28,30,31,32,34,36,37,40,41,42,43,45,46,47,48,49,50]

# train_id=[4, 10, 11, 14, 16, 17, 18, 19, 20, 21, 23, 25, 27, 28, 29, 30, 31, 34, 35, 37, 38, 40, 41, 43, 45, 46, 47, 48, 49, 50]
# test_id=[32, 1, 33, 36, 42, 12, 13, 44, 15, 22, 26]

#
#test_id=[1,12,19,14,31]
# Train
#train_id=[4,5,6,7,10,11,13,15,16,17,18,20,21,22,23]

# test_id=[1,10,19,14]
# # Train
# train_id=[4,5,6,7,11,12,13,15,16,17,18,20,21,22,23]

# test_id=[1,18,19,14]
# # Train
# train_id=[4,5,6,7,10,11,12,13,15,16,17,20,21,22,23]

# test_id=[1,19,15,14]
# # Train
# train_id=[4,5,6,7,10,11,12,13,16,17,18,20,21,22,23]


train_id= [1, 4, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 42, 43, 46, 47, 48, 50]
test_id= [41, 10, 44, 13, 14, 45, 49, 21, 25, 26, 31]

#print(len(train_id))
feature_set_train=feature_set.loc[feature_set['ID'].isin(train_id)]

feature_set_train=feature_set_train.reset_index()
feature_set_train=feature_set_train.drop(['index'],axis=1)


stressed=feature_set_train.loc[feature_set_train['Label_2'] == 1]

not_stressed=feature_set_train.loc[feature_set_train['Label_2'] == 0]


stressed_1=stressed.sample(n=50)
non_stressed_1=not_stressed.sample(n=50)

valid_set=pd.concat([stressed_1,non_stressed_1],axis=0,ignore_index=False)


train_set = feature_set_train[feature_set_train.index.isin(valid_set.index) == False]





valid_y=valid_set['Label_2']
valid_x=valid_set.drop(['ID','Label_2','Phase','Value'],axis=1)

train_y=train_set['Label_2']
train_x=train_set.drop(['ID','Label_2','Phase','Value'],axis=1)



# Test

feature_set_test=feature_set.loc[feature_set['ID'].isin(test_id)]

label_2_test=feature_set_test['Label_2']
#label_test=feature_set_test['VALUE']
label_3_test=feature_set_test['Phase']
# X_test=feature_set_test.drop(['ID','Label_2','Phase','Value'],axis=1)
X_test=feature_set_test.drop(['ID','Label_2','Phase','Value'],axis=1)
#print(feature_set)
#df=pd.read_csv('features.csv')
k=32

col_names=list(train_x)
#X_train=np.asarray(X_train)

values=train_x.values
scaler_feature = StandardScaler()
scaler_feature = scaler_feature.fit(values)
train_x=scaler_feature.transform(train_x)
X_test=scaler_feature.transform(X_test)   
valid_x=scaler_feature.transform(valid_x)

# scaler = MinMaxScaler(feature_range=(0, 1))
# fit=scaler.fit(X_train)
# X_train=fit.transform(X_train)
# X_test=fit.transform(X_test)   
X_train_new=pd.DataFrame.from_records(train_x)
list_features=select_k_best(X_train_new,train_y,k)

# feature_names=select_k_best_3(X_train_new,train_y)
# list_features=return_k_best(feature_names,k)

#print(list_features)
X_train_new=X_train_new[list_features]
#print(X_train_new)
X_train_new=np.asarray(X_train_new)
X_train_new=X_train_new.reshape(len(X_train_new),1,k)



# X_train_new=pd.DataFrame.from_records(train_x)
# list_features=select_k_best(X_train_new,train_y,k)
# #print(list_features)
# X_train_new=X_train_new[list_features]
# #print(X_train_new)
# X_train_new=np.asarray(X_train_new)
# X_train_new=X_train_new.reshape(len(X_train_new),1,k)



valid_x=pd.DataFrame.from_records(valid_x)
valid_x=valid_x[list_features]
valid_x=np.asarray(valid_x)
valid_x=valid_x.reshape(len(valid_x),1,k)

X_test_new=pd.DataFrame.from_records(X_test)
X_test_new=X_test_new[list_features]
X_test_new=np.asarray(X_test_new)
X_test_new=X_test_new.reshape(len(X_test_new),1,k)

#label_train = label_train.reshape((len(label_train), 1))
#label_test = label_test.reshape((len(label_test), 1))



label_2_train=to_categorical(train_y)
label_2_test_new=to_categorical(label_2_test)
valid_y=to_categorical(valid_y)


#print(X_train[0][0])
num_classes=2
batch_size=256
epochs=10000

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=4000)
mc = ModelCheckpoint('best_model_1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# callbacks = [EarlyStopping(monitor='val_loss', patience=200),
#              ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

#print(new_array)

model_first=Sequential()
model_first.add(LSTM(100,activation='sigmoid',input_shape=(1,k),return_sequences=True))
model_first.add(Dropout(0.2))
model_first.add(LSTM(200,activation='sigmoid',return_sequences=True))
model_first.add(Dropout(0.2))
#model.add(LSTM(200,activation='sigmoid',return_sequences=True))
#model.add(Dropout(0.2))
model_first.add(LSTM(100,activation='sigmoid'))
model_first.add(Dropout(0.2))
#model.add(LSTM(50,activation='sigmoid'))
model_first.add(Dense(1000, activation='sigmoid'))
model_first.add(BatchNormalization())
#model.add(Dense(500, activation='sigmoid'))
model_first.add(Dense(100, activation='sigmoid'))
# model_first.add(Dense(50, activation='sigmoid'))
model_first.add(Dense(num_classes, activation='softmax'))
# model_first.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001))
model_first.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model_first.summary()
history=model_first.fit(X_train_new, label_2_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_y),callbacks=[es,mc])

# _, accuracy = model_train.evaluate(X_test_new, label_2_test_new, batch_size=batch_size, verbose=1)
# saved_model = load_model('best_model.h5')
predicted=model_first.predict_classes(X_test_new)

#model_json = model_first.to_json()
#with open("model_first.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model_first.save_weights("model_first_weight.h5")
#print("Saved model to disk")

metric=sklearn.metrics.classification_report(label_2_test,predicted)
print(metric) 


print(roc_auc_score(label_2_test, predicted))


print('micro average',f1_score(label_2_test, predicted, average='micro'))
print('macro average',f1_score(label_2_test, predicted, average='macro'))
# print(accuracy)

plt.figure(figsize=(4, 4))
plt.plot(history.history['accuracy'],linewidth=2)
plt.plot(history.history['val_accuracy'],linewidth=2)
plt.title('Model Accuracy',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.legend(['train', 'Validation'], loc='lower right')
plt.show()




