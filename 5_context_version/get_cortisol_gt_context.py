# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 01:11:29 2023

@author: Saif
"""

#from final_functions_context import *
import numpy as np
import pandas as pd
sample_rate_label = 1


all_gt = 'dataset/Cortisol_gt.csv'
gt_all = np.array(pd.read_csv(all_gt))

subject_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
            31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]

subject_idx = [sub-1 for sub in subject_id]
total_data_len = 50
time_slots = ['T1', 'T2', 'T3', 'T4', 'T5']
classes = ['not_stressed', 'low_stress', 'stress']
sub_v = np.zeros([total_data_len,5])
i = 0
for s in range(0,total_data_len*5,5):
    sub_v[i] = gt_all[s:s+5,1]
    i+=1

cort_reading = sub_v[subject_idx]
labels = np.ones(list(cort_reading.shape))
baseline_idx = []
stressed_idx = []
for i in range(len(cort_reading)):
    baseline_idx.append(np.argmin(cort_reading[i]))
    stressed_idx.append(np.argmax(cort_reading[i]))
    labels[i,baseline_idx[i]] = 0
    labels[i,stressed_idx[i]] = 2

all_labels = np.array(labels, dtype = int)
#%%

all_subject_context_1 = np.ones(list(labels.shape))*np.array([[0],
                                                            [1],
                                                            [1],
                                                            [2],
                                                            [1]]).T

all_subject_context_2 = np.ones(list(labels.shape))*np.array([[0],
                                                            [1],
                                                            [1],
                                                            [2],
                                                            [0]]).T

all_subject_context_3 = np.ones(list(labels.shape))*np.array([[1],
                                                            [1],
                                                            [1],
                                                            [2],
                                                            [0]]).T

#%%
'''
plt.rcParams["figure.figsize"] = (20,20)
plt.rcParams.update({'font.size': 22})
for i in range(len(cort_reading)):
    plt.scatter(time_slots, cort_reading[i])
    plt.plot(time_slots, cort_reading[i])
plt.show()

'''
#%%
gcount = 0
bcount = 0
good_data_idx = []
bad_data_idx = []
for i in range(len(cort_reading)):
    if cort_reading[i,2] > np.mean(cort_reading[i,0:2]):
        good_data_idx.append(i)
        gcount+=1
    else:
        bad_data_idx.append(i)
        bcount+=1
        
        
sub_id = np.array(subject_id)
good_subject_id = sub_id[good_data_idx]
good_cort_reading = cort_reading[good_data_idx]
good_labels = all_labels[good_data_idx]
good_subject_context_1 = all_subject_context_1[good_data_idx]
good_subject_context_2 = all_subject_context_2[good_data_idx]
good_subject_context_3 = all_subject_context_3[good_data_idx]
bad_subject_id = sub_id[bad_data_idx]
bad_cort_reading = cort_reading[bad_data_idx]
bad_subject_context_1 = all_subject_context_1[bad_data_idx]
bad_subject_context_2 = all_subject_context_2[bad_data_idx]
bad_subject_context_3 = all_subject_context_3[bad_data_idx]
bad_labels = all_labels[bad_data_idx]

#%%
'''
plt.clf()
plt.rcParams["figure.figsize"] = (20,20)
plt.rcParams.update({'font.size': 22})
for i in range(len(good_cort_reading)):
    plt.scatter(time_slots, good_cort_reading[i])
    plt.plot(time_slots, good_cort_reading[i])
plt.show()

plt.clf()
plt.rcParams["figure.figsize"] = (20,20)
plt.rcParams.update({'font.size': 22})
for i in range(len(bad_cort_reading)):
    plt.scatter(time_slots, bad_cort_reading[i])
    plt.plot(time_slots, bad_cort_reading[i])
plt.show()
'''
#%%
sample_labels = np.ones([sample_rate_label*60, len(subject_id)])

for j in range(len(all_labels)):
    k = 0
    for i in range(0, len(sample_labels), int(len(sample_labels)/len(time_slots))):
        sample_labels[i:i+int(len(sample_labels)/len(time_slots)), j] = sample_labels[i:i+int(len(sample_labels)/len(time_slots)),j]*all_labels[j,k]
        k+=1
df_labels = pd.DataFrame(sample_labels, index = None, columns = sub_id)


# plt.plot(sample_labels[:,38])

    





