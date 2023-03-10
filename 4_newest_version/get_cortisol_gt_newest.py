# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 01:11:29 2023

@author: Saif
"""

from final_functions_newest import *

all_gt = 'dataset/Cortisol_gt.csv'
gt_all = np.array(pd.read_csv(all_gt))

subject_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
            31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]

subject_idx = [sub-1 for sub in subject_id]
total_data_len = 50
time_slots = ['T1', 'T2', 'T3', 'T4', 'T5']

sub_v = np.zeros([total_data_len,5])
i = 0
for s in range(0,total_data_len*5,5):
    sub_v[i] = gt_all[s:s+5,1]
    i+=1

cort_reading = sub_v[subject_idx]

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
bad_subject_id = sub_id[bad_data_idx]
bad_cort_reading = cort_reading[bad_data_idx]

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