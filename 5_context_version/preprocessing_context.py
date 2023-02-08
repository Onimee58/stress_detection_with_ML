# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:38:41 2023

@author: monim
"""

from final_functions_context import *
from get_cortisol_gt_context import subject_id, ground_truth
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

sample_rate_gsr=4
sample_rate_ppg=64
sample_rate_hr=1
sample_rate_acc=32
sample_rate_st=4
frequency_gsr=1
frequency_ppg=10
window_size=120
sample_rate_bvp=64
window_length=30
time_to_output=5
sample_rate_label = 1


# subject_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
            # 31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50] 
            
#subject_id = good_subject_id

dataframe_labels = ground_truth()

for i in tqdm(subject_id):
    ID=i
    if(i>=10):
        try:
            gsr_filename='dataset/E4_Data/ST0'+str(i)+'/EDA.csv'
            ppg_filename='dataset/E4_Data/ST0'+str(i)+'/BVP.csv'
            ibi_filename='dataset/E4_Data/ST0'+str(i)+'/IBI.csv'
            st_filename='dataset/E4_Data/ST0'+str(i)+'/TEMP.csv'
            output_file='dataset/E4_Data/ST0'+str(i)+'/processed_context.csv'
    #        label_list='st'+str(i)
        except:
            print('missing ST0'+str(i))
        
    # stress_1, stress_2 = (5,8)
    # no_stress_1, no_stress_2 = (25,38)
    # relax_1, relax_2 = (45, 50)
    time_1, time_2 = (5,70)


    
    df_all=get_all_feature(time_1,time_2,gsr_filename,ppg_filename,ibi_filename,st_filename,ID)
    dataframe_labels=get_segment(dataframe_labels,sample_rate_label,time_1,time_2)
    lbl = np.array(dataframe_labels[i])
    df_all['Labels'] = lbl
    df_all=df_all.fillna(np.mean(df_all))
    df_all=df_all.fillna(0)
    
    # df_all = pd.concat([df_no_stress,df_stress,df_low_stress], axis=0, join='outer', ignore_index=False)
    # df_all = pd.concat([df_no_stress,df_stress], axis=0, join='outer', ignore_index=False)
    df_all.to_csv(output_file)
    # print('ST0'+str(i)+ 'done')