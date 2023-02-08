# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:37:36 2023

@author: monim
"""

from final_functions_context import *
from get_cortisol_gt_context import subject_id
import warnings
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('always')

# subject_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
#             31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50] 

# subject_id = good_subject_id

list_df = []
for i in tqdm(subject_id):
    file_name='dataset/E4_Data/ST0'+str(i)+'/processed_context.csv'
    df = pd.read_csv(file_name)
    list_df.append(df)

all_features = pd.concat(list_df)
all_features.to_csv('dataset/all_features_context.csv')