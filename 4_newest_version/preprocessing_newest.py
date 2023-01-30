# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:38:41 2023

@author: monim
"""

from final_functions_newest import *
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
    
subject_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
            31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50] 
#label2   
st0=[]
st1=[1,1,1]
st2=[]
st3=[]
st4=[1,1,1]
st5=[1,1,1]
st6=[1,1,1]
st7=[1,1,1]
st8=[]
st9=[]
st10=[1,1,1]
st11=[0,1,1]
st12=[1,1,1]
st13=[1,1,1]
st14=[1,1,1]
st15=[1,1,1]
st16=[1,0,0]
st17=[1,1,1]
st18=[1,1,1]
st19=[0,0,0]
st20=[1,0,0]
st21=[1,1,0]
st22=[1,1,1]
st23=[1,1,1]
st24=[]
st25=[1,1,1]
st26=[1,0,0]
st27=[0,1,1]
st28=[0,0,0]
st29=[1,1,1]
st30=[0,0,0]
st31=[0,1,0]
st32=[1,1,1]
st33=[0,0,0]
st34=[0,0,0]
st35=[1,1,1]
st36=[1,0,0]
st37=[0,0,0]
st38=[1,1,1]
st39=[1,1,0]
st40=[1,1,1]
st41=[1,0,0]
st42=[1,1,0]
st43=[0,0,0]
st44=[1,1,1]
st45=[1,1,0]
st46=[1,1,0]
st47=[0,0,1]
st48=[1,1,1]
st49=[1,1,1]
st50=[0,0,0]

# label3
#st0=[]
#st1=[2,2,1]
#st2=[]
#st3=[]
#st4=[2,2,1]
##st5=[1,1,1]
##st6=[1,1,1]
##st7=[1,1,1]
##st8=[]
##st9=[]
#st10=[1,1,1]
#st11=[0,1,1]
#st12=[1,2,1]
#st13=[1,1,1]
#st14=[1,1,2]
#st15=[2,1,1]
#st16=[1,0,0]
#st17=[1,1,1]
#st18=[1,1,1]
#st19=[0,0,0]
#st20=[2,0,0]
#st21=[2,1,0]
#st22=[1,2,2]
#st23=[2,2,1]
#st24=[]
#st25=[1,2,1]
#st26=[2,0,0]
#st27=[0,1,1]
#st28=[0,0,0]
#st29=[2,1,1]
#st30=[0,0,0]
#st31=[0,2,0]
#st32=[1,2,2]
#st33=[0,0,0]
#st34=[0,0,0]
#st35=[1,1,1]
#st36=[1,0,0]
#st37=[0,0,0]
#st38=[2,2,1]
##st39=[1,1,1]
#st40=[2,1,2]
#st41=[1,0,0]
#st42=[1,1,0]
#st43=[0,1,0]
#st44=[2,2,1]
#st45=[2,2,0]
#st46=[1,1,1]
#st47=[0,0,1]
#st48=[2,1,1]
#st49=[1,1,1]
#st50=[0,0,0]
# values
st0_v=[]
st1_v=[50,69,47]
st2_v=[]
st3_v=[]
st4_v=[55,52,35]
st5_v=[31,54,24]
st6_v=[51,30,14]
st7_v=[30,9,30]
st8_v=[]
st9_v=[]
st10_v=[41,22,16]
st11_v=[-31,18,22]
st12_v=[27,71,32]
st13_v=[30,35,31]
st14_v=[3,27,51]
st15_v=[50,7,36]
st16_v=[46,-6,-4]
st17_v=[11,17,17]
st18_v=[38,25,36]
st19_v=[-343,381,-496]
st20_v=[52,-38,-42]
st21_v=[178,33,-38]
st22_v=[44,96,59]
st23_v=[64,105,24]
st24_v=[]
st25_v=[41,55,19]
st26_v=[244,-9,-53]
st27_v=[-40,33,5]
st28_v=[-109,-223,-189]
st29_v=[57,25,9]
st30_v=[-89,-154,-224]
st31_v=[-76,61,-88]
st32_v=[40,123,101]
st33_v=[-31,-120,-76]
st34_v=[-39,-83,-104]
st35_v=[27,38,24]
st36_v=[19,-32,-120]
st37_v=[-41,-69,-82]
st38_v=[119,61,18]
st39_v=[22,50,-64]
st40_v=[61,30,117]
st41_v=[7,-90,-110]
st42_v=[36,38,-22]
st43_v=[-22,-1,-29]
st44_v=[189,87,17]
st45_v=[96,305,-16]
st46_v=[25,16,-1]
st47_v=[-8,-27,14]
st48_v=[50,47,41]
st49_v=[21,24,19]
st50_v=[-176,-222,-182]


subject_label=[st0,st1,st2,st3,st4,st5,st6,st7,st8,st9,st10,st11,st12,st13,st14,st15,st16,st17,
               st18,st19,st20,st21,st22,st23,st24,st25,st26,st27,st28,st29,st30,st31,st32,st33,
               st34,st35,st36,st37,st38,st39,st40,st41,st42,st43,st44,st45,st46,st47,st48,st49,st50]

subject_label_v=[st0_v,st1_v,st2_v,st3_v,st4_v,st5_v,st6_v,st7_v,st8_v,st9_v,st10_v,st11_v,st12_v,st13_v,
                 st14_v,st15_v,st16_v,st17_v,st18_v,st19_v,st20_v,st21_v,st22_v,st23_v,st24_v,st25_v,st26_v,
                 st27_v,st28_v,st29_v,st30_v,st31_v,st32_v,st33_v,st34_v,st35_v,st36_v,st37_v,st38_v,st39_v,
                 st40_v,st41_v,st42_v,st43_v,st44_v,st45_v,st46_v,st47_v,st48_v,st49_v,st50_v]


for i in tqdm(subject_id):
    ID=i

    # if(i<10):
    #     try:
    #         gsr_filename='dataset/E4_Data/ST00'+str(i)+'/EDA.csv'
    #         ppg_filename='dataset/E4_Data/ST00'+str(i)+'/BVP.csv'
    #         output_file='dataset/E4_Data/ST00'+str(i)+'/processed.csv'
    # #        label_list='st'+str(i)
    #     except:
    #         print('missing ST00'+str(i))
    if(i>=10):
        try:
            gsr_filename='dataset/E4_Data/ST0'+str(i)+'/EDA.csv'
            ppg_filename='dataset/E4_Data/ST0'+str(i)+'/BVP.csv'
            ibi_filename='dataset/E4_Data/ST0'+str(i)+'/IBI.csv'
            st_filename='dataset/E4_Data/ST0'+str(i)+'/TEMP.csv'
            output_file='dataset/E4_Data/ST0'+str(i)+'/processed_newV1.csv'
    #        label_list='st'+str(i)
        except:
            print('missing ST0'+str(i))

    labelb=subject_label[i][0]
    labels=subject_label[i][1]
    labelr=subject_label[i][2]
    
    labelb_v=subject_label_v[i][0]
    labels_v=subject_label_v[i][1]
    labelr_v=subject_label_v[i][2]  
    
    df_baseline=get_baseline_feature(gsr_filename,ppg_filename,ibi_filename,st_filename,ID,labelb,labelb_v)
    df_baseline=df_baseline.fillna(np.mean(df_baseline))
    df_baseline=df_baseline.fillna(0)
    df_stress=get_stress_feature(gsr_filename,ppg_filename,ibi_filename,st_filename,ID,labels,labels_v)
    df_stress=df_stress.fillna(np.mean(df_stress))
    df_stress=df_stress.fillna(0)
    df_relax=get_relax_feature(gsr_filename,ppg_filename,ibi_filename,st_filename,ID,labelr,labelr_v)
    df_relax=df_relax.fillna(np.mean(df_relax))
    df_relax=df_relax.fillna(0)
    
    df_all = pd.concat([df_baseline,df_stress,df_relax], axis=0, join='outer', ignore_index=False)
    # df_all = pd.concat([df_baseline,df_stress], axis=0, join='outer', ignore_index=False)
    df_all.to_csv(output_file)
    # print('ST0'+str(i)+ 'done')