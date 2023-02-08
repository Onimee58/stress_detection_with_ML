# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:32:46 2023

@author: monim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import glob
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,mutual_info_regression
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from scipy.fftpack import fft
import warnings
warnings.filterwarnings('ignore')
from get_cortisol_gt_context import ground_truth





# constants
sample_rate_gsr=4
sample_rate_ppg=64
sample_rate_bvp=64
sample_rate_hr=1
sample_rate_acc=32
sample_rate_st=4
frequency_gsr=1
frequency_ppg=10
window_size=120
window_length=30
time_to_output=5
sample_rate_label = 1


# functions
def find_range(array_data):
    if(len(array_data)>0):
        range_data=abs(np.max(array_data)-np.min(array_data))
    if(len(array_data)==0):
        range_data=0
    return range_data
def find_max(array_data):
    if(len(array_data)>0):
        max_data=abs(np.max(array_data))
    if(len(array_data)==0):
        max_data=0
    return max_data
def find_min(array_data):
    if(len(array_data)>0):
        min_data=abs(np.min(array_data))
    if(len(array_data)==0):
        min_data=0
    return min_data
def normalize(dataframe):
    minimum=np.min(dataframe)
    maximum=np.max(dataframe)
    dataframe=(dataframe-minimum)/(maximum-minimum)
    return dataframe
def calculate_sample_rate(dataframe_signal):
    time=dataframe_signal['time']
    # time=time/1000
    d_time=time.diff().mean()
    sample_frequency=int(1/d_time)
    return int(sample_frequency)
def get_segment(dataframe,sample_rate,start_time,end_time):
    start_sample=start_time*60*sample_rate
    end_sample=end_time*60*sample_rate
    segment=dataframe[start_sample:end_sample]
    return segment

def sine_normalize(dataframe):
    dataframe_values=dataframe.values
    sined_values=[]
    for i in dataframe_values:
        value_sine=math.sin(i)
        sined_values.append(value_sine)
    return sined_values

def get_estimated_frequencies(bvp_signal_segment, window_length,time_to_output,sample_rate):
    estimated_frequencies=[] # Contains the list of predicted frequencies
    start=0
    bvp_signal_segment = bvp_signal_segment.flatten()
    end=int(window_length*sample_rate_bvp)
    overlap=int(time_to_output*sample_rate_bvp)
    while(end<len(bvp_signal_segment)):
        sample_frequency, power_spectrum=signal.periodogram(bvp_signal_segment[start:end], sample_rate_bvp)
        # You can modify other parameters such as window or nfft to get better precision in the estimation of power spectrum using the periodohgram method.
        # You can also explore other power spectral method. And ofcourse, you can look into existing literature for other methods. 
        index_max,=np.where(power_spectrum==np.max(power_spectrum))
        estimated_frequencies.append(sample_frequency[index_max])
        start=start+overlap
        end=end+overlap
    return estimated_frequencies

def butterworth_new(raw_signal,n,desired_cutoff,sample_rate,btype):
    if(btype=='high' or btype=='low'):
        critical_frequency=(2*desired_cutoff)/sample_rate
        B, A = signal.butter(n, critical_frequency, btype=btype, output='ba')
    elif(btype=='bandpass'):
        critical_frequency_1=(2*desired_cutoff[0])/sample_rate
        critical_frequency_2=(2*desired_cutoff[1])/sample_rate
        B, A = signal.butter(n, [critical_frequency_1,critical_frequency_2], btype=btype, output='ba')
    filtered_signal = signal.filtfilt(B,A, raw_signal, axis=0)
    return filtered_signal

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
        if(p_value<0.05): #if(p_value<0.05):
            feature_names.append(column)
    return feature_names

def get_segment_ibi(dataframe,start_fraction,end_fraction):
    dataframe.columns=['time','ibi']
    length=len(dataframe)
    segment=dataframe[int(start_fraction*length):int(end_fraction*length)]
    return segment

def get_timings_ibi(dataframe):
    dataframe.columns=['time','ibi']
    time=dataframe['time']
    start_time=time.iloc[0]
    end_time=time.iloc[-1]
    duration=end_time-start_time
    return duration/60
def get_timings_eda(dataframe):
    return len(dataframe)/(4*60)
        
  
def get_scl(dataframe_gsr):
    dataframe_gsr=dataframe_gsr.fillna(np.mean(dataframe_gsr))
    min_scl=np.min(dataframe_gsr)
    max_scl=np.max(dataframe_gsr)
    scl_mean=np.mean(dataframe_gsr)
    scl=(scl_mean-min_scl)/(max_scl-min_scl)  
    return scl
def butterworth(dataframe,n,wn):
    B, A = signal.butter(n, wn, output='ba')
    dataframe = signal.filtfilt(B,A, dataframe,axis=0)
    return dataframe
def get_slope_intercept_ibi(segment_ibi):
    # print(segment_ibi)
    if(len(segment_ibi)==0 or len(segment_ibi)==1):
        return 0,0,0,0
    else:
        return stats.theilslopes(segment_ibi['ibi'], segment_ibi['time'], 0.90) 
def get_slope_intercept(s_temp):
    if(len(s_temp)==0):
        return 0,0,0,0
    else:
        return stats.theilslopes(s_temp, np.arange(len(s_temp)), 0.90) 

  
def get_all_feature(time_1, time_2, gsr_filename,ppg_filename,ibi_filename,st_filename,ID):
    #extracting data
    #time_1,time_2 = (25, 38) #!!!!
    dataframe_gsr=pd.read_csv(gsr_filename)
    dataframe_ppg=pd.read_csv(ppg_filename)
    dataframe_st=pd.read_csv(st_filename)
    dataframe_ibi=pd.read_csv(ibi_filename)
    
    ibi=get_segment_ibi(dataframe_ibi,0.2,0.6)
    ibi_n=normalize(ibi['ibi'])
    ibi['ibi']=ibi_n
    dataframe_st=get_segment(dataframe_st,sample_rate_st,time_1,time_2)
    s_temp=np.asarray(dataframe_st)
    s_temp=normalize(s_temp)
    s_temp=s_temp.reshape(len(s_temp),)
    dataframe_gsr=get_segment(dataframe_gsr,sample_rate_gsr,time_1,time_2)
    dataframe_gsr=dataframe_gsr-(get_scl(dataframe_gsr))
    dataframe_ppg=get_segment(dataframe_ppg,sample_rate_ppg,time_1,time_2) 
    cuttoff_gsr=(2*frequency_gsr)/sample_rate_gsr
    cuttoff_ppg=(2*frequency_ppg)/sample_rate_ppg
    variable_sample_gsr=normalize(dataframe_gsr)
    variable_sample_ppg=normalize(dataframe_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    #sample_rate_bvp=calculate_sample_rate(dataframe_ppg) # for new added feature
    bvp_filtered=butterworth_new(variable_sample_ppg,3,[0.2,0.8],sample_rate_bvp,btype='bandpass') # for new added feature
    estimated_frequencies=get_estimated_frequencies(bvp_filtered,window_length,time_to_output,sample_rate_bvp) # for new added feature
    estimated_respiratory_rate=list(np.asarray(estimated_frequencies).flatten()*60) # for new added feature
    max_estimates_rr = np.max(estimated_respiratory_rate)
    avg_estimated_rr = np.mean(estimated_respiratory_rate)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
    variable_sample_gsr=variable_sample_gsr.reshape(len(variable_sample_gsr),)
    variable_sample_ppg=variable_sample_ppg.reshape(len(variable_sample_ppg),)
    gsr_1=np.gradient(variable_sample_gsr)
    # GSR
    rms_gsr=[]
    # PEAK
    mean_gsr_peak=[]
    median_gsr_peak=[]
    std_gsr_peak=[]
    rms_gsr_peak=[]
    max_gsr_peak=[]
    min_gsr_peak=[]
    # PEAK WIDTH
    mean_peak_width=[]
    median_peak_width=[]
    std_peak_width=[]
    rms_peak_width=[]
    max_peak_width=[]
    min_peak_width=[]
    # PEAK PROMINENCE
    mean_peak_prom=[]
    median_peak_prom=[]
    std_peak_prom=[]
    rms_peak_prom=[]
    max_peak_prom=[]
    min_peak_prom=[]
    start=0
    end=int(window_size*sample_rate_gsr)
    overlap=int((window_size*sample_rate_gsr)/2)
    height=0.0001
    # PEAK GSR and GSR_1
    while(end<=len(variable_sample_gsr)):
        peak_indices, _ = find_peaks(gsr_1[start:end],height)
        gsr_values=variable_sample_gsr[peak_indices]
        mean_gsr_peak.append(np.mean(gsr_values))
        median_gsr_peak.append(np.median(gsr_values))
        std_gsr_peak.append(np.std(gsr_values))
        rms_gsr_peak.append(np.sqrt(np.mean(gsr_values)**2))
        max_gsr_peak.append(find_max(gsr_values))
        min_gsr_peak.append(find_min(gsr_values))
        peak_widths=signal.peak_widths(gsr_1[start:end],peak_indices)
        peak_prominance=signal.peak_prominences(gsr_1[start:end],peak_indices)
        p_widths=peak_widths[0]
        p_prom=peak_prominance[0]
        mean_peak_width.append(np.mean(p_widths))
        median_peak_width.append(np.median(p_widths))
        std_peak_width.append(np.std(p_widths))
        rms_peak_width.append(np.sqrt(np.mean(p_widths)**2))
        max_peak_width.append(find_max(p_widths))
        min_peak_width.append(find_min(p_widths))
        mean_peak_prom.append(np.mean(p_prom))
        median_peak_prom.append(np.median(p_prom))
        std_peak_prom.append(np.std(p_prom))
        rms_peak_prom.append(np.sqrt(np.mean(p_prom)**2))
        max_peak_prom.append(find_max(p_prom))
        min_peak_prom.append(find_min(p_prom))
        rms_gsr.append(np.sqrt(np.mean(variable_sample_gsr[start:end])**2)/2)
        start=start+overlap
        end=end+overlap
    #PPG
    ppg_rms=[]
    #PEAK
    hrs=[]
    hrv=[]
    mean_ppg=[]
    std_ppg=[]
    rms_ppg=[]
    range_ppg=[]
    hrv_std=[]
    hrv_rms=[]
    range_hrv=[]
    # WIDTH
    mean_ppg_width=[]
    median_ppg_width=[]
    std_ppg_width=[]
    rms_ppg_width=[]
    max_ppg_width=[]
    min_ppg_width=[]
    # PROMINENCE
    mean_ppg_prom=[]
    median_ppg_prom=[]
    std_ppg_prom=[]
    rms_ppg_prom=[]
    max_ppg_prom=[]
    min_ppg_prom=[]
    start=0
    end=int(window_size*sample_rate_ppg)
    overlap=int((window_size*sample_rate_ppg)/2)
    dist=int(sample_rate_ppg*0.65)
    width=int(sample_rate_ppg*0.1)
    while(end<=len(variable_sample_ppg)):
        peaks, _ = find_peaks(variable_sample_ppg[start:end],width=width, height=0.4,distance=dist)
        ppg_values=variable_sample_ppg[peaks]
        mean_ppg.append(np.mean(ppg_values))
        std_ppg.append(np.std(ppg_values))
        range_ppg.append(find_range(ppg_values))
        rms_ppg.append(np.sqrt(np.mean(ppg_values)**2))
        ppg_rms.append(np.sqrt(np.mean(variable_sample_ppg[start:end])**2))
        bps=len(peaks)/window_size
        bpm=bps*60
        hrs.append(bpm)
        hrv_time=[]
        for i in range(1,len(peaks)):
            sample=peaks[i]-peaks[i-1]
            time=(sample/sample_rate_ppg)*window_size
            hrv_time.append(time) 
        hrv_mean=np.mean(hrv_time)
        std=np.std(hrv_time)
        range_hrv.append(find_range(hrv_time))
        rms=np.sqrt(np.mean(hrv_time)**2)
        hrv.append(hrv_mean)
        hrv_std.append(std)
        hrv_rms.append(rms)
        peak_widths_ppg=signal.peak_widths(variable_sample_ppg[start:end],peaks)
        peak_prominance_ppg=signal.peak_prominences(variable_sample_ppg[start:end],peaks)
        p_widths_ppg=peak_widths_ppg[0]
        p_prom_ppg=peak_prominance_ppg[0]
        mean_ppg_width.append(np.mean(p_widths_ppg))
        median_ppg_width.append(np.median(p_widths_ppg))
        std_ppg_width.append(np.std(p_widths_ppg))
        rms_ppg_width.append(np.sqrt(np.mean(p_widths_ppg)**2))
        max_ppg_width.append(find_max(p_widths_ppg))
        min_ppg_width.append(find_min(p_widths_ppg))
        mean_ppg_prom.append(np.mean(p_prom_ppg))
        median_ppg_prom.append(np.median(p_prom_ppg))
        std_ppg_prom.append(np.std(p_prom_ppg))
        rms_ppg_prom.append((np.sqrt(np.mean(p_prom_ppg)**2)))
        max_ppg_prom.append(find_max(p_prom_ppg))
        min_ppg_prom.append(find_min(p_prom_ppg))
        start=start+overlap
        end=end+overlap
    # ST
    st_mean=[]
    st_sd=[]
    st_median=[]
    st_rms=[]
    st_range=[]
    st_slope=[]
    st_intercept=[]
    st_lb_slope=[]
    st_ub_slope=[]
    start=0
    end=int(window_size*sample_rate_st)
    overlap=int((window_size*sample_rate_st)/2)
    while(end<=len(s_temp)):
        st_mean.append(np.mean(s_temp[start:end]))
        st_sd.append(np.std(s_temp[start:end]))
        st_median.append(np.median(s_temp[start:end]))
        st_rms.append(np.sqrt(np.mean(s_temp[start:end])**2))
        st_range.append(find_range(s_temp[start:end]))
        res=get_slope_intercept(s_temp[start:end])
        st_slope.append(res[0])
        st_intercept.append(res[1])
        st_lb_slope.append(res[2])
        st_ub_slope.append(res[3])
        start=start+overlap
        end=end+overlap
        
    # IBI
    mean_ibi=[]
    sd_ibi=[]
    median_ibi=[]
    max_ibi=[]
    min_ibi=[]
    rms_ibi=[]
    slope_ibi=[]
    intercept_ibi=[]
    lb_slope_ibi=[]
    ub_slope_ibi=[]
    # print(ibi)
    # print(ID)
    start=int(ibi['time'].iloc[0])
    end=int(window_size+start)
    overlap=int(45+start)
    end_time=ibi['time'].iloc[-1]
    while(end<=end_time):
        segment_ibi=ibi.loc[(ibi['time']>=start) & (ibi['time']<=end)]
        mean_ibi.append(np.mean(segment_ibi['ibi']))
        sd_ibi.append(np.std(segment_ibi['ibi']))
        max_ibi.append(find_max(segment_ibi['ibi']))
        min_ibi.append(find_min(segment_ibi['ibi']))
        median_ibi.append(np.median(segment_ibi['ibi']))
        rms_ibi.append(np.sqrt(np.mean(segment_ibi['ibi'])**2))
        res = get_slope_intercept_ibi(segment_ibi)
        slope_ibi.append(res[0])
        intercept_ibi.append(res[1])
        lb_slope_ibi.append(res[2])
        ub_slope_ibi.append(res[3])
        start=start+int(window_size/2)
        end=end+int(window_size/2)
    
    # print(len(mean_gsr_peak))
    # print(len(ub_slope_ibi))
    
    length=len(mean_gsr_peak)
    length_ibi=len(ub_slope_ibi)
    
    if(length_ibi>length):
        mean_ibi=mean_ibi[:length]
        sd_ibi=sd_ibi[:length]
        median_ibi=median_ibi[:length]
        max_ibi=max_ibi[:length]
        min_ibi=min_ibi[:length]
        rms_ibi=rms_ibi[:length]
        slope_ibi=slope_ibi[:length]
        intercept_ibi=intercept_ibi[:length]
        lb_slope_ibi=lb_slope_ibi[:length]
        ub_slope_ibi=ub_slope_ibi[:length]
    
    if(length_ibi<length):
        for i in range(length-length_ibi):
            mean_ibi.append(np.mean(mean_ibi))
            sd_ibi.append(np.mean(sd_ibi))
            median_ibi.append(np.mean(median_ibi))
            max_ibi.append(np.mean(max_ibi))
            min_ibi.append(np.mean(min_ibi))
            rms_ibi.append(np.mean(rms_ibi))
            slope_ibi.append(np.mean(slope_ibi))
            intercept_ibi.append(np.mean(intercept_ibi))
            lb_slope_ibi.append(np.mean(lb_slope_ibi))
            ub_slope_ibi.append(np.mean(ub_slope_ibi))
    
    # Merge and create dataframe
    df=pd.DataFrame({
        # GSR
            'MEAN_PEAK_GSR':mean_gsr_peak,
            'MEDIAN_PEAK_GSR':median_gsr_peak,
            'STD_PEAK_GSR':std_gsr_peak,
            'RMS_PEAK_GSR':rms_gsr_peak,
            'MAX_PEAK_GSR':max_gsr_peak,
            'MIN_PEAK_GSR':min_gsr_peak,
            'MEAN_PEAK_WIDTH':mean_peak_width,
            'MEDIAN_PEAK_WIDTH':median_peak_width,
            'STD_PEAK_WIDTH':std_peak_width,
            'RMS_PEAK_WIDTH':rms_peak_width,
            'MAX_PEAK_WIDTH':max_peak_width,
            'MIN_PEAK_WIDTH':min_peak_width,
            'MEAN_PEAK_PROM':mean_peak_prom,
            'MEDIAN_PEAK_PROM':median_peak_prom,
            'STD_PEAK_PROM':std_peak_prom,
            'RMS_PEAK_PROM':rms_peak_prom,
            'MAX_PEAK_PROM':max_peak_prom,
            'MIN_PEAK_PROM':min_peak_prom,
            # PPG
            'HR':hrs,
            'HRV':hrv,
            'RANGE_HRV':range_hrv,
            'MEAN_PPG':mean_ppg,
            'STD_PPG':std_ppg,
            'RMS_PPG':rms_ppg,
            'RANGE_PPG':range_ppg,
            'HRV_STD':hrv_std,
            'HRV_RMS':hrv_rms,
            'MEAN_PPG_WIDTH':mean_ppg_width,
            'MEDIAN_PPG_WIDTH':median_ppg_width,
            'STD_PPG_WIDTH':std_ppg_width,
            'RMS_PPG_WIDTH':rms_ppg_width,
            'MAX_PPG_WIDTH':max_ppg_width,
            'MIN_PPG_WIDTH':min_ppg_width,
            'MEAN_PPG_PROM':mean_ppg_prom,
            'MEDIAN_PPG_PROM':median_ppg_prom,
            'STD_PPG_PROM':std_ppg_prom,
            'RMS_PPG_PROM':rms_ppg_prom,
            'MAX_PPG_PROM':max_ppg_prom,
            'MIN_PPG_PROM':min_ppg_prom,
            'AVG_RESP_FREQUENCY':avg_estimated_rr, # for new added feature
            'MAX_RESP_FREQUENCY': max_estimates_rr,
            # # ST
            'ST_MEAN':st_mean,
            'ST_SD':st_sd,
            'ST_MEDIAN':st_median,
            'ST_RMS':st_rms,
            'ST_RANGE':st_range,
            'ST_SLOPE':st_slope,
            'ST_INTERCEPT':st_intercept,
            'ST_LB_SLOPE':st_lb_slope,
            'ST_UB_SLOPE':st_ub_slope,
            # IBI
            'MEAN_IBI':mean_ibi,
            'SD_IBI':sd_ibi,
            'MEDIAN_IBI':median_ibi,
            'MAX_IBI':max_ibi,
            'MIN_IBI':min_ibi,
            'RMS_IBI':rms_ibi,
            'SLOPE_IBI':slope_ibi,
            'INTERCEPT_IBI':intercept_ibi,
            'LB_SLOPE_IBI':lb_slope_ibi,
            'UB_SLOPE_IBI':ub_slope_ibi})
    df['ID']=ID
    # df['Phase']='stress'
    # df['Labels'] = dataframe_labels
    return df
    
        
if __name__ == '__main__':
    print('do not run this file, its is a list of functions')
    
    


    
    
    
    
    
    