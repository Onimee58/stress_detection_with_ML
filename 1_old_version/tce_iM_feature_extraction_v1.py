import pandas as pd
import numpy as np
import scipy
import peakutils
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.fftpack import fft, ifft
from sklearn import datasets, linear_model, metrics
import statistics as st
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks



#
#cuttoff=(2*frequency)/sample_rate
# FUNCTIONS
def calculate_sample_rate(dataframe_signal):
    time=dataframe_signal[TIME]
    time=time/1000
    d_time=time.diff().mean()
    sample_frequency=int(1/d_time)
    return int(sample_frequency)
def butterworth(dataframe,n,wn):
    B, A = signal.butter(n, wn, output='ba')
    dataframe = signal.filtfilt(B,A, dataframe)
    return dataframe
def normalize(dataframe):
    minimum=np.min(dataframe)
    maximum=np.max(dataframe)
    dataframe=(dataframe-minimum)/(maximum-minimum)
    return dataframe
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


#X_train_new = np.empty((0))
# X_train_new=np.append(X_train_new,X_train[i])
def cal_baseline_feature(df_baseline,labelb,labelb_v,ID):
    sample_rate=calculate_sample_rate(df_baseline)
    length=sample_rate*60*2
    df_baseline=df_baseline.iloc[:length,:]
    variable_sample_gsr=df_baseline[GSR]-scl
    variable_sample_ppg=df_baseline[PPG]
    cuttoff_gsr=(2*frequency_gsr)/sample_rate
    cuttoff_ppg=(2*frequency_ppg)/sample_rate
    variable_sample_gsr=normalize(variable_sample_gsr)
    variable_sample_ppg=normalize(variable_sample_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
    dist=int(sample_rate*0.65)
    width=int(sample_rate*0.1)
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
            time=(sample/sample_rate)*window_size
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
    
 
    # Merge and create dataframe
    df=pd.DataFrame({
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
            'MIN_PPG_PROM':min_ppg_prom})
    df['ID']=ID
    df['Label_2']=labelb
    df['Phase']='b'
    return df
    
def cal_stress_feature(df_stress,labels,labels_v,ID):
    sample_rate=calculate_sample_rate(df_stress)
    length=sample_rate*60*10
    df_stress=df_stress.iloc[:length,:]
    variable_sample_gsr=df_stress[GSR]-scl
    variable_sample_ppg=df_stress[PPG]
    cuttoff_gsr=(2*frequency_gsr)/sample_rate
    cuttoff_ppg=(2*frequency_ppg)/sample_rate
    variable_sample_gsr=normalize(variable_sample_gsr)
    variable_sample_ppg=normalize(variable_sample_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
    dist=int(sample_rate*0.65)
    width=int(sample_rate*0.1)
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
            time=(sample/sample_rate)*window_size
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
    
    # Merge and create dataframe
    df=pd.DataFrame({
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
            'MIN_PPG_PROM':min_ppg_prom})
    df['ID']=ID
    df['Label_2']=labels
    df['Phase']='s'
    return df
    
    

def cal_relax_feature(df_intervention,labelr,labelr_v,ID):
    variable_sample_gsr=df_intervention[GSR]-scl
    variable_sample_ppg=df_intervention[PPG]
    sample_rate=calculate_sample_rate(df_intervention)
    time_sec=len(df_stress)/sample_rate
    time_min=time_sec*60
    print(time_min)
    cuttoff_gsr=(2*frequency_gsr)/sample_rate
    cuttoff_ppg=(2*frequency_ppg)/sample_rate
#    variable_sample_gsr=signal.detrend(variable_sample_gsr)
#    variable_sample_ppg=signal.detrend(variable_sample_ppg)
    variable_sample_gsr=normalize(variable_sample_gsr)
    variable_sample_ppg=normalize(variable_sample_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
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
    end=int(window_size*sample_rate)
    overlap=int((window_size*sample_rate)/2)
    dist=int(sample_rate*0.65)
    width=int(sample_rate*0.1)
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
            time=(sample/sample_rate)*window_size
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
    
    # Merge and create dataframe
    df=pd.DataFrame({
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
            'MIN_PPG_PROM':min_ppg_prom})
#            'MEAN_FFT_PPG_1':mean_fft_ppg_1,
#            'MAX_FFT_PPG_1':max_fft_ppg_1,
#            'MIN_FFT_PPG_1':min_fft_ppg_1,
#            'MEDIAN_FFT_PPG_1':median_fft_ppg_1,
#            'RMS_FFT_PPG_1':rms_fft_ppg_1,
#            'STD_FFT_PPG_1':std_fft_ppg_1,
#            'VAR_FFT_PPG_1':var_fft_ppg_1,
#            'SKEW_FFT_PPG_1':skew_fft_ppg_1,
#            'KURT_FFT_PPG_1':kurt_fft_ppg_1})
    df['ID']=ID
    df['Label_2']=labelr
    df['Phase']='r'
    return df
 
#baseline_file='st019_baseline.csv'
#stress_file='st019_stress.csv'
#intervention_file='st019_intervention.csv'
GSR='GSR CAL (µSiemens) (Shimmer3 GSR+)' 
TIME='Timestamp CAL (mSecs) (Shimmer3 GSR+)'
PPG='Internal ADC A13 PPG CAL (mVolts) (Shimmer3 GSR+)'
#ID='19'
#output_file='st019_processed.csv'
## LABEL 2
#label_2_baseline=0
#label_2_stress=0
#label_2_intervention=0
## VARIABLES
frequency_gsr=1
frequency_ppg=10
input_signal=[GSR,PPG]
window_size=120

subject_id=[1,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50]
print(len(subject_id))
# two classes
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


subject_label=[st0,st1,st2,st3,st4,st5,st6,st7,st8,st9,st10,st11,st12,st13,st14,st15,st16,st17,st18,st19,st20,st21,st22,st23,st24,st25,st26,st27,st28,st29,st30,st31,st32,st33,st34,st35,st36,st37,st38,st39,st40,st41,st42,st43,st44,st45,st46,st47,st48,st49,st50]

subject_label_v=[st0_v,st1_v,st2_v,st3_v,st4_v,st5_v,st6_v,st7_v,st8_v,st9_v,st10_v,st11_v,st12_v,st13_v,st14_v,st15_v,st16_v,st17_v,st18_v,st19_v,st20_v,st21_v,st22_v,st23_v,st24_v,st25_v,st26_v,st27_v,st28_v,st29_v,st30_v,st31_v,st32_v,st33_v,st34_v,st35_v,st36_v,st37_v,st38_v,st39_v,st40_v,st41_v,st42_v,st43_v,st44_v,st45_v,st46_v,st47_v,st48_v,st49_v,st50_v]


for i in subject_id:
    ID=i
    if(i<10):
        baseline_file='st00'+str(i)+'_baseline.csv'
        stress_file='st00'+str(i)+'_stress.csv'
        intervention_file='st00'+str(i)+'_intervention.csv'
        output_file='st00'+str(i)+'_processed.csv'
#        label_list='st'+str(i)
    if(i>=10):
        baseline_file='st0'+str(i)+'_baseline.csv'
        stress_file='st0'+str(i)+'_stress.csv'
        intervention_file='st0'+str(i)+'_intervention.csv'
        output_file='st0'+str(i)+'_processed.csv'
#        label_list='st'+str(i)
    labelb=subject_label[i][0]
    labels=subject_label[i][1]
    labelr=subject_label[i][2]
    
    labelb_v=subject_label_v[i][0]
    labels_v=subject_label_v[i][1]
    labelr_v=subject_label_v[i][2]
    
    
    df_baseline=pd.read_csv(baseline_file,encoding = "ISO-8859-1")
    df_stress=pd.read_csv(stress_file,encoding = "ISO-8859-1")
#    df_intervention=pd.read_csv(intervention_file,encoding = "ISO-8859-1")
#    # Baseline Conditioning
    df_baseline_cond=df_baseline[input_signal]
    df_baseline_cond=df_baseline_cond.fillna(np.mean(df_baseline_cond))
    gsr_baseline=df_baseline_cond[GSR]
    ppg_baseline=df_baseline_cond[PPG]
    mean_gsr=np.mean(gsr_baseline)
    mean_ppg=np.mean(ppg_baseline)
    min_scl=np.min(gsr_baseline)
    max_scl=np.max(gsr_baseline)
    scl_mean=np.mean(gsr_baseline)
    scl=(scl_mean-min_scl)/(max_scl-min_scl)
#    # Execution
    baseline=cal_baseline_feature(df_baseline,labelb,labelb_v,ID)
    baseline=baseline.fillna(np.mean(baseline))
    baseline=baseline.fillna(0)
    stress=cal_stress_feature(df_stress,labels,labels_v,ID) 
    stress=stress.fillna(np.mean(stress))
    stress=stress.fillna(0)
#    relax=cal_relax_feature(df_intervention,labelr,labelr_v,ID) 
#    relax=relax.fillna(np.mean(relax))
#    relax=relax.fillna(0)
    # Concat and print
#    df_all = pd.concat([baseline,stress,relax], axis=0, join='outer', ignore_index=False)
    df_all = pd.concat([baseline,stress], axis=0, join='outer', ignore_index=False)
    df_all.to_csv(output_file)
    print(df_all)

#print(df_all)
#sample_rate=157
#dist=int(sample_rate*0.65)
#width=int(sample_rate*0.1)
#
##baseline_1=np.gradient(stress)
#baseline_new=baseline[20000:25000]
#peaks, _ = find_peaks(baseline_new,width=width,height=0.4,distance=dist)
#plt.plot(baseline_new,'black',linewidth=2)
#plt.plot(peaks, baseline_new[peaks], "*",'r',markersize=18)
#plt.show()
#
#plt.plot(baseline_new,'black',linewidth=2)
#plt.show()
























