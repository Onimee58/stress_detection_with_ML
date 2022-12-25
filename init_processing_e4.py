# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:36:34 2020

@author: rajde
"""

import pandas as pd
import numpy as np
import scipy
import peakutils
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.tools import FigureFactory as FF
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.signal as signal
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from scipy.fftpack import fft, ifft
from sklearn import datasets, linear_model, metrics
import statistics as st
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm
from scipy import signal
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

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
def get_segment(dataframe,sample_rate,start_time,end_time):
    start_sample=start_time*60*sample_rate
    end_sample=end_time*60*sample_rate
    segment=dataframe[start_sample:end_sample]
    return segment
    
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
def normalize(dataframe):
    minimum=np.min(dataframe)
    maximum=np.max(dataframe)
    dataframe=(dataframe-minimum)/(maximum-minimum)
    return dataframe


def get_baseline_feature(gsr_filename,ppg_filename,ID,labelb,labelb_v):
    #extracting data
    dataframe_gsr=pd.read_csv(gsr_filename)
    dataframe_ppg=pd.read_csv(ppg_filename)
    dataframe_gsr=get_segment(dataframe_gsr,sample_rate_gsr,5,8)
    dataframe_gsr=dataframe_gsr-(get_scl(dataframe_gsr))
    dataframe_ppg=get_segment(dataframe_ppg,sample_rate_ppg,5,8) 
    cuttoff_gsr=(2*frequency_gsr)/sample_rate_gsr
    cuttoff_ppg=(2*frequency_ppg)/sample_rate_ppg
    variable_sample_gsr=normalize(dataframe_gsr)
    variable_sample_ppg=normalize(dataframe_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
    variable_sample_gsr=variable_sample_gsr.reshape(len(variable_sample_gsr),)
    variable_sample_ppg=variable_sample_ppg.reshape(len(variable_sample_ppg),)
    gsr_1=np.gradient(variable_sample_gsr)
    fft_gsr=abs(fft(variable_sample_gsr))
    fft_gsr_1=abs(fft(gsr_1))
    ppg_1=np.gradient(variable_sample_ppg)
    fft_ppg=abs(fft(variable_sample_ppg))
    fft_ppg_1=abs(fft(ppg_1))
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
    # GSR and GSR_1 FFT
    mean_fft_gsr=[]
    max_fft_gsr=[]
    min_fft_gsr=[]
    median_fft_gsr=[]
    rms_fft_gsr=[]
    std_fft_gsr=[]
    var_fft_gsr=[]
    skew_fft_gsr=[]
    kurt_fft_gsr=[]
    mean_fft_gsr_1=[]
    max_fft_gsr_1=[]
    min_fft_gsr_1=[]
    median_fft_gsr_1=[]
    rms_fft_gsr_1=[]
    std_fft_gsr_1=[]
    var_fft_gsr_1=[]
    skew_fft_gsr_1=[]
    kurt_fft_gsr_1=[]
    start=0
    end=int(window_size*sample_rate_gsr)
    overlap=int((window_size*sample_rate_gsr)/2)
    while(end<=len(fft_gsr)):
        mean_fft_gsr.append(np.mean(fft_gsr[start:end]))
        max_fft_gsr.append(np.max(fft_gsr[start:end]))
        min_fft_gsr.append(np.min(fft_gsr[start:end]))
        median_fft_gsr.append(np.median(fft_gsr[start:end]))
        rms_fft_gsr.append(np.sqrt(np.mean(fft_gsr[start:end])**2))
        std_fft_gsr.append(np.std(fft_gsr[start:end]))
        var_fft_gsr.append(np.var(fft_gsr[start:end]))
        skew_fft_gsr.append(skew(fft_gsr[start:end]))
        kurt_fft_gsr.append(kurtosis(fft_gsr[start:end]))
        mean_fft_gsr_1.append(np.mean(fft_gsr_1[start:end]))
        max_fft_gsr_1.append(np.max(fft_gsr_1[start:end]))
        min_fft_gsr_1.append(np.min(fft_gsr_1[start:end]))
        median_fft_gsr_1.append(np.median(fft_gsr_1[start:end]))
        rms_fft_gsr_1.append(np.sqrt(np.mean(fft_gsr_1[start:end])**2))
        std_fft_gsr_1.append(np.std(fft_gsr_1[start:end]))
        var_fft_gsr_1.append(np.var(fft_gsr_1[start:end]))
        skew_fft_gsr_1.append(skew(fft_gsr_1[start:end]))
        kurt_fft_gsr_1.append(kurtosis(fft_gsr_1[start:end]))
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
    
    # PPG and PPG_1 FFT
    mean_fft_ppg=[]
    max_fft_ppg=[]
    min_fft_ppg=[]
    median_fft_ppg=[]
    rms_fft_ppg=[]
    std_fft_ppg=[]
    var_fft_ppg=[]
    skew_fft_ppg=[]
    kurt_fft_ppg=[]
    mean_fft_ppg_1=[]
    max_fft_ppg_1=[]
    min_fft_ppg_1=[]
    median_fft_ppg_1=[]
    rms_fft_ppg_1=[]
    std_fft_ppg_1=[]
    var_fft_ppg_1=[]
    skew_fft_ppg_1=[]
    kurt_fft_ppg_1=[]
    start=0
    end=int(window_size*sample_rate_ppg)
    overlap=int((window_size*sample_rate_ppg)/2)
    while(end<=len(fft_ppg)):
        mean_fft_ppg.append(np.mean(fft_ppg[start:end]))
        max_fft_ppg.append(np.max(fft_ppg[start:end]))
        min_fft_ppg.append(np.min(fft_ppg[start:end]))
        median_fft_ppg.append(np.median(fft_ppg[start:end]))
        rms_fft_ppg.append(np.sqrt(np.mean(fft_ppg[start:end])**2))
        std_fft_ppg.append(np.std(fft_ppg[start:end]))
        var_fft_ppg.append(np.var(fft_ppg[start:end]))
        skew_fft_ppg.append(skew(fft_ppg[start:end]))
        kurt_fft_ppg.append(kurtosis(fft_ppg[start:end]))
        mean_fft_ppg_1.append(np.mean(fft_ppg_1[start:end]))
        max_fft_ppg_1.append(np.max(fft_ppg_1[start:end]))
        min_fft_ppg_1.append(np.min(fft_ppg_1[start:end]))
        median_fft_ppg_1.append(np.median(fft_ppg_1[start:end]))
        rms_fft_ppg_1.append(np.sqrt(np.mean(fft_ppg_1[start:end])**2))
        std_fft_ppg_1.append(np.std(fft_ppg_1[start:end]))
        var_fft_ppg_1.append(np.var(fft_ppg_1[start:end]))
        skew_fft_ppg_1.append(skew(fft_ppg_1[start:end]))
        kurt_fft_ppg_1.append(kurtosis(fft_ppg_1[start:end]))
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
            'MIN_PPG_PROM':min_ppg_prom,
            'RMS_GSR':rms_gsr,
            'PPG_RMS':ppg_rms,
            'MEAN_FFT_GSR':mean_fft_gsr,
            'MAX_FFT_GSR':max_fft_gsr,
            'MIN_FFT_GSR':min_fft_gsr,
            'MEDIAN_FFT_GSR':median_fft_gsr,
            'RMS_FFT_GSR':rms_fft_gsr,
            'STD_FFT_GSR':std_fft_gsr,
            'VAR_FFT_GSR':var_fft_gsr,
            'SKEW_FFT_GSR':skew_fft_gsr,
            'KURT_FFT_GSR':kurt_fft_gsr,
#            'MEAN_FFT_GSR_1':mean_fft_gsr_1,
#            'MAX_FFT_GSR_1':max_fft_gsr_1,
#            'MIN_FFT_GSR_1':min_fft_gsr_1,
#            'MEDIAN_FFT_GSR_1':median_fft_gsr_1,
#            'RMS_FFT_GSR_1':rms_fft_gsr_1,
#            'STD_FFT_GSR_1':std_fft_gsr_1,
#            'VAR_FFT_GSR_1':var_fft_gsr_1,
#            'SKEW_FFT_GSR_1':skew_fft_gsr_1,
#            'KURT_FFT_GSR_1':kurt_fft_gsr_1,
            'MEAN_FFT_PPG':mean_fft_ppg,
            'MAX_FFT_PPG':max_fft_ppg,
            'MIN_FFT_PPG':min_fft_ppg,
            'MEDIAN_FFT_PPG':median_fft_ppg,
            'RMS_FFT_PPG':rms_fft_ppg,
            'STD_FFT_PPG':std_fft_ppg,
            'VAR_FFT_PPG':var_fft_ppg,
            'SKEW_FFT_PPG':skew_fft_ppg,
            'KURT_FFT_PPG':kurt_fft_ppg})
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
    df['Label_2']=labelb
    df['Value']=labelb_v
    df['Phase']='b'
    return df
    
    
def get_stress_feature(gsr_filename,ppg_filename,ID,labels,labels_v):
    #extracting data
    dataframe_gsr=pd.read_csv(gsr_filename)
    dataframe_ppg=pd.read_csv(ppg_filename)
    dataframe_gsr=get_segment(dataframe_gsr,sample_rate_gsr,20,35)
    dataframe_gsr=dataframe_gsr-(get_scl(dataframe_gsr))
    dataframe_ppg=get_segment(dataframe_ppg,sample_rate_ppg,20,35) 
    cuttoff_gsr=(2*frequency_gsr)/sample_rate_gsr
    cuttoff_ppg=(2*frequency_ppg)/sample_rate_ppg
    variable_sample_gsr=normalize(dataframe_gsr)
    variable_sample_ppg=normalize(dataframe_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
    variable_sample_gsr=variable_sample_gsr.reshape(len(variable_sample_gsr),)
    variable_sample_ppg=variable_sample_ppg.reshape(len(variable_sample_ppg),)
    gsr_1=np.gradient(variable_sample_gsr)
    fft_gsr=abs(fft(variable_sample_gsr))
    fft_gsr_1=abs(fft(gsr_1))
    ppg_1=np.gradient(variable_sample_ppg)
    fft_ppg=abs(fft(variable_sample_ppg))
    fft_ppg_1=abs(fft(ppg_1))
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
    # GST and GSR_1 FFT
    mean_fft_gsr=[]
    max_fft_gsr=[]
    min_fft_gsr=[]
    median_fft_gsr=[]
    rms_fft_gsr=[]
    std_fft_gsr=[]
    var_fft_gsr=[]
    skew_fft_gsr=[]
    kurt_fft_gsr=[]
    mean_fft_gsr_1=[]
    max_fft_gsr_1=[]
    min_fft_gsr_1=[]
    median_fft_gsr_1=[]
    rms_fft_gsr_1=[]
    std_fft_gsr_1=[]
    var_fft_gsr_1=[]
    skew_fft_gsr_1=[]
    kurt_fft_gsr_1=[]
    start=0
    end=int(window_size*sample_rate_gsr)
    overlap=int((window_size*sample_rate_gsr)/2)
    while(end<=len(fft_gsr)):
        mean_fft_gsr.append(np.mean(fft_gsr[start:end]))
        max_fft_gsr.append(np.max(fft_gsr[start:end]))
        min_fft_gsr.append(np.min(fft_gsr[start:end]))
        median_fft_gsr.append(np.median(fft_gsr[start:end]))
        rms_fft_gsr.append(np.sqrt(np.mean(fft_gsr[start:end])**2))
        std_fft_gsr.append(np.std(fft_gsr[start:end]))
        var_fft_gsr.append(np.var(fft_gsr[start:end]))
        skew_fft_gsr.append(skew(fft_gsr[start:end]))
        kurt_fft_gsr.append(kurtosis(fft_gsr[start:end]))
        mean_fft_gsr_1.append(np.mean(fft_gsr_1[start:end]))
        max_fft_gsr_1.append(np.max(fft_gsr_1[start:end]))
        min_fft_gsr_1.append(np.min(fft_gsr_1[start:end]))
        median_fft_gsr_1.append(np.median(fft_gsr_1[start:end]))
        rms_fft_gsr_1.append(np.sqrt(np.mean(fft_gsr_1[start:end])**2))
        std_fft_gsr_1.append(np.std(fft_gsr_1[start:end]))
        var_fft_gsr_1.append(np.var(fft_gsr_1[start:end]))
        skew_fft_gsr_1.append(skew(fft_gsr_1[start:end]))
        kurt_fft_gsr_1.append(kurtosis(fft_gsr_1[start:end]))
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
    
    # PPG and PPG_1 FFT
    mean_fft_ppg=[]
    max_fft_ppg=[]
    min_fft_ppg=[]
    median_fft_ppg=[]
    rms_fft_ppg=[]
    std_fft_ppg=[]
    var_fft_ppg=[]
    skew_fft_ppg=[]
    kurt_fft_ppg=[]
    mean_fft_ppg_1=[]
    max_fft_ppg_1=[]
    min_fft_ppg_1=[]
    median_fft_ppg_1=[]
    rms_fft_ppg_1=[]
    std_fft_ppg_1=[]
    var_fft_ppg_1=[]
    skew_fft_ppg_1=[]
    kurt_fft_ppg_1=[]
    start=0
    end=int(window_size*sample_rate_ppg)
    overlap=int((window_size*sample_rate_ppg)/2)
    while(end<=len(fft_ppg)):
        mean_fft_ppg.append(np.mean(fft_ppg[start:end]))
        max_fft_ppg.append(np.max(fft_ppg[start:end]))
        min_fft_ppg.append(np.min(fft_ppg[start:end]))
        median_fft_ppg.append(np.median(fft_ppg[start:end]))
        rms_fft_ppg.append(np.sqrt(np.mean(fft_ppg[start:end])**2))
        std_fft_ppg.append(np.std(fft_ppg[start:end]))
        var_fft_ppg.append(np.var(fft_ppg[start:end]))
        skew_fft_ppg.append(skew(fft_ppg[start:end]))
        kurt_fft_ppg.append(kurtosis(fft_ppg[start:end]))
        mean_fft_ppg_1.append(np.mean(fft_ppg_1[start:end]))
        max_fft_ppg_1.append(np.max(fft_ppg_1[start:end]))
        min_fft_ppg_1.append(np.min(fft_ppg_1[start:end]))
        median_fft_ppg_1.append(np.median(fft_ppg_1[start:end]))
        rms_fft_ppg_1.append(np.sqrt(np.mean(fft_ppg_1[start:end])**2))
        std_fft_ppg_1.append(np.std(fft_ppg_1[start:end]))
        var_fft_ppg_1.append(np.var(fft_ppg_1[start:end]))
        skew_fft_ppg_1.append(skew(fft_ppg_1[start:end]))
        kurt_fft_ppg_1.append(kurtosis(fft_ppg_1[start:end]))
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
            'MIN_PPG_PROM':min_ppg_prom,
            'RMS_GSR':rms_gsr,
            'PPG_RMS':ppg_rms,
            'MEAN_FFT_GSR':mean_fft_gsr,
            'MAX_FFT_GSR':max_fft_gsr,
            'MIN_FFT_GSR':min_fft_gsr,
            'MEDIAN_FFT_GSR':median_fft_gsr,
            'RMS_FFT_GSR':rms_fft_gsr,
            'STD_FFT_GSR':std_fft_gsr,
            'VAR_FFT_GSR':var_fft_gsr,
            'SKEW_FFT_GSR':skew_fft_gsr,
            'KURT_FFT_GSR':kurt_fft_gsr,
#            'MEAN_FFT_GSR_1':mean_fft_gsr_1,
#            'MAX_FFT_GSR_1':max_fft_gsr_1,
#            'MIN_FFT_GSR_1':min_fft_gsr_1,
#            'MEDIAN_FFT_GSR_1':median_fft_gsr_1,
#            'RMS_FFT_GSR_1':rms_fft_gsr_1,
#            'STD_FFT_GSR_1':std_fft_gsr_1,
#            'VAR_FFT_GSR_1':var_fft_gsr_1,
#            'SKEW_FFT_GSR_1':skew_fft_gsr_1,
#            'KURT_FFT_GSR_1':kurt_fft_gsr_1,
            'MEAN_FFT_PPG':mean_fft_ppg,
            'MAX_FFT_PPG':max_fft_ppg,
            'MIN_FFT_PPG':min_fft_ppg,
            'MEDIAN_FFT_PPG':median_fft_ppg,
            'RMS_FFT_PPG':rms_fft_ppg,
            'STD_FFT_PPG':std_fft_ppg,
            'VAR_FFT_PPG':var_fft_ppg,
            'SKEW_FFT_PPG':skew_fft_ppg,
            'KURT_FFT_PPG':kurt_fft_ppg})
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
    df['Label_2']=labels
    df['Value']=labels_v
    df['Phase']='s'
    return df
    
def get_relax_feature(gsr_filename,ppg_filename,ID,labelr,labelr_v):
    #extracting data
    dataframe_gsr=pd.read_csv(gsr_filename)
    dataframe_ppg=pd.read_csv(ppg_filename)
    total_time=len(dataframe_gsr)/(sample_rate_gsr*60)
    end_time=int(total_time-5)
    dataframe_gsr=get_segment(dataframe_gsr,sample_rate_gsr,45,end_time)
    dataframe_gsr=dataframe_gsr-(get_scl(dataframe_gsr))
    dataframe_ppg=get_segment(dataframe_ppg,sample_rate_ppg,45,end_time) 
    cuttoff_gsr=(2*frequency_gsr)/sample_rate_gsr
    cuttoff_ppg=(2*frequency_ppg)/sample_rate_ppg
    variable_sample_gsr=normalize(dataframe_gsr)
    variable_sample_ppg=normalize(dataframe_ppg)
    variable_sample_gsr=butterworth(variable_sample_gsr,5,cuttoff_gsr)
    variable_sample_ppg=butterworth(variable_sample_ppg,5,cuttoff_ppg)
    variable_sample_gsr=variable_sample_gsr.reshape(len(variable_sample_gsr),)
    variable_sample_ppg=variable_sample_ppg.reshape(len(variable_sample_ppg),)
    gsr_1=np.gradient(variable_sample_gsr)
    fft_gsr=abs(fft(variable_sample_gsr))
    fft_gsr_1=abs(fft(gsr_1))
    ppg_1=np.gradient(variable_sample_ppg)
    fft_ppg=abs(fft(variable_sample_ppg))
    fft_ppg_1=abs(fft(ppg_1))
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
    # GST and GSR_1 FFT
    mean_fft_gsr=[]
    max_fft_gsr=[]
    min_fft_gsr=[]
    median_fft_gsr=[]
    rms_fft_gsr=[]
    std_fft_gsr=[]
    var_fft_gsr=[]
    skew_fft_gsr=[]
    kurt_fft_gsr=[]
    mean_fft_gsr_1=[]
    max_fft_gsr_1=[]
    min_fft_gsr_1=[]
    median_fft_gsr_1=[]
    rms_fft_gsr_1=[]
    std_fft_gsr_1=[]
    var_fft_gsr_1=[]
    skew_fft_gsr_1=[]
    kurt_fft_gsr_1=[]
    start=0
    end=int(window_size*sample_rate_gsr)
    overlap=int((window_size*sample_rate_gsr)/2)
    while(end<=len(fft_gsr)):
        mean_fft_gsr.append(np.mean(fft_gsr[start:end]))
        max_fft_gsr.append(np.max(fft_gsr[start:end]))
        min_fft_gsr.append(np.min(fft_gsr[start:end]))
        median_fft_gsr.append(np.median(fft_gsr[start:end]))
        rms_fft_gsr.append(np.sqrt(np.mean(fft_gsr[start:end])**2))
        std_fft_gsr.append(np.std(fft_gsr[start:end]))
        var_fft_gsr.append(np.var(fft_gsr[start:end]))
        skew_fft_gsr.append(skew(fft_gsr[start:end]))
        kurt_fft_gsr.append(kurtosis(fft_gsr[start:end]))
        mean_fft_gsr_1.append(np.mean(fft_gsr_1[start:end]))
        max_fft_gsr_1.append(np.max(fft_gsr_1[start:end]))
        min_fft_gsr_1.append(np.min(fft_gsr_1[start:end]))
        median_fft_gsr_1.append(np.median(fft_gsr_1[start:end]))
        rms_fft_gsr_1.append(np.sqrt(np.mean(fft_gsr_1[start:end])**2))
        std_fft_gsr_1.append(np.std(fft_gsr_1[start:end]))
        var_fft_gsr_1.append(np.var(fft_gsr_1[start:end]))
        skew_fft_gsr_1.append(skew(fft_gsr_1[start:end]))
        kurt_fft_gsr_1.append(kurtosis(fft_gsr_1[start:end]))
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
    
    # PPG and PPG_1 FFT
    mean_fft_ppg=[]
    max_fft_ppg=[]
    min_fft_ppg=[]
    median_fft_ppg=[]
    rms_fft_ppg=[]
    std_fft_ppg=[]
    var_fft_ppg=[]
    skew_fft_ppg=[]
    kurt_fft_ppg=[]
    mean_fft_ppg_1=[]
    max_fft_ppg_1=[]
    min_fft_ppg_1=[]
    median_fft_ppg_1=[]
    rms_fft_ppg_1=[]
    std_fft_ppg_1=[]
    var_fft_ppg_1=[]
    skew_fft_ppg_1=[]
    kurt_fft_ppg_1=[]
    start=0
    end=int(window_size*sample_rate_ppg)
    overlap=int((window_size*sample_rate_ppg)/2)
    while(end<=len(fft_ppg)):
        mean_fft_ppg.append(np.mean(fft_ppg[start:end]))
        max_fft_ppg.append(np.max(fft_ppg[start:end]))
        min_fft_ppg.append(np.min(fft_ppg[start:end]))
        median_fft_ppg.append(np.median(fft_ppg[start:end]))
        rms_fft_ppg.append(np.sqrt(np.mean(fft_ppg[start:end])**2))
        std_fft_ppg.append(np.std(fft_ppg[start:end]))
        var_fft_ppg.append(np.var(fft_ppg[start:end]))
        skew_fft_ppg.append(skew(fft_ppg[start:end]))
        kurt_fft_ppg.append(kurtosis(fft_ppg[start:end]))
        mean_fft_ppg_1.append(np.mean(fft_ppg_1[start:end]))
        max_fft_ppg_1.append(np.max(fft_ppg_1[start:end]))
        min_fft_ppg_1.append(np.min(fft_ppg_1[start:end]))
        median_fft_ppg_1.append(np.median(fft_ppg_1[start:end]))
        rms_fft_ppg_1.append(np.sqrt(np.mean(fft_ppg_1[start:end])**2))
        std_fft_ppg_1.append(np.std(fft_ppg_1[start:end]))
        var_fft_ppg_1.append(np.var(fft_ppg_1[start:end]))
        skew_fft_ppg_1.append(skew(fft_ppg_1[start:end]))
        kurt_fft_ppg_1.append(kurtosis(fft_ppg_1[start:end]))
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
            'MIN_PPG_PROM':min_ppg_prom,
            'RMS_GSR':rms_gsr,
            'PPG_RMS':ppg_rms,
            'MEAN_FFT_GSR':mean_fft_gsr,
            'MAX_FFT_GSR':max_fft_gsr,
            'MIN_FFT_GSR':min_fft_gsr,
            'MEDIAN_FFT_GSR':median_fft_gsr,
            'RMS_FFT_GSR':rms_fft_gsr,
            'STD_FFT_GSR':std_fft_gsr,
            'VAR_FFT_GSR':var_fft_gsr,
            'SKEW_FFT_GSR':skew_fft_gsr,
            'KURT_FFT_GSR':kurt_fft_gsr,
#            'MEAN_FFT_GSR_1':mean_fft_gsr_1,
#            'MAX_FFT_GSR_1':max_fft_gsr_1,
#            'MIN_FFT_GSR_1':min_fft_gsr_1,
#            'MEDIAN_FFT_GSR_1':median_fft_gsr_1,
#            'RMS_FFT_GSR_1':rms_fft_gsr_1,
#            'STD_FFT_GSR_1':std_fft_gsr_1,
#            'VAR_FFT_GSR_1':var_fft_gsr_1,
#            'SKEW_FFT_GSR_1':skew_fft_gsr_1,
#            'KURT_FFT_GSR_1':kurt_fft_gsr_1,
            'MEAN_FFT_PPG':mean_fft_ppg,
            'MAX_FFT_PPG':max_fft_ppg,
            'MIN_FFT_PPG':min_fft_ppg,
            'MEDIAN_FFT_PPG':median_fft_ppg,
            'RMS_FFT_PPG':rms_fft_ppg,
            'STD_FFT_PPG':std_fft_ppg,
            'VAR_FFT_PPG':var_fft_ppg,
            'SKEW_FFT_PPG':skew_fft_ppg,
            'KURT_FFT_PPG':kurt_fft_ppg})
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
    df['Value']=labelr_v
    df['Phase']='r'
    return df
    
        
    
    
    

sample_rate_gsr=4
sample_rate_ppg=64
sample_rate_hr=1
frequency_gsr=1
frequency_ppg=10
window_size=30
    
subject_id=[1,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,
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
   
for i in subject_id:
    ID=i
    if(i<10):
        gsr_filename='ST00'+str(i)+'_EDA.csv'
        ppg_filename='ST00'+str(i)+'_BVP.csv'
        output_file='ST00'+str(i)+'_processed.csv'
#        label_list='st'+str(i)
    if(i>=10):
        gsr_filename='ST0'+str(i)+'_EDA.csv'
        ppg_filename='ST0'+str(i)+'_BVP.csv'
        output_file='ST0'+str(i)+'_processed.csv'
#        label_list='st'+str(i)
    labelb=subject_label[i][0]
    labels=subject_label[i][1]
    labelr=subject_label[i][2]
    
    labelb_v=subject_label_v[i][0]
    labels_v=subject_label_v[i][1]
    labelr_v=subject_label_v[i][2]  
    
    df_baseline=get_baseline_feature(gsr_filename,ppg_filename,ID,labelb,labelb_v)
    df_baseline=df_baseline.fillna(np.mean(df_baseline))
    df_baseline=df_baseline.fillna(0)
    df_stress=get_stress_feature(gsr_filename,ppg_filename,ID,labels,labels_v)
    df_stress=df_stress.fillna(np.mean(df_stress))
    df_stress=df_stress.fillna(0)
#    df_relax=get_relax_feature(gsr_filename,ppg_filename,ID,labelr,labelr_v)
#    df_relax=df_relax.fillna(np.mean(df_relax))
#    df_relax=df_relax.fillna(0)
    
#    df_all = pd.concat([df_baseline,df_stress,df_relax], axis=0, join='outer', ignore_index=False)
    df_all = pd.concat([df_baseline,df_stress], axis=0, join='outer', ignore_index=False)
    df_all.to_csv(output_file)
    print(df_all)
    
    
    
    
    
    