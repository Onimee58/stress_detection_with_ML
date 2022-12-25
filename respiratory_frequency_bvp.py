# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:58:21 2022

@author: Admin
"""

# library import
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.signal as signal
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Function Definitions

# Data Import Function. Generate xlrd workbook and then using pandas read_excel method to generate the dataframe. Inputs= path where the file is 
# located and the filename
def read_datafiles(path,filename):
    filepath=path+filename
    workbook=xlrd.open_workbook_xls(filepath, ignore_workbook_corruption=True)
    dataframe=pd.read_excel(workbook)
    return dataframe
# Function for nomalizing. Returns the normalized values within 0 and 1
def normalize(dataframe):
    dataframe=(dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
    return dataframe
# Function to calculate sampling rate. Sampling rate is average number of samples collected per second. Hence can be obtained by calculating 
# the mean of the discrete difference of times (in seconds). 
def calculate_sample_rate(dataframe_signal):
    time=dataframe_signal['time']
    d_time=time.diff().mean()
    sample_frequency=int(1/d_time)
    return int(sample_frequency)
# Function for butterworth filter
def butterworth(raw_signal,n,desired_cutoff,sample_rate,btype):
    if(btype=='high' or btype=='low'):
        critical_frequency=(2*desired_cutoff)/sample_rate
        B, A = signal.butter(n, critical_frequency, btype=btype, output='ba')
    elif(btype=='bandpass'):
        critical_frequency_1=(2*desired_cutoff[0])/sample_rate
        critical_frequency_2=(2*desired_cutoff[1])/sample_rate
        B, A = signal.butter(n, [critical_frequency_1,critical_frequency_2], btype=btype, output='ba')
    filtered_signal = signal.filtfilt(B,A, raw_signal)
    return filtered_signal
# Function for estimating respiratory frequency from BVP signal. Calculate the frequency associated with maximum power in a given window of 
# BVP signal (window_length) and outputs the respiratory rate after a fixed interval (time_to_output).
def get_estimated_frequencies(bvp_signal_segment, window_length,time_to_output,sample_rate):
    estimated_frequencies=[] # Contains the list of predicted frequencies
    start=0
    end=int(window_length*sample_rate_bvp)
    overlap=int(time_to_output*sample_rate_bvp)
    while(end<len(bvp_filtered)):
        sample_frequency, power_spectrum=signal.periodogram(bvp_signal_segment[start:end], sample_rate_bvp)
        # You can modify other parameters such as window or nfft to get better precision in the estimation of power spectrum using the periodohgram method.
        # You can also explore other power spectral method. And ofcourse, you can look into existing literature for other methods. 
        index_max,=np.where(power_spectrum==np.max(power_spectrum))
        estimated_frequencies.append(sample_frequency[index_max])
        start=start+overlap
        end=end+overlap
    return estimated_frequencies

# Data import. Specify the path and the filenames for BVP and respiratory signals. df_bvp=dataframe containing BVP signal and
# df_reference_frequencies=dataframe containing respiratory frequency signal
path=r'C:\Users\Admin\Documents\Applications\Empatica\Assignment\Data\\'
filename_bvp='BVP.xls'
filename_reference_frequencies='respiratory_frequency.xls'
df_bvp=read_datafiles(path,filename_bvp)
df_reference_frequencies=read_datafiles(path,filename_reference_frequencies)

# Getting data ready for plotting. Extract the bvp, respiratory frequency, and the time data points in three different variables
bvp=df_bvp['BVP']
reference_frequencies=df_reference_frequencies['respiratory_frequency_Hz']
time_bvp=df_bvp['time']
time_reference_frequencies=df_reference_frequencies['time']
# Normalize BVP and rf data so that plotting the BVP and rf signals together makes sense
bvp_normalized=normalize(bvp)
reference_frequencies_normalized=normalize(reference_frequencies)
# Plot the two normalized signals together. Generates Figure 1.
figure(figsize=(16, 6), dpi=160)
plt.grid(axis='both',linewidth=2)
plt.xlabel('Time (seconds)', fontsize=18)
plt.ylabel('Values', fontsize=16)
plt.plot(time_bvp,bvp_normalized,'red',linewidth=3.0,label='BVP')
plt.plot(time_reference_frequencies,reference_frequencies_normalized,'blue',linewidth=3.0,label='Respiratory Frequency')
plt.legend()
plt.savefig('visualization.png')
plt.show()


# Calculating sampling rate of BVP signal. Respiratory frequency signal presented here is a frequency and not sampled using any sensor, rather 
# it is derived by processing of capnography signal. Hence calculation of sampling rate of respiratory frequency signal does not makes sense here.
sample_rate_bvp=calculate_sample_rate(df_bvp)

# Preprocessing. Do a bandpass filtering between 0.2 and 0.8 Hz on the BVP signal
bvp_filtered=butterworth(bvp,3,[0.2,0.8],sample_rate_bvp,btype='bandpass')

# Generates Figure 2. Unfiltered BVP signal.
figure(figsize=(16, 6), dpi=160)
plt.grid(axis='both',linewidth=2)
plt.xlabel('Time (seconds)', fontsize=18)
plt.ylabel('Values', fontsize=16)
plt.plot(time_bvp,bvp,'red',linewidth=3.0,label='Raw BVP Signal')
plt.legend()
plt.savefig('unfiltered.png')
plt.show()

# Generates Figure 3. Filtered BVP signal.
figure(figsize=(16, 6), dpi=160)
plt.grid(axis='both',linewidth=2)
plt.xlabel('Time (seconds)', fontsize=18)
plt.ylabel('Values', fontsize=16)
plt.plot(time_bvp,bvp_filtered,'red',linewidth=3.0,label='Filtered BVP Signal')
plt.legend()
plt.savefig('filtered.png')
plt.show()


# Processing. Estimate the frequencies from BVP signal. Here the minimum observing window (window_length) is set to 30 seconds and 
# output time (time_to_output) is set to 5 seconds.
window_length=30
time_to_output=5
estimated_frequencies=get_estimated_frequencies(bvp_filtered,window_length,time_to_output,sample_rate_bvp)

# Generate respiratory rates from respiratory frequencies for both estimated and reference.
estimated_respiratory_rate=list(np.asarray(estimated_frequencies).flatten()*60)
reference_respiratory_rate=list(reference_frequencies*60)
# Merge every quantities of comparison in a dataframe for performance analysis
estimated_frequencies=list(np.asarray(estimated_frequencies).flatten())
df_results=pd.DataFrame({
    'Reference': reference_respiratory_rate,
    'Estimated': estimated_respiratory_rate,
    'Reference_frequency':reference_frequencies,
    'Estimated_frequencies':estimated_frequencies})

# Calculate the performance metrices. 
print('RMSE of respiratory rates', mean_squared_error(df_results['Reference'], df_results['Estimated'], squared=False))
print('RMSE of respiratory frequencies', mean_squared_error(df_results['Reference_frequency'], df_results['Estimated_frequencies'], squared=False))
print('P-value between estimated and reference',mannwhitneyu(df_results['Reference'],df_results['Estimated'])[1])
print('P-value between estimated and reference frequencies',mannwhitneyu(df_results['Reference_frequency'],df_results['Estimated_frequencies'])[1])
# Generate Bland-Altman Plot (Figure 4)
sm.graphics.mean_diff_plot(df_results['Reference'],df_results['Estimated'])
plt.savefig('bland-altman.png')
plt.show()

# Respiratory rates and frequencies without filtering
estimated_frequencies_without_filter=get_estimated_frequencies(bvp,window_length,time_to_output,sample_rate_bvp)
estimated_respiratory_rate_without_filter=list(np.asarray(estimated_frequencies_without_filter).flatten()*60)
# Append respiratory rates and frequencies without filtering to the results dataframe
df_results['Estimated_Frequencies_Without_Filter']=estimated_frequencies_without_filter
df_results['Estimated_Respiratory_Rate_Without_Filter']=estimated_respiratory_rate_without_filter
# Calculate the performance metrices without filter
print('RMSE of respiratory rates', mean_squared_error(df_results['Reference'], df_results['Estimated_Respiratory_Rate_Without_Filter'], squared=False))
print('RMSE of respiratory frequencies', mean_squared_error(df_results['Reference_frequency'], df_results['Estimated_Frequencies_Without_Filter'], squared=False))
print('P-value between estimated and reference',mannwhitneyu(df_results['Reference'],df_results['Estimated_Respiratory_Rate_Without_Filter'])[1])
print('P-value between estimated and reference',mannwhitneyu(df_results['Reference_frequency'],df_results['Estimated_Respiratory_Rate_Without_Filter'])[1])

# Plot the estimated frequencies with and without filtering and reference frequencies for visualization. Generates Figure 3
figure(figsize=(16, 6), dpi=160)
plt.grid(axis='both',linewidth=2)
plt.xlabel('Time (seconds)', fontsize=18)
plt.ylabel('Respiration Rate', fontsize=16)
plt.plot(time_reference_frequencies,estimated_respiratory_rate,'blue',linewidth=3.0,label='Estimated')
plt.plot(time_reference_frequencies,reference_respiratory_rate,'red',linewidth=3.0,label='Reference')
plt.plot(time_reference_frequencies,estimated_respiratory_rate_without_filter,'black',linewidth=3.0,label='Estimated without filter')
plt.legend()
plt.savefig('comparison.png')
plt.show()