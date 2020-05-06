#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 04:07:25 2020

@author: myron
"""

import pyleoclim as pyleo

#%%step 1: data generation

import numpy as np
import matplotlib.pyplot as plt

# Generate a mixed signal with known frequencies
freqs=[1/20,1/80]
time=np.arange(2001)
signals=[]
for freq in freqs:
    signals.append(np.cos(2*np.pi*freq*time))
signal=sum(signals)

# Add outliers

#outliers_start = np.mean(signal)+5*np.std(signal)
#outliers_end = np.mean(signal)+7*np.std(signal)
#outlier_values = np.arange(outliers_start,outliers_end,0.1)
#index = np.random.randint(0,len(signal),6)
#signal_out = signal
#for i,ind in enumerate(index):
#    signal_out[ind] = outlier_values[i]

# Add a non-linear trend
slope = 1e-5
intercept = -1
nonlinear_trend = slope*time**2 + intercept
signal_trend = signal + nonlinear_trend
#signal_trend = signal_out + nonlinear_trend

#Add white noise
sig_var = np.var(signal)
noise_var = sig_var / 2 #signal is twice the size of noise
white_noise = np.random.normal(0, np.sqrt(noise_var), size=np.size(signal))
signal_noise = signal_trend + white_noise

#Remove data points
del_percent = 0.4
n_del = int(del_percent*np.size(time))
deleted_idx = np.random.choice(range(np.size(time)), n_del, replace=False)
time_unevenly =  np.delete(time, deleted_idx)
signal_unevenly =  np.delete(signal_noise, deleted_idx)


#Plot
plt.plot(time_unevenly,signal_unevenly)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
#%%= step 2: create series object
ts=pyleo.Series(time=time_unevenly,value=signal_unevenly)
fig,ax=ts.plot()

#%% preprocessing- standardize
ts_std=ts.standardize()
fig,ax=ts_std.plot()
#%% preprocessing - detrend
    
ts_detrended=ts_std.detrend(method='emd')
fig,ax=ts_detrended.plot()
#%% detect and remove outliers
is_outlier=ts_detrended.detect_outliers()
if len(np.where(is_outlier==True))>0:
    ts_outliers = ts_detrended.remove_outliers()
else:
    ts_outliers=ts_detrended
#%% interpolation
ts_interp=ts_outliers.interp(method='linear')
fig,ax=ts_interp.plot()

#%% Analysis-cwt

#error, the coefs is not a 2d array
scales = np.arange(1, 128) #not sure what scales to use
cwt_res=ts_interp.wavelet(method='cwt',settings={'scales':scales})
cwt_res.plot(title='cwt')
plt.show()
#%% Analysis - wwz
wwz_res=ts_interp.wavelet(method='wwz',settings={})
wwz_res.plot(title='wwz')
plt.show()
#%% signif test- cwt
cwt_signif=cwt_res.signif_test(qs=[0.95])
fig,ax=cwt_signif.plot(title='cwt analysis')
plt.show()
#%% signif test -wwz
wwz_signif=wwz_res.signif_test(qs=[0.95])
fig,ax=wwz_signif.plot(title='wwz analysis')
plt.show()
