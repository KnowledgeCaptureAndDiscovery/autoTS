#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:50:00 2020

@author: myron
"""
import numpy as np
import pandas as pd
from pyleoclim.utils import spectral,decomposition
from run_algs import run_welch,run_periodogram,run_mtm,add_noise,cost_function
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,peak_widths,peak_prominences
from scipy.optimize import linear_sum_assignment
from run_algs import del_points
#%% normal freq welch widths
time = np.arange(2001)
f = 1/50
signal = np.cos(2*np.pi*f*time)


series = pd.Series(signal, index=time)

fig, res_psd,widths = run_welch(series,actual_freq=[f])
cost=cost_function(res_psd,[f])
print(cost)


#%%welch widths 2 close freq
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])
time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)
fig, res_psd, widths= run_welch(series,actual_freq=[f1,f2])
index=find_peaks(res_psd['psd'])

print(cost_function(res_psd,[f1,f2],0))
#%%welch widths with white noise+pure freq
time = np.arange(2001)
f = 1/50
signal = add_noise(np.cos(2*np.pi*f*time))


series = pd.Series(signal, index=time)

fig, res_psd,widths = run_welch(series,actual_freq=[f])
plt.show()
print(cost_function(res_psd,[f1],0,0))

#%%welch widths white noise + 2 freq

time1 = np.arange(1000)
f1 = 1/50
signal1 = add_noise(np.cos(2*np.pi*f1*time1))

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = add_noise(np.cos(2*np.pi*f2*time2))

signal = np.concatenate([signal1, signal2])
time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

fig, res_psd, widths= run_welch(series,actual_freq=[f1,f2])
print(cost_function(res_psd,[f1,f2],0))

#%% periodogram widths pure freq
time = np.arange(2001)
f = 1/50
signal = np.cos(2*np.pi*f*time)


series = pd.Series(signal, index=time)

fig, res_psd,widths = run_periodogram(series)
print(sum(widths[0]/len(widths[0])))
#%%periodogram widths 2 close freq
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])
time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

fig, res_psd, widths= run_periodogram(series,actual_freq=[f1,f2])
print(cost_function(res_psd,[f1,f2]))

#%% periodogram widths pure freq+white noise
time = np.arange(2001)
f = 1/50
signal = add_noise(np.cos(2*np.pi*f*time))


series = pd.Series(signal, index=time)

fig, res_psd,widths = run_periodogram(series)
#%% periodogram widths 2 close freq+white noise

time1 = np.arange(1000)
f1 = 1/50
signal1 = add_noise(np.cos(2*np.pi*f1*time1))

time2 = np.arange(1000, 2001)

f2 = 1/55

signal2 = add_noise(np.cos(2*np.pi*f2*time2))

signal = np.concatenate([signal1, signal2])
time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

fig, res_psd, widths= run_periodogram(series,actual_freq=[f1,f2])
print(cost_function(res_psd,[f1,f2]))


#%% test cost func, compares ration of (avg prominence/avg width). higher ratio is better.
time1 = np.arange(1000)
f1 = 1/50
signal1 = add_noise(np.cos(2*np.pi*f1*time1))

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = add_noise(np.cos(2*np.pi*f2*time2))

signal = np.concatenate([signal1, signal2])
time = np.concatenate([time1, time2])
series = pd.Series(signal, index=time)
res_psd=spectral.welch(signal,time,prep_args={'detrend': False})

res_psd2 = spectral.periodogram(signal, time, prep_args={'detrend': False})   
print(cost_function(res_psd,[f1,f2]))
fig, res_psd,widths = run_welch(series,actual_freq=[f1,f2])
print(cost_function(res_psd2,[f1,f2]))
fig, res_psd,widths = run_periodogram(series,actual_freq=[f1,f2])
 
#%% welch test nperseg and len(ts)//n up to n==6 
#perfect signal

'''
'''
time = np.arange(2001)
f = 1/50
signal = add_noise(np.cos(2*np.pi*f*time))
series = pd.Series(signal, index=time)

max_cost=0
costs=[]
for n in range(1,8):
    fig,res_psd,widths = run_welch(series,ana_args={'nperseg' : len(time)//n},actual_freq=[f],dist_tol=0.2*f,peak_tol=0)
    num_peaks,acc,ratio=cost_function(res_psd,[f])
    plt.title(str(n)+' '+str(acc))
    plt.show()
    costs.append((n,acc,ratio))
costs.sort(key=lambda x: (x[1],-x[2]))
for i,x in enumerate(costs):
    print(x[0],x[1],x[2])

best_nperseg=costs[0][0]
print('best n for nperseg:',best_nperseg)
#%% welch nperseg, noverlap single pure freq

time = np.arange(1001)
f = 1/50
#signal = np.cos(2*np.pi*f*time)
signal = add_noise(np.cos(2*np.pi*f*time))
series = pd.Series(signal, index=time)
n=8 
costs=[]
tol=0
for nperseg_n in range(1,n-1):
    for noverlap_n in range(nperseg_n+1,n):
        fig,res_psd,widths = run_welch(series,ana_args={'nperseg' : len(time)//nperseg_n,'noverlap': len(time)//noverlap_n},actual_freq=[f],peak_tol=1)
        correct_peaks,acc,ratio=cost_function(res_psd,[f],tol,0)
        plt.title(str(nperseg_n)+', '+str(noverlap_n)+': '+str(acc))
        plt.show()
        costs.append((correct_peaks,(nperseg_n,noverlap_n),acc,ratio))
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)

#%% test welch nperseg noverlap 2 signals
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])

time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

n=8 
costs=[]
tol=0
for nperseg_n in range(1,n-1):
    for noverlap_n in range(nperseg_n+1,n):
        fig,res_psd,widths = run_welch(series,ana_args={'nperseg' : len(time)//nperseg_n,'noverlap': len(time)//noverlap_n},actual_freq=[f1,f2],peak_tol=1)
        correct_peaks,acc,ratio=cost_function(res_psd,[f1,f2],tol,0)
        plt.title(str(nperseg_n)+', '+str(noverlap_n)+': '+str(acc))
        plt.show()
        costs.append((correct_peaks,(nperseg_n,noverlap_n),acc,ratio))
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)


#%% test both with white noise
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])
signal = add_noise(signal)


time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

n=8 
costs=[]
tol=0
for nperseg_n in range(1,n-1):
    for noverlap_n in range(nperseg_n+1,n):
        fig,res_psd,widths = run_welch(series,ana_args={'nperseg' : len(time)//nperseg_n,'noverlap': len(time)//noverlap_n},actual_freq=[f1,f2],peak_tol=1)
        correct_peaks,acc,ratio=cost_function(res_psd,[f1,f2],tol,0)
        plt.title(str(nperseg_n)+', '+str(noverlap_n)+': '+str(acc))
        plt.show()
        costs.append((correct_peaks,(nperseg_n,noverlap_n),acc,ratio))
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)


#%%test periodogram window -single pur freq
time = np.arange(1001)
f = 1/50
#signal = np.cos(2*np.pi*f*time)
signal = add_noise(np.cos(2*np.pi*f*time))
series = pd.Series(signal, index=time)


costs=[]
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
windows=['boxcar','triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']
for win in windows:
    fig,res_psd,widths = run_periodogram(series,ana_args= {'window' : win }, actual_freq=[f], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f],dist_tol=tol,peak_tol=0)
    plt.title(win+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,win ,acc,ratio,win))    
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)
#%% test periodogram window - two close freqs, if the first peak is below the mean proominence of the peaks, then it is not detected...
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])

time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)



costs=[]
tol=0
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
windows=['boxcar','triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']
for win in windows:
    fig,res_psd,widths = run_periodogram(series,ana_args= {'window' : win }, actual_freq=[f1,f2], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f1,f2],dist_tol=tol,peak_tol=0)
    plt.title(win+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,win ,acc,ratio,win))
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
        print(x)
#%% test periodogram two close freqs+ white noisetime1 = np.arange(1000)
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])
signal = add_noise(signal)


time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)
costs=[]
tol=0
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
windows=['boxcar','triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann']
for win in windows:
    fig,res_psd,widths = run_periodogram(series,ana_args= {'window' : win }, actual_freq=[f1,f2], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f1,f2],dist_tol=tol,peak_tol=0)
    plt.title(win+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,win ,acc,ratio,win))
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)
#%% testrun mtm
time = np.arange(2001)
f = 1/50
signal = np.cos(2*np.pi*f*time)


series = pd.Series(signal, index=time)

fig, res_psd,widths = run_mtm(series,actual_freq=[f])
cost=cost_function(res_psd,[f])
plt.show()    
#%% tets mtm adaptive on single freq
time = np.arange(1001)
f = 1/50
#signal = np.cos(2*np.pi*f*time)
signal = add_noise(np.cos(2*np.pi*f*time))
series = pd.Series(signal, index=time)
tol=0

costs=[]
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
for adpt in [True,False]:
    fig,res_psd,widths = run_mtm(series,ana_args= {'adaptive' : adpt }, actual_freq=[f], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f],dist_tol=tol,peak_tol=0)
    plt.title(str(adpt)+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,adpt ,acc,ratio))    
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)
#%% test mtm adaptive on 2 close freq
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)

f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])

time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)

tol=0

costs=[]
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
for adpt in [True,False]:
    fig,res_psd,widths = run_mtm(series,ana_args= {'adaptive' : adpt }, actual_freq=[f], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f],dist_tol=tol,peak_tol=0)
    plt.title(str(adpt)+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,adpt ,acc,ratio))    
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)
#%% 2 close freq with noise mtm adaptive
time1 = np.arange(1000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time1)

time2 = np.arange(1000, 2001)
f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time2)

signal = np.concatenate([signal1, signal2])

signal = add_noise(signal)


time = np.concatenate([time1, time2])

series = pd.Series(signal, index=time)
series.plot()
plt.show()
tol=0

costs=[]
#windows that need parameters:
# 'kaiser', 'gaussian' , 'general_gaussian', 'slepian', 'chebwin'
for adpt in [True,False]:
    fig,res_psd,widths = run_mtm(series,ana_args= {'adaptive' : adpt }, actual_freq=[f], peak_tol=0)
    correct_peaks,acc,ratio=cost_function(res_psd,[f],dist_tol=tol,peak_tol=0)
    plt.title(str(adpt)+': ' +str(acc)+' '+str(ratio))
    plt.show()
    costs.append((correct_peaks,adpt ,acc,ratio))    
costs.sort(key=lambda x: (-x[0],x[2],-x[3]))
for i,x in enumerate(costs):
    print(x)
    #%%
time=np.arange(1000)
f1=1/50
f2=1/20
signal1=np.cos(2*np.pi*f1*time)
signal2=np.cos(2*np.pi*f2*time)
signal=signal1+signal2
series=pd.Series(signal,index=time)
series.plot()
plt.show()
#%% test ssa with no missing data
time = np.arange(2001)
f = 1/50
signal = np.cos(2*np.pi*f*time)


series = pd.Series(signal, index=time)

res_dict=decomposition.ssa(series,time,2)
#%% test ssa with 5 freqs pure
time = np.arange(2000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time)


f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time)

f3 = 1/20

signal3 = np.cos(2*np.pi*f3*time)
f4 = 1/75

signal4 = np.cos(2*np.pi*f4*time)
f5 = 1/30

signal5 = np.cos(2*np.pi*f5*time)

signal = signal1+signal2+signal3+signal4+signal5

series = pd.Series(signal, index=time)
series.plot()
plt.show()
res_dict=decomposition.ssa(series,time,2)


#%% test ssa with 5 freqs with missing values
time = np.arange(2000)
f1 = 1/50
signal1 = np.cos(2*np.pi*f1*time)


f2 = 1/55
signal2 = np.cos(2*np.pi*f2*time)

f3 = 1/20

signal3 = np.cos(2*np.pi*f3*time)
f4 = 1/75

signal4 = np.cos(2*np.pi*f4*time)
f5 = 1/30

signal5 = np.cos(2*np.pi*f5*time)

signal = signal1+signal2+signal3+signal4+signal5
signal,time=del_points(signal,time,0.30)
series = pd.Series(signal, index=time)
series.plot()
plt.show()
res_dict=decomposition.ssa(series,time,2)