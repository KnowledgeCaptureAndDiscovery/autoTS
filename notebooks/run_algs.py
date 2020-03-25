#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:29:55 2020

@author: myron
"""
import numpy as np
from pyleoclim.utils import spectral
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.signal import find_peaks,peak_widths,peak_prominences
from statistics import mean
from scipy.optimize import linear_sum_assignment
import itertools

def run_periodogram(series, xlim=[0, 0.05], label='periodogram', loglog=False, title=None, detrend=False,actual_freq=[],ana_args={},peak_tol=0,dist_tol=0):
    time = series.index.values
    signal = series.values
    #tau = np.linspace(np.min(time), np.max(time), 51)
    #res_psd = spectral.wwz_psd(signal, time, tau=tau, nMC=0, standardize=False, detrend=detrend)
    res_psd = spectral.periodogram(signal, time, prep_args={'detrend': detrend},ana_args=ana_args)   
    sns.set(style='ticks', font_scale=1.5)
    fig = plt.figure(figsize=[5, 5])       

    peaks,h=find_peaks(res_psd['psd'],height=0)
    peak_tol=peak_tol*mean(h['peak_heights'])
    prom,_,__=peak_prominences(res_psd['psd'],peaks)
    prom_thresh=mean(prom)*peak_tol
    peaks,props=find_peaks(res_psd['psd'],prominence=prom_thresh,height=0)
    ax_spec = plt.subplot(1, 1, 1)
    if loglog:
        ax_spec.loglog(res_psd['freq'], res_psd['psd'], lw=3, label=label)        
                                                                                                                                              
    else:
        ax_spec.plot(res_psd['freq'], res_psd['psd'], lw=3, label=label)
        ax_spec.set_xlim(xlim)
    #plot peaks
    ax_spec.plot(res_psd['freq'][peaks],res_psd['psd'][peaks],'x')
    for f in actual_freq:
        ax_spec.axvline(x=f, color='k', ls='--', lw=2, label='analytical')

    #find peak widths
    widths=peak_widths(res_psd['psd'],peaks,rel_height=0.99)
    y=res_psd['psd']
    x=res_psd['freq']
    plt.hlines(y=widths[1],xmin=(widths[2]/len(y))*x[-1],xmax=(widths[3]/len(y))*x[-1],colors='r')
    ax_spec.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_spec.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_spec.set_ylabel('Spectral Density')                                                                                                                                                 
    ax_spec.set_xlabel('Frequency')                                                                                                                                                   
    ax_spec.legend(frameon=False)
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['top'].set_visible(False)
    
    return fig, res_psd,widths

def run_mtm(series, xlim=[0, 0.05], label='mtm', loglog=False, title=None, detrend=False,actual_freq=[],ana_args={},peak_tol=0,dist_tol=0):
    time = series.index.values
    signal = series.values
    #tau = np.linspace(np.min(time), np.max(time), 51)
    #res_psd = spectral.wwz_psd(signal, time, tau=tau, nMC=0, standardize=False, detrend=detrend)
    res_psd=spectral.mtm(signal,time,NW=1,prep_args={'detrend': detrend},ana_args=ana_args)
    sns.set(style='ticks', font_scale=1.5)
    fig = plt.figure(figsize=[5, 5])                                                                                                                                           

    peaks,_=find_peaks(res_psd['psd'])
    prom,l,r=peak_prominences(res_psd['psd'],peaks)    
    prom_thresh=mean(prom)
    
    peaks,_=find_peaks(res_psd['psd'],prominence=prom_thresh) #what should the prominence threshold be? too high and relevant peaks will not be present
    ax_spec = plt.subplot(1, 1, 1)
    if loglog:
        ax_spec.loglog(res_psd['freq'], res_psd['psd'], lw=3, label=label)        
                                                                                                                                              
    else:
        ax_spec.plot(res_psd['freq'], res_psd['psd'], lw=3, label=label)
        ax_spec.set_xlim(xlim)
    #plot peaks
    ax_spec.plot(res_psd['freq'][peaks],res_psd['psd'][peaks],'x')
    for f in actual_freq:
        ax_spec.axvline(x=f, color='k', ls='--', lw=2, label='analytical')
    #find peak widths
    widths=peak_widths(res_psd['psd'],peaks,rel_height=0.99)
    y=res_psd['psd']
    x=res_psd['freq']
    plt.hlines(y=widths[1],xmin=(widths[2]/len(y))*x[-1],xmax=(widths[3]/len(y))*x[-1],colors='r')
    ax_spec.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_spec.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_spec.set_ylabel('Spectral Density')                                                                                                                                                 
    ax_spec.set_xlabel('Frequency')                                                                                                                                                   
    ax_spec.legend(frameon=False)
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['top'].set_visible(False)
    
    return fig, res_psd,widths
def run_welch(series, xlim=[0, 0.05], label='welch', loglog=False, title=None, detrend=False,ana_args={},actual_freq=[],peak_tol=0,dist_tol=0):
    time = series.index.values
    signal = series.values
    #tau = np.linspace(np.min(time), np.max(time), 51)
    #res_psd = spectral.wwz_psd(signal, time, tau=tau, nMC=0, standardize=False, detrend=detrend)
    res_psd=spectral.welch(signal,time,prep_args={'detrend': detrend},ana_args=ana_args)
   
    sns.set(style='ticks', font_scale=1.5)
    fig = plt.figure(figsize=[5, 5])                                                                                                                                           

    peaks,_=find_peaks(res_psd['psd'])
    prom,l,r=peak_prominences(res_psd['psd'],peaks)
    prom_thresh=mean(prom)#prominence thresehold? use harmonic mean to get lower avg
    peaks,_=find_peaks(res_psd['psd'],prominence=prom_thresh) #what should the prominence threshold be? too high and relevant peaks will not be present
    ax_spec = plt.subplot(1, 1, 1)
    if loglog:
        ax_spec.loglog(res_psd['freq'], res_psd['psd'], lw=3, label=label)        
                                                                                                                                              
    else:
        ax_spec.plot(res_psd['freq'], res_psd['psd'], lw=3, label=label)
        ax_spec.set_xlim(xlim)
    #plot peaks
    ax_spec.plot(res_psd['freq'][peaks],res_psd['psd'][peaks],'x')
    for f in actual_freq:
        ax_spec.axvline(x=f, color='k', ls='--', lw=2, label='analytical')
    #find peak widths
    widths=peak_widths(res_psd['psd'],peaks,rel_height=0.99)
    y=res_psd['psd']
    x=res_psd['freq']
    plt.hlines(y=widths[1],xmin=(widths[2]/len(y))*x[-1],xmax=(widths[3]/len(y))*x[-1],colors='r')
    ax_spec.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_spec.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_spec.set_ylabel('Spectral Density')                                                                                                                                                 
    ax_spec.set_xlabel('Frequency')                                                                                                                                                   
    ax_spec.legend(frameon=False)
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['top'].set_visible(False)
    
    return fig, res_psd,widths

def add_noise(signal):
    np.random.seed(2333)
    sig_var=np.var(signal)
    noise_var=sig_var/0.5 #SNR=0.5
    white_noise=np.random.normal(0,np.sqrt(noise_var),size=np.size(signal))
    signal_noise=signal+white_noise
    return signal_noise

def peak_dist(peak,freq):
    return abs(freq-peak)

    
   
def cost_function(res_psd,actual_freqs,dist_tol=0,peak_tol=0):
    #num_peaks= number of actual peaks in the frequency
    #tol = tolerance, if inaccuracy is less than tol, then return 0
    '''
    1. find all peaks
    2. calc cost function for num_peaks, find peaks closest to actual peak.
    3. 
    #rank by correct num peaks, distance, height/width ratio
    #try instead of adding distance, try normalized mean of distances
    '''
        
    correct_num_peaks=True
    peaks,h=find_peaks(res_psd['psd'],height=0)
    height_tol=peak_tol*mean(h['peak_heights'])
    prom,_,__=peak_prominences(res_psd['psd'],peaks)
    prom_thresh=mean(prom)*peak_tol
    peaks,props=find_peaks(res_psd['psd'],prominence=prom_thresh,height=height_tol)
    if len(peaks) < len(actual_freqs):
        correct_num_peaks=False       
    widths=np.asarray(peak_widths(res_psd['psd'],peaks,rel_height=0.99)[0])
    #only consider peaks clostest to actual freqs, need te do bipartite matching
    #TODO assignment problem between peaks and actual_freqs
    #create cost matrix, rows=peaks, cols= actual_freq, cost= dist
    temp_combs=np.asarray(list(itertools.product(res_psd['freq'][peaks],actual_freqs)))
    #create cost matrix given these combinations
    dist=lambda x,y:abs(x-y)
    cost=dist(temp_combs[:,0],temp_combs[:,1]).reshape(-1,len(actual_freqs)) #rows = peak, 
    row_ind,col_ind=linear_sum_assignment(cost)
    dists=np.mean(cost[row_ind,col_ind],dtype=float)
    #print(dists)#dists from closest assigned peaks
    
    peakidx=row_ind
    peak_heights=props['peak_heights'][peakidx]
    avg_height_width_ratio=mean([peak_height/widths[i] for i,peak_height in enumerate(peak_heights)])

    #print(dists)
    if dists<dist_tol:
        dists=0
    return (correct_num_peaks,dists,avg_height_width_ratio)

def del_points(signal,time,del_percent):
    n_del=int(del_percent*np.size(time))
    del_idx=np.random.choice(range(np.size(time)), n_del, replace=False)
    signal=np.delete(signal,del_idx)
    time=np.delete(time,del_idx)
    return signal,time
    
    
    