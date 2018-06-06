import yaml,os
import sys
import shutil
import librosa
import soundfile as sf, numpy as np
import pandas as pd
import argparse, textwrap
import wave, numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
from scipy import signal
from scipy.io import wavfile

from IPython import embed
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from tqdm import tqdm
import hashlib
import warnings


'''
Feature Prep Related
'''
''' DCASE prep '''
def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def prep_from_yaml(data,audio_path,listen,noise,trunc,nperseg, noverlap):
    
    x_wave,fs = sf.read(audio_path+"/"+data['mixture_audio_filename'])

    x = x_wave.astype(float)
    if(noise>0):
        x += np.random.normal(0, x.max()*noise,x.shape)       
    x_play = x.astype(np.int32)    
    # spectrogram
    if(x.shape==(x.shape[0],)):
        f, t, Sxx = signal.spectrogram(x, fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)
    else:
        f, t, Sxx = signal.spectrogram(x[:,0], fs,window=("hann"),nperseg=nperseg, noverlap=noverlap)
    #Sxx /= Sxx.max()
    labels = np.zeros(Sxx.shape[1])
    #print("Sxx.shape=",Sxx.shape)
    
    #print("\nRaw spectrogram")
    #plt.pcolormesh(t, f, np.log(Sxx+1e-8))
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
  
    if(listen==1):
        sd.play(x_play,fs,blocking=True)
        
    if(data['event_present'] == True):
        
        bb = int(data['event_start_in_mixture_seconds']*fs)
        ee = min(int(bb+fs*data['event_length_seconds']),(x.shape[0]))

        #Sxx  = 1-Sxx
        t_bb= int(t.shape[0]*bb/x_play.shape[0])+1
        t_ee= int(t.shape[0]*ee/x_play.shape[0])
        #print("t_begin=",t_bb,"t_end=",t_ee)

        # laybel_gen - 0 is everything else, 1 is scream class            
        for i in range(t_bb,t_ee):
            labels[i] = 1

    for i in range(0,len(labels)):
        labels[i] = int(labels[i])
        
    if(data['event_present'] == True and trunc == True):  
        if(t_bb > 1500):
            Sxx = Sxx.T
            Sxx = Sxx[t_bb-1500:]
            Sxx = Sxx.T
            labels = labels[t_bb-1500:]
    
    return Sxx.T,labels,fs

'''
MFCC Related
'''

def get_mfcc_filter_bank(Sxx_in,num_bank,fs):
    m = np.linspace(start=1125 * np.log(1 + 10 / 700.0),stop=1125 * np.log(1 + fs/2*0.8 / 700.0),num=num_bank)
    # get list of freq centers
    #print(m)
    f = np.zeros(len(m)+1)
    f[0:len(m)] = 700 * (np.exp(m / 1125) - 1)
    
    # get list of indices
    f = np.floor((Sxx_in.shape[1]+1)*f/(fs/2))
    f[len(m)] = int(f[num_bank-1]*f[num_bank-1]/f[num_bank-2])
    print("filter bins:")
    print(f)
    # get coef for each filter bank for all time steps
    filter_bank = np.zeros((num_bank,Sxx_in.shape[1]))
    for i in range(0,num_bank):             
        for k in range(Sxx_in.shape[1]):        
            if(f[i] == f[i-1] or f[i] == f[i+1]): # at DC there might be large overlap 
                if(k == f[i]):
                    filter_bank[i][k] = 1;
                else:
                    filter_bank[i][k] = 0;
            elif(f[i-1] <= k and k <= f[i]):
                filter_bank[i][k] = (k-f[i-1]) / (f[i] - f[i-1])/(f[i+1] - f[i])
            elif(f[i] <= k and k <= f[i+1]):
                filter_bank[i][k] = (f[i+1]-k) / (f[i+1] - f[i])/(f[i+1] - f[i])
                
    return filter_bank;

def Sxxin_to_mfcc(Sxx_in,num_bank,fs,filter_bank):
    from scipy import fftpack
    mfcc_mat = np.zeros((Sxx_in.shape[0],num_bank)) # dim = (t,num_bank), same as Sxx_in

    for t in range(0,Sxx_in.shape[0]):
        mfcc_mat[t] = np.log(1e-15+np.dot(filter_bank,Sxx_in[t]))
    
    #print(mfcc_mat.shape)
    return mfcc_mat;

def mfcc_difference(mfcc_mat):
    diff_mat = np.zeros_like(mfcc_mat);
    
    for t in range(1,mfcc_mat.shape[0]):
        diff_mat[t] = mfcc_mat[t] - mfcc_mat[t-1]
    return diff_mat

def zero_crossing_rate(data,audio_path,trunc,nperseg, noverlap,DCASE=True,cross=False):
    if(DCASE== True):
        x_wave,fs = sf.read(audio_path+"/"+data['mixture_audio_filename'])
    else:
        x_wave,fs = sf.read(audio_path)
    x = x_wave.astype(float)
    if(x.shape!=(x.shape[0],)):
        x = x[:,0]    
    zcr = np.zeros(int(x.shape[0]/(nperseg-noverlap)))
    for seg_idx in range(int(x.shape[0]/(nperseg-noverlap))):
        x_tmp = x[seg_idx*(nperseg-noverlap):seg_idx*(nperseg-noverlap)+nperseg]
        for i in range(1,min(nperseg,x.shape[0]-seg_idx*(nperseg-noverlap))):
            if(x_tmp[i]*x_tmp[i-1] < 0):
                zcr[seg_idx] += 1
    
    if(DCASE== False and cross == False):
        interval = np.genfromtxt(data, delimiter='\t')     
        if(interval.shape == (2,)):     
            bb,ee = int(interval[0]*fs),min(int(interval[1]*fs),(x.shape[0]))
            t_bb= int(zcr.shape[0]*bb/x.shape[0])+1
            t_ee= int(zcr.shape[0]*ee/x.shape[0]) 
            zcr = np.hstack((zcr[max(int(t_bb/2),int(t_bb-300)):t_ee],zcr[max(int(t_bb/3),int(t_bb-100)):t_ee]))

        else:
            zcr_prev = zcr[0:2]
            prev_ee = 0;
            for j in range(0,interval.shape[0]):
                bb,ee = int(interval[j][0]*fs),int(interval[j][1]*fs)  
                t_bb= int(zcr.shape[0]*bb/x.shape[0])+1
                t_ee= int(zcr.shape[0]*ee/x.shape[0])
                zcr_prev = np.hstack((zcr_prev, zcr[max(prev_ee,int(t_bb-300)):min(zcr.shape[1],t_ee)]))
                prev_ee = t_ee     
                
    return zcr/nperseg;

def zero_idft_energy(Sxx_spec):
    Sxx_in = Sxx_spec.T
    Sxx_low = Sxx_in[int(Sxx_in.shape[0]/8):int(Sxx_in.shape[0]/4)]
    Sxx_hi = Sxx_in[int(Sxx_in.shape[0]*3/4):int(Sxx_in.shape[0]*7/8)]
    print(Sxx_in.shape,Sxx_low.shape,Sxx_hi.shape)
    Sxx_in = np.vstack((Sxx_low,Sxx_hi))
    Sxx_in = Sxx_in.T
    
    Sxx_spec_idft = np.zeros_like(Sxx_in)
    zero_freq = np.zeros(Sxx_spec_idft.shape[0])
    for i in range(Sxx_in.shape[0]):   
        #tmp = Sxx_in[i][:]
        #tmp /= tmp.max()
        #lo = np.sum(tmp[1:20])
        #mid= np.sum(tmp[int(Sxx_in.shape[1]/2)-10:int(Sxx_in.shape[1]/2)+10])
        #zero_freq[i] = np.sum(Sxx_in[i][:])/np.abs(lo-mid)
        zero_freq[i] = np.sum(Sxx_in[i][:])
        
    zero_freq = zero_freq/zero_freq.max()
    
    
    return zero_freq

def mean_var_normalization(Sxx_in, byRow = True):
    
    if(byRow == False):
        Sxx_in = Sxx_in.T
        
    for i in range(Sxx_in.shape[0]):
        Sxx_in[i] -= np.mean(Sxx_in[i]) 
        Sxx_in[i] /= np.std(Sxx_in[i]) 
        
    if(byRow == False):
        Sxx_in = Sxx_in.T
        
    return Sxx_in

def event_list_gen(labels,threshold,continuous_frame_1,continuous_frame_0=13):

    count = 0
    is_same = 0 # same as previous label
    cont_count = 0
    onset_detected = False
    final_out_labels = np.zeros(labels.shape[0])
    event_list = []
    
    for i in range(0,labels.shape[0]):  
        if(i > 0):
            final_out_labels[i] = final_out_labels[i-1];

        count += 1

        if((labels[i] >= threshold and labels[i-1]>=threshold) or (labels[i] < threshold and labels[i-1] < threshold)):
            is_same = 1;
        else:
            is_same = 0;
        if(is_same == 1):
            cont_count += 1
        else:
            cont_count = 0; # if change is detected, reset cont_count
        if(cont_count == continuous_frame_1 and labels[i] >= threshold):
            if(onset_detected == False):
                onset_detected = True;
                event_list.append(i-continuous_frame_1-1)
            #print("onset:",i-continuous_frame-1)
            for j in range(i-continuous_frame_1-1,i+1):
                final_out_labels[j] = 1; # label those past labels as 1
        elif(cont_count == continuous_frame_0 and labels[i] < threshold):
            if(onset_detected == True):
                event_list.append(i-continuous_frame_0-1)
                onset_detected = False # offset
                #print("offset:",i-continuous_frame-1)
            for j in range(i-continuous_frame_0-1,i+1):
                final_out_labels[j] = 0;
    if(onset_detected == True):
        event_list.append(labels.shape[0])
    return final_out_labels, np.array(event_list)
