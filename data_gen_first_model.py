import tensorflow as tf
import keras
import copy
import yaml,os
import sys
import shutil
import librosa
import soundfile as sf, numpy as np
#import pandas as pd
import argparse, textwrap
import wave, numpy as np
import multiprocessing
import matplotlib.pyplot as plt
#import sounddevice as sd
import h5py
from scipy import signal
from scipy.io import wavfile
from sound_detection_utils_tf import *

from joblib import Parallel, delayed

from keras.layers import Input, Embedding, LSTM, Dense, concatenate, RNN, SimpleRNN, Reshape
from keras.layers import Conv1D, MaxPooling1D,Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import *


# In[106]:

class  DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, mode,signature,sample_weight, list_IDs, audio_len,kernel_width, dim, fs, yaml_path,yaml_file,audio_path,sample_spectrogram, nperseg, noverlap, batch_size=32, n_channels=1,n_classes=2, shuffle=True):
        'Initialization'
        self.audio_len  = audio_len
        self.dim        = dim
        self.fs         = fs
        self.num_bank   = self.dim[1]
        self.audio_path = audio_path
        self.nperseg    = nperseg
        self.noverlap   = noverlap
        self.kernel_width = kernel_width
		
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle       
        self.signature = signature
        self.sample_weight = sample_weight
        
        self.data_list = read_meta_yaml(yaml_path +"/" + yaml_file)
        print("\n-----Initializing filter bank-----")
        self.filter_bank =  get_mfcc_filter_bank(sample_spectrogram,self.num_bank,self.fs)
        print("self.batch_size",self.batch_size)
        print("self.audio_len",self.audio_len)
        print("self.dim[0]",self.dim[0])
        self.mode = mode
        self.serial=str(self.nperseg)+"_"+str(self.num_bank)
        import os
        if(os.path.exists("./feature_save/"+self.mode+"/"+self.serial+"/"+self.data_list[100]['mixture_audio_filename']+"_mfcc.npy")):
            print(">>files exist")
            print("length of file = ",len(self.data_list))
        else:
            print(">>preparing file...:")
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores-1)(delayed(self.genfiles)(i) for i in range(len(self.data_list)))

 
        self.on_epoch_end()

    def genfiles(self,i):

        print("preparing %s/%s" % (i, len(self.data_list)))    
        spectrogram_temp,labels,dummy = prep_from_yaml(self.data_list[i],self.audio_path,False,0,                                                         False,self.nperseg, self.noverlap)
		
        mfcc_temp = Sxxin_to_mfcc(spectrogram_temp,self.num_bank, self.fs,self.filter_bank)
        zcr_temp = zero_crossing_rate(self.data_list[i],self.audio_path,False,                                            self.nperseg, self.noverlap,                                            DCASE=True,cross=False)
		
        np.save("./feature_save/"+self.mode+"/"+self.serial+"/"+self.data_list[i]['mixture_audio_filename']+"_mfcc.npy",mfcc_temp)
        np.save("./labels_save/"+self.mode+"/"+self.serial+"/"+self.data_list[i]['mixture_audio_filename']+"_labels.npy",labels)
        np.save("./feature_save/"+self.mode+"/"+self.serial+"/"+self.data_list[i]['mixture_audio_filename']+"_zcr.npy",zcr_temp)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)*(self.audio_len/self.dim[0])) / self.batch_size)
        # because each audio file contains N samples, N = audio_len/timestepRec
        # list_ID is only for file ID, not sample ID.
        # if timestepRec == audio_len, then each file correspond to 1 sample

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        num_file_per_batch = int(np.ceil(self.batch_size/(self.audio_len/self.dim[0]))) # see annotation above

        indexes = self.indexes[index*num_file_per_batch:(index+1)*num_file_per_batch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #sample_w = np.zeros((self.batch_size,2))
        #sample_w[:,0] += 0.05
        #sample_w[:,1] += 0.95
        return X, y#sample_w

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # (batch_size, timestepRec, numbank, n_channels=1)
        mfcc_input_0    = np.zeros((self.batch_size, self.dim[0]+self.kernel_width-1,self.dim[1], self.n_channels))
        mfcc_input_1    = np.zeros_like(mfcc_input_0)
        mfcc_input_2    = np.zeros_like(mfcc_input_0)
	#delta_input   = np.zeros((self.batch_size, *self.dim))
        #zcr_input     = np.zeros((self.batch_size, self.dim[0],1))        
        y = np.zeros((self.batch_size,self.dim[0]), dtype=int)
    
        timestep = self.dim[0]
        num_sample_per_batch = int(np.floor(self.audio_len/timestep))
        
        # Generate data
        for i in range(len(list_IDs_temp)): # len(list_IDs_temp) = ceil ( batch_size / num_sample_per_batch )
            
            #print("file no.",list_IDs_temp[i])
            # Generate data for N samples from 1 audio file (each file is very long, so divide into N samples)     
            mfcc_temp = np.load("./feature_save/"+self.mode+"/"+self.serial+"/"+self.data_list[list_IDs_temp[i]]['mixture_audio_filename']+"_mfcc.npy")
            labels = np.load("./labels_save/"+self.mode+"/"+self.serial+"/"+self.data_list[list_IDs_temp[i]]['mixture_audio_filename']+"_labels.npy")

            mfcc     = mfcc_temp[:,:,np.newaxis]
            #zcr_temp = np.load("./feature_save/"+self.mode+"/"+self.data_list[list_IDs_temp[i]]['mixture_audio_filename']+self.serial+"_zcr.npy")

            #zcr  = zcr_temp[:,np.newaxis] # turn it from 1D to 2D (timestep,1)
            
            for j in range(0,num_sample_per_batch):
                if( i*num_sample_per_batch+j >= self.batch_size):
                    break;
                 # generate inputs
                mfcc_input_0[ i*num_sample_per_batch+j,] = mfcc[ j*timestep: (j+1)*timestep+self.kernel_width-1,]

                mfcc_input_1[ i*num_sample_per_batch+j,] = mfcc[ j*timestep+1: (j+1)*timestep+self.kernel_width,]

                mfcc_input_2[ i*num_sample_per_batch+j,] = mfcc[ j*timestep+2: (j+1)*timestep+self.kernel_width+1,]


                #delta_input[i*num_sample_per_batch+j,] = delta[j*timestep: (j+1)*timestep,]
                #zcr_input[  i*num_sample_per_batch+j,] = zcr[  j*timestep: (j+1)*timestep,]
                
                # Store label
                y[  i*num_sample_per_batch+j,] = labels[  j*timestep: (j+1)*timestep]
        if(self.signature == "_train"):
            X_list = [mfcc_input_0]
            y_ret = keras.utils.to_categorical(y, num_classes=2) # weight balance
            y_ret[:,:,0] *= self.sample_weight[0]
            y_ret[:,:,1] *= self.sample_weight[1]
            y_ret = np.flip(y_ret,1)
        else:
            X_list = [mfcc_input_0,mfcc_input_1,mfcc_input_2]

            y_ret = keras.utils.to_categorical(y, num_classes=2)
        #y = np.reshape(y,(self.batch_size*timestep))
        return X_list, y_ret		
		
		
