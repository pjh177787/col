import tensorflow as tf
import keras
import copy
import datetime
import yaml,os
import sys
import shutil
import librosa
import soundfile as sf, numpy as np
import argparse, textwrap
import wave, numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal
from scipy.io import wavfile
from sound_detection_utils_tf import *
from data_gen import *
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, RNN, SimpleRNN, Reshape
from keras.layers import Conv1D, MaxPooling1D,Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import *
from keras.utils import multi_gpu_model
from keras.models import model_from_yaml


# Give one class of sound
mode = input("Type test mode (glass/baby/gun): ")
dataset = input("which dataset (s1/s2/s3/s4):")
train_yaml_path = "audio_set/DCASE/devtrain/10000/meta"
train_audio_path = "audio_set/DCASE/devtrain/10000/audio/"+dataset

test_yaml_path = "audio_set/DCASE/devtest/500/meta"


test_audio_path = "audio_set/DCASE/devtest/500/audio"
if(mode == "glass"):
	print(">> test on glass_break")	
	train_yaml_file = "mixture_recipes_devtrain_glassbreak.yaml"
	test_yaml_file = "mixture_recipes_devtest_glassbreak.yaml"
	
elif(mode == "baby"):
	print(">> test on baby cry")
	train_yaml_file = "mixture_recipes_devtrain_babycry_"+dataset+".yaml"
	test_yaml_file = "mixture_recipes_devtest_babycry.yaml"
else:
	print(">> test on gunshot")
	train_yaml_file = "mixture_recipes_devtrain_gunshot.yaml"
	test_yaml_file = "mixture_recipes_devtest_gunshot.yaml"
	
train_data = read_meta_yaml(train_yaml_path +"/" + train_yaml_file)
test_data  = read_meta_yaml( test_yaml_path +"/" +  test_yaml_file)

nperseg = 2048
noverlap= 1024

sample,labels,fs = prep_from_yaml(train_data[0],train_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)

model_name = input("model serial number (e.g. 20180422_01):")
model_name = 'model_'+model_name+"_"+mode
print("loading model ",model_name)
with open("./model_save/"+model_name+".yaml", 'r') as yaml_string:
    model = model_from_yaml(yaml_string)
model.load_weights("./model_save/"+model_name+".h5",by_name=True)
# In[113]:
# Retriving model info
model_shape0 = model.layers[0].get_output_at(0).get_shape().as_list()
num_bank = model_shape0[2] 
model_shape1 = model.layers[1].get_output_at(0).get_shape().as_list()
timestepRec = model_shape1[1]
kernel_width= model_shape0[1]-model_shape1[1]+1
filter_len  = model_shape0[2]-model_shape1[2]+1

diff = False
diff_diff = False
zero_cross = True
zero_freq = True
diff_by_time = True

# network params
max_pool_len = num_bank - filter_len + 1
dropout_rate = 0.3

# Data generator
dim = (timestepRec,num_bank)
train_params = {    'dim':dim, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}

test_params = {    'dim':dim, 'fs': fs, 'yaml_path':test_yaml_path,    'yaml_file':test_yaml_file,'audio_path':test_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}

train_id = {}
test_id = {}
for i in range(len(train_data)):
    train_id[i] = i
for i in range(len(test_data)):    
    test_id[i] = i
    
sample, labels, fs = prep_from_yaml(train_data[train_id[1]],train_audio_path,False,0,False,nperseg, noverlap)    #DataGenerator(list_IDs, audio_len, dim, fs, yaml_path,yaml_file,audio_path,sample_spectrogram, nperseg, noverlap,\
#                 batch_size=32, n_channels=1,n_classes=2, shuffle=True):
training_generator = DataGenerator(train_id, sample.shape[0],kernel_width, **train_params)
test_generator = DataGenerator(test_id,sample.shape[0],kernel_width, **test_params)

# Start to test
num_batch = int(test_generator.__len__())

#from Ipython.display import clear_output
for ii in range(num_batch):

    print("Testing on %s/%s batch" %(ii+1, num_batch))
    X,y = test_generator.__getitem__(ii)
    y_pred = model.predict(X, batch_size=None, verbose=0, steps=None)
    # plot X and y
    for i in range(int(y.shape[0]/32)):
        y_comb = y[i*32,:,1]
        y_pred_comb = 1-y_pred[i*32,:,0]
        for j in range(32): 
            y_comb = np.hstack((y_comb,y[i*32+j,:,1]))
            y_pred_comb =  np.hstack((y_pred_comb,1-y_pred[i*32+j,:,0]))
            
        output_labels, predict_event_list = event_list_gen(y_pred_comb,0.5,1)
        dummy        , ground_truth_event_list = event_list_gen(y_comb,0.5,1)
        #plt.plot(y_comb+2)
        #plt.plot(y_pred_comb)
        #plt.plot(output_labels)
        #plt.title("Sample %s to %s, Batch %s/%s" %(i*32, i*32+31,ii,num_batch))
        #plt.show()
        #
        
        predict_event_list = (predict_event_list+ii*timestepRec*32)*(nperseg-noverlap)/44100        
        ground_truth_event_list = (ground_truth_event_list+ii*timestepRec*32)*(nperseg-noverlap)/44100
        
        if(ii+i == 0):
            predict_event_list_total = predict_event_list
            ground_truth_event_list_total = ground_truth_event_list
        else: 
            predict_event_list_total = np.hstack((predict_event_list_total,predict_event_list))
            ground_truth_event_list_total = np.hstack((ground_truth_event_list_total,ground_truth_event_list))
        
predict_event_list_total = predict_event_list_total.reshape((int(len(predict_event_list_total)/2),2))

if(mode == "glass"):
	newline_ = ' glassbreak\n'
elif(mode == "baby"):
	newline_ = ' babycry\n'
else:
	newline_ = ' gunshot\n'
	
np.savetxt("event_list/"+model_name+"_test.txt",            predict_event_list_total, fmt='%.4f',            newline=newline_,)

ground_truth_event_list_total = ground_truth_event_list_total.reshape((int(len(ground_truth_event_list_total)/2),2))
np.savetxt("event_list/"+model_name+"_test_truth.txt",            ground_truth_event_list_total, fmt='%.4f',            newline=newline_,)

print("Test done!")


