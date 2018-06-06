import tensorflow as tf
import keras
import copy
import datetime
import yaml,os
import multiprocessing
import sys
import shutil
import librosa
import soundfile as sf, numpy as np
import argparse, textwrap
import wave, numpy as np
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed
from scipy import signal
from scipy.io import wavfile
from sound_detection_utils_tf import *
from data_gen_first_model_zcr import *
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, RNN, SimpleRNN, Reshape
from keras.layers import Conv1D, MaxPooling1D,Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import *
from keras.utils import multi_gpu_model


os.environ["CUDA_VISIBLE_DEVICES"]=input("select GPU 0/1:")
print("\nThis model duplicates the DCASE 2017 winner's model\n")
# Give one class of sound
mode = input("type a mode (glass/baby/gun): ")

train_yaml_path = "audio_set/DCASE/devtrain/10000/meta"
train_audio_path = "audio_set/DCASE/devtrain/10000/audio"

dataset = input("choose dataset s1/s2/s3/s4:")

test_audio_path = "audio_set/DCASE/devtest/500/audio"
if(mode == "glass"):
	print(">>training glass_break")	
	train_yaml_file = "mixture_recipes_devtrain_glassbreak.yaml"
	
elif(mode == "baby"):
	print(">>training baby cry")
	train_yaml_file = "mixture_recipes_devtrain_babycry_"+dataset+"_new.yaml"
	train_audio_path += "/babycry/"+dataset
else:
	print(">>training gunshot")
	train_yaml_file = "mixture_recipes_devtrain_gunshot.yaml"
	
train_data = read_meta_yaml(train_yaml_path +"/" + train_yaml_file)

nperseg = 2048
noverlap= 1024

sample,labels,fs = prep_from_yaml(train_data[0],train_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)
print("\n",train_data[0],"\n")


# In[113]:

# Feature params
default_mode = input("Run with default mode (128bank/100step/1x32kernel(y/n):")
if(default_mode == "y"):
	num_bank = 128 
	timestepRec  = 100 # one frame is about 1/44100*1024 = 23 ms, 50 frames is about 1 sec
	kernel_width = 1
	filter_len   = 32
else:
	num_bank = int(input("numbank:"))
	timestepRec = int(input("timestepRec:"))
	kernel_width= int(input("kernel width:"))
	filter_len  = int(input("filter len(kernel height):"))
diff = False
diff_diff = False
zero_cross = True
zero_freq = True
diff_by_time = True


# model bookkeeping
serial_num = input("Input this is nth model today (e.g. 02 for second model):")
# network params


# CNN params


max_pool_len = num_bank - filter_len + 1
dropout_rate = 0.3

# Data generator
dim = (timestepRec,num_bank)
train_params = {    'dim':dim, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}


train_id = {}
for i in range(len(train_data)):
    train_id[i] = i
    
sample, labels, fs = prep_from_yaml(train_data[train_id[1]],train_audio_path,False,0,False,nperseg, noverlap)    
sample_w = [0.04,0.96]
#DataGenerator(list_IDs, audio_len, dim, fs, yaml_path,yaml_file,audio_path,sample_spectrogram, nperseg, noverlap,\
#                 batch_size=32, n_channels=1,n_classes=2, shuffle=True):
training_generator = DataGenerator(mode,"_train",sample_w,train_id, sample.shape[0],kernel_width, **train_params)


batch_size = training_generator.batch_size

# Network structure
input_shape = (timestepRec+kernel_width-1,num_bank+1,1)                           
mfcc_input = Input(shape=input_shape, name='mfcc_input')        


cnn_layer = mfcc_input           
cnn_layer = Conv2D(filters=num_bank,
               kernel_size=(kernel_width,filter_len),
               padding='valid',
               activation='relu')(cnn_layer)

cnn_layer = BatchNormalization(axis=3)(cnn_layer)   # axis=3, which is channel dimension 
cnn_layer = MaxPooling2D(pool_size=(1,int(num_bank/2)),
                     padding='same')(cnn_layer)  
cnn_layer = Dropout(dropout_rate)(cnn_layer)
cnn_layer = Reshape((timestepRec,num_bank*2))(cnn_layer)


lstm_layer = LSTM(num_bank*2, return_sequences=True,go_backwards=True)(cnn_layer)         
lstm_layer = LSTM(num_bank, return_sequences=True,go_backwards=False)(lstm_layer)
lstm_layer = Dropout(dropout_rate)(lstm_layer)


dense_layer = Dense(num_bank, activation = 'relu')(lstm_layer)
dense_layer = BatchNormalization(axis = 2)(dense_layer)        

#model_output = TimeDistributed(Dense(2))(dense_layer) 

model_output = SimpleRNN(2,return_sequences=True)(dense_layer) 
model_output = Activation('softmax')(model_output)

#model_output = Rieshape((batch_size*timestepRec,2))(model_output)

model = Model(inputs=[mfcc_input], outputs=[model_output])

adam_opt = keras.optimizers.Adam(decay=0.01)
model.compile(loss='categorical_crossentropy',
                      optimizer=adam_opt)

model.summary()
#input("continue")
print("\ntraining on GPU...")
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
init_ep = 0


for ep in range(0,40):
	init_ep = ep*2;
	print("Training on epoch ",init_ep)
	hist= model.fit_generator(generator=training_generator,
						#validation_data=test_generator,
						epochs = 2+init_ep,
			initial_epoch=init_ep,
			callbacks=[keras.callbacks.History()],
                    	use_multiprocessing=True,workers=7)




	# serialize model to YAML
	model_yaml = model.to_yaml()
	curr_date = datetime.datetime.now().strftime("%Y%m%d")
	model_name = "model_dup_"+curr_date+"_"+serial_num+'_'+mode+"_first"
	with open('model_save/'+model_name+".yaml", "w") as yaml_file:
		 yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights('model_save/'+model_name+".h5")
	print("Saved model to disk, epoch ",init_ep+2)


print("All training is done!")


