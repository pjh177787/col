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
from data_gen_presave import *
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, RNN, SimpleRNN, Reshape
from keras.layers import Conv1D, MaxPooling1D,Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import *
from keras.utils import multi_gpu_model
from keras.models import model_from_yaml

os.environ["CUDA_VISIBLE_DEVICES"] ="1"

# Give one class of sound
mode = input("Type mode (glass/baby/gun): ")

train_yaml_path = "audio_set/DCASE/devtrain/1000/meta"
train_audio_path = "audio_set/DCASE/devtrain/1000/audio"

test_yaml_path = "audio_set/DCASE/devtest/500/meta"


test_audio_path = "audio_set/DCASE/devtest/500/audio"
if(mode == "glass"):
	print(">> releasing training on glass_break")	
	train_yaml_file = "mixture_recipes_devtrain_glassbreak.yaml"
	test_yaml_file = "mixture_recipes_devtest_glassbreak.yaml"
	
elif(mode == "baby"):
	print(">> release training on baby cry")
	train_yaml_file = "mixture_recipes_devtrain_babycry.yaml"
	test_yaml_file = "mixture_recipes_devtest_babycry.yaml"
else:
	print(">> release training on gunshot")
	train_yaml_file = "mixture_recipes_devtrain_gunshot.yaml"
	test_yaml_file = "mixture_recipes_devtest_gunshot.yaml"
	
train_data = read_meta_yaml(train_yaml_path +"/" + train_yaml_file)
test_data  = read_meta_yaml( test_yaml_path +"/" +  test_yaml_file)

nperseg = 2048
noverlap= 1024

sample,labels,fs = prep_from_yaml(train_data[0],train_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)

model_name = input("model date_serialNumber (e.g. 20180422_01):")
serial_num = input("new serial number for this file (e.g. 02):")
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
print("num_bank:",num_bank)
print("timestepRec",timestepRec)
print("kernel_width",kernel_width)
print("filter_len",filter_len)
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
training_generator = DataGenerator(mode,"_train",train_id, sample.shape[0],kernel_width, **train_params)
test_generator = DataGenerator(mode,"_test",test_id,sample.shape[0],kernel_width, **test_params)

# Start to test
adam_opt = keras.optimizers.Adam(decay=0.01)
model.compile(loss='categorical_crossentropy',
                       optimizer=adam_opt)

model.summary()

print("\nKeep training on GPU...")
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
nn = int(input("Type the number of epochs you want to train(e.g. 5):"))
init_ep = int(input("Type the epoch you wish to start with(e.g. 20):"))
for ep in range(int(nn/2)):
	print("Training on epoch ",ep+init_ep)
	hist= model.fit_generator(generator=training_generator,
						#validation_data=test_generator,
						epochs = 2+ep+init_ep,
			initial_epoch=ep+init_ep,
			callbacks=[keras.callbacks.History()],
                    	use_multiprocessing=True,workers=8)

	# serialize model to YAML
	model_yaml = model.to_yaml()
	curr_date = datetime.datetime.now().strftime("%Y%m%d")
	model_name = "model_"+curr_date+"_"+serial_num+'_'+str(init_ep+nn)+"_"+mode
	with open('model_save/'+model_name+".yaml", "w") as yaml_file:
		 yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights('model_save/'+model_name+".h5")
	print("Saved model to disk, epoch ",ep*2+init_ep+2)



