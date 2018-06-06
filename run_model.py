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

os.environ["CUDA_VISIBLE_DEVICES"]=input("Use GPU 0/1:")
# Give one class of sound
mode = input("type a mode (glass/baby/gun): ")
dataset = input("which dataset s1/s2/s3/s4: ")

train_yaml_path = "audio_set/DCASE/devtrain/10000/meta"
train_audio_path = "audio_set/DCASE/devtrain/10000/audio/babycry/"+dataset

test_yaml_path = "audio_set/DCASE/devtest/500/meta"
test_audio_path = "audio_set/DCASE/devtest/500/audio"

if(mode == "glass"):
	print("training glass_break")	
	train_yaml_file = "mixture_recipes_devtrain_glassbreak.yaml"
	test_yaml_file = "mixture_recipes_devtest_glassbreak.yaml"
	
elif(mode == "baby"):
	print("training baby cry")
	train_yaml_file = "mixture_recipes_devtrain_babycry_"+dataset+"_new.yaml"
	test_yaml_file = "mixture_recipes_devtest_babycry.yaml"
else:
	print("training gunshot")
	train_yaml_file = "mixture_recipes_devtrain_gunshot.yaml"
	test_yaml_file = "mixture_recipes_devtest_gunshot.yaml"
	
train_data = read_meta_yaml(train_yaml_path +"/" + train_yaml_file)
test_data  = read_meta_yaml( test_yaml_path +"/" +  test_yaml_file)

nperseg = 2048
noverlap= 1024

sample,labels,fs = prep_from_yaml(train_data[0],train_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)
print("\n",train_data[0],"\n")


# In[113]:

# Feature params
default_mode = input("Run with default mode (80bank/100step/2x20kernel(y/n):")
if(default_mode == "y"):
	num_bank = 80
	timestepRec  = 100 # one frame is about 1/44100*1024 = 23 ms, 50 frames is about 1 sec
	kernel_width = 2
	filter_len   = 20
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
serial_num = input("Input this is nth model today (e.g. 2 for second model):")
# network params


# CNN params


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


sample_w_train = [0.04,0.96] 
sample_w_test  = [1.0,1.0]   
sample, labels, fs = prep_from_yaml(train_data[train_id[1]],train_audio_path,False,0,False,nperseg, noverlap)    
#DataGenerator(list_IDs, audio_len, dim, fs, yaml_path,yaml_file,audio_path,sample_spectrogram, nperseg, noverlap,\
#                 batch_size=32, n_channels=1,n_classes=2, shuffle=True):
training_generator = DataGenerator(mode,"_train",sample_w_train,train_id, sample.shape[0],kernel_width, **train_params)
#test_generator = DataGenerator(mode,"_test",sample_w_test,test_id,sample.shape[0],kernel_width, **test_params)


# In[115]:

# Network structure
'''
Main branch:mfcc_input->Conv1D,ReLu,BN,MaxPool->LSTM\
                    \                                Merge(concatenate)->Dense->activation->Single RNN unit->sigmoid
Aux branch           mfcc_delta ->BN -> Dense ->tanh/
                  zero_crossing
'''

### Main branch - this branch include CNN+LSTM
# batch size :unknown
# each truncated back propagation will contain 50 timesteps
# each timestep has one 64 dimensional mfcc vector as input to a CNN layer with 64 filters, each filter has 16 taps
# so each CNN filter produces 64-16+1=49 dimensional convolution result, maxpooling yields 1 feature
# eventually there are 64 features, they are reshaped into one 64 dimensional vector.
input_shape = (timestepRec+kernel_width-1,num_bank,1)                           
mfcc_input = Input(shape=input_shape, name='mfcc_input')        

cnn_layer = mfcc_input           
cnn_layer = Conv2D(filters=num_bank,
               kernel_size=(kernel_width,filter_len),
               padding='valid',
               activation='relu')(cnn_layer)

cnn_layer = BatchNormalization(axis=3)(cnn_layer)   # axis=3, which is channel dimension 
cnn_layer = MaxPooling2D(pool_size=(1,num_bank),
                     padding='same')(cnn_layer)  
cnn_layer = Dropout(dropout_rate)(cnn_layer)
cnn_layer = Reshape((timestepRec,num_bank))(cnn_layer)

lstm_layer = LSTM(num_bank, return_sequences=True)(cnn_layer)         
lstm_layer = Dropout(dropout_rate)(lstm_layer)


## Auxilliary branch - this branch only has feedforward
delta_input = Input(shape=(timestepRec,num_bank), dtype='float', name='delta_input')
zcr_input   = Input(shape=(timestepRec,1), dtype='float', name='zcr_input')
aux_input   = concatenate([delta_input, zcr_input])
aux_input   = Dense(num_bank+1)(aux_input) 
#print(aux_input)

## Merge branches - combine main branch and aux branch
merged_layer   = concatenate([lstm_layer, aux_input])
#merged_layer = lstm_layer
dense_layer = Dense(num_bank+num_bank+1, activation = 'relu')(merged_layer)
dense_layer = BatchNormalization(axis = 2)(dense_layer)        
#dense_layer = Dropout(dropout_rate/2)(dense_layer)

model_output = SimpleRNN(2, return_sequences=True)(dense_layer) 
#model_output = TimeDistributed(Dense(1))(dense_layer)
model_output = Activation('softmax')(model_output)

model = Model(inputs=[mfcc_input,delta_input,zcr_input], outputs=[model_output])

#model = Model(inputs=[mfcc_input], outputs=[model_output])


adam_opt = keras.optimizers.Adam(decay=0.01)
model.compile(loss='categorical_crossentropy',
                      optimizer=adam_opt)

model.summary()

print("\ntraining on GPU...")
# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
init_ep = 0


for ep in range(0,45):
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
	model_name = "model_"+curr_date+"_"+serial_num+'_'+mode
	with open('model_save/'+model_name+".yaml", "w") as yaml_file:
		 yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights('model_save/'+model_name+".h5")
	print("Saved model to disk, epoch ",init_ep+2)


print("All training is done!")
exit()

num_batch = int(test_generator.__len__())

import multiprocessing
from joblib import Parallel,delayed

def test_par(ii):

	print("Preparing %s/%s batch" %(ii+1, num_batch))
	X,y = test_generator.__getitem__(ii)
	return X,y


num_cores = multiprocessing.cpu_count()


X = []
y = []

for Xs,ys in(Parallel(n_jobs=num_cores)(delayed(test_par)(ii) for ii in range(num_batch))):
	X.append(Xs)
	y.append(ys)

for ii in range(num_batch):
    print("running on %s/%s batch" %(ii+1, num_batch))

    y_pred = model.predict(X[ii][0], batch_size=None, verbose=0, steps=None)
    # plot X and y
    for i in range(1):
        y_comb = y[ii][0,:,1]
        y_pred_comb = 1-y_pred[0,:,0]
        for j in range(32): 
            y_comb = np.hstack((y_comb,y[ii][j,:,1]))
            y_pred_comb =  np.hstack((y_pred_comb,1-y_pred[j,:,0]))
            
        output_labels, predict_event_list = event_list_gen(y_pred_comb,0.5,5)
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
        
p_len = int(len(predict_event_list_total)/2) 
predict_event_list_total = predict_event_list_total.reshape((p_len,2))

g_len = int(len(ground_truth_event_list_total)/2) 

ground_truth_event_list_total = ground_truth_event_list_total.reshape((g_len,2))




if(mode == "glass"):
	newline_ = ' glassbreak\n'
elif(mode == "baby"):
	newline_ = ' babycry\n'
else:
	newline_ = ' gunshot\n'
	
np.savetxt("event_list/"+model_name+"_test.txt",            predict_event_list_total, fmt='%.4f',            newline=newline_,)


np.savetxt("event_list/"+model_name+"_test_truth.txt",            ground_truth_event_list_total, fmt='%.4f',            newline=newline_,)

print("Test done!")


