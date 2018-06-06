import tensorflow as tf
import keras
import copy
import datetime
import multiprocessing
import yaml,os
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
from data_gen_first_model import *
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, RNN, SimpleRNN, Reshape
from keras.layers import Conv1D, MaxPooling1D,Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dropout, TimeDistributed
from keras.models import Model
from keras.utils.vis_utils import *
from keras.utils import multi_gpu_model
from keras.models import model_from_yaml



os.environ["CUDA_VISIBLE_DEVICES"]=input("chose gpu 0/1: ")

test_yaml_path = "audio_set/DCASE/devtest/500/meta"
test_audio_path = "audio_set/DCASE/devtest/500/audio"

mode = "baby"#input("Type test mode (glass/baby/gun): ")
nperseg = 2048
noverlap= 1024

train_test = "0"#input("\nTest(0) or Training(1) dataset:")
if(train_test == "1"):

	dataset = input("which dataset (s1/s2/s3/s4):")
	train_yaml_path = "audio_set/DCASE/devtrain/10000/meta"
	train_audio_path = "audio_set/DCASE/devtrain/10000/audio/babycry/"+dataset
	if(mode == "glass"):
		print(">> test on glass_break")	
		train_yaml_file = "mixture_recipes_devtrain_glassbreak.yaml"
	
	elif(mode == "baby"):
		print(">> test on baby cry")
		train_yaml_file = "mixture_recipes_devtrain_babycry_"+dataset+".yaml"
	else:
		print(">> test on gunshot")
		train_yaml_file = "mixture_recipes_devtrain_gunshot.yaml"

	train_data  = read_meta_yaml( train_yaml_path +"/" +  train_yaml_file)
	sample,labels,fs = prep_from_yaml(train_data[0],train_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)
else:
	if(mode == "glass"):
		print(">> test on glass_break")	
		test_yaml_file = "mixture_recipes_devtest_glassbreak.yaml"
	
	elif(mode == "baby"):
		print(">> test on baby cry")
		test_yaml_file = "mixture_recipes_devtest_babycry.yaml"
	else:
		print(">> test on gunshot")
		test_yaml_file = "mixture_recipes_devtest_gunshot.yaml"
	
	test_data  = read_meta_yaml( test_yaml_path +"/" +  test_yaml_file)
	
	sample,labels,fs = prep_from_yaml(test_data[0],test_audio_path,listen=0,noise=0,trunc = False,nperseg=nperseg, noverlap=noverlap)

model_number = "20180507_bls1_ts3_10"#input("\nmodel serial number 1(e.g. 20180422_01):")
model_name = 'model_dup_'+model_number+"_"+mode+"_first"
print("\nloading model ",model_name)
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


model_number2 = "20180507_bls1_ts4_15"#input("\nmodel serial number 2(e.g. 20180422_02):")
model_name2 = 'model_dup_'+model_number2+"_"+mode+"_first"
print("\nloading model ",model_name2)
with open("./model_save/"+model_name2+".yaml", 'r') as yaml_string:
    model2 = model_from_yaml(yaml_string)
model2.load_weights("./model_save/"+model_name2+".h5",by_name=True)
# In[113]:
# Retriving model info
model2_shape0 = model2.layers[0].get_output_at(0).get_shape().as_list()
num_bank2 = model2_shape0[2] 
model2_shape1 = model2.layers[1].get_output_at(0).get_shape().as_list()
timestepRec2 = model2_shape1[1]
kernel_width2= model2_shape0[1]-model2_shape1[1]+1
filter_len2  = model2_shape0[2]-model2_shape1[2]+1
m3 = input("is there a third model? (y/n): ")
if(m3 == "n"):
	df_w = input("Use default ensemble weight (1,1) for models?(y/n): ")
	if(df_w == "y"):
		w = [1,1]
	else:
		w1 = int(input("weight for model 1:"))
		w2 = int(input("weight for model 2:"))
		w = [w1,w2]

if(m3 == "y"):
	model_number3 = "20180508_bls2_ts1_15"#input("\nmodel serial number 3(e.g. 20180422_03):")
	model_name3 = 'model_dup_'+model_number3+"_"+mode+"_first"
	print("\nloading model ",model_name3)
	with open("./model_save/"+model_name3+".yaml", 'r') as yaml_string:
	    	model3 = model_from_yaml(yaml_string)
	model3.load_weights("./model_save/"+model_name3+".h5",by_name=True)
	model3_shape0 = model3.layers[0].get_output_at(0).get_shape().as_list()
	num_bank3 = model3_shape0[2] 
	model3_shape1 = model3.layers[1].get_output_at(0).get_shape().as_list()
	timestepRec3 = model3_shape1[1]
	kernel_width3= model3_shape0[1]-model3_shape1[1]+1
	filter_len3  = model3_shape0[2]-model3_shape1[2]+1

m4 = input("is there a fourth model? (y/n): ")
if(m4 == "n" and m3 == "y"):
	df_w = input("Use default ensemble weight (1,1,1) for models?(y/n): ")
	if(df_w == "y"):
		w = [1,1,1]
	else:
		w1 = int(input("weight for model 1:"))
		w2 = int(input("weight for model 2:"))
		w3 = int(input("weight for model 3:"))
		w = [w1,w2,w3]
elif(m3 == "y" and m4 == "y"):
	model_number4 = "20180510_bls1_ts3_30"#input("\nmodel serial number 4(e.g. 20180422_04):")
	model_name4 = 'model_dup_'+model_number4+"_"+mode+"_first"
	print("\nloading model ",model_name4)
	with open("./model_save/"+model_name4+".yaml", 'r') as yaml_string:
	    	model4 = model_from_yaml(yaml_string)
	model4.load_weights("./model_save/"+model_name4+".h5",by_name=True)
	model4_shape0 = model4.layers[0].get_output_at(0).get_shape().as_list()
	num_bank4 = model4_shape0[2] 
	model4_shape1 = model4.layers[1].get_output_at(0).get_shape().as_list()
	timestepRec4 = model4_shape1[1]
	kernel_width4= model4_shape0[1]-model4_shape1[1]+1
	filter_len4  = model4_shape0[2]-model4_shape1[2]+1
	df_w = input("Use default ensemble weight (1,1,1,1) for models?(y/n): ")
	if(df_w == "y"):
		w = [1,1,1,1]
	else:
		w1 = int(input("weight for model 1:"))
		w2 = int(input("weight for model 2:"))
		w3 = int(input("weight for model 3:"))
		w4 = int(input("weight for model 4:"))
		w = [w1,w2,w3,w4]
# network params
max_pool_len = num_bank - filter_len + 1
max_pool_len2 = num_bank2 - filter_len2 + 1
dropout_rate = 0.3

# Data generator
dim = (timestepRec,num_bank)
dim2 = (timestepRec2,num_bank2)
if(train_test == "0"):
	test_params = {    'dim':dim, 'fs': fs, 'yaml_path':test_yaml_path,    'yaml_file':test_yaml_file,'audio_path':test_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}

	test_params2 = {    'dim':dim2, 'fs': fs, 'yaml_path':test_yaml_path,    'yaml_file':test_yaml_file,'audio_path':test_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}
	test_id = {}
	for i in range(len(test_data)):    
		test_id[i] = i

if(train_test == "1"):
	train_params = {    'dim':dim, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}
	train_params2 = {    'dim':dim2, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}
	train_id = {}
	for i in range(len(train_data)):
		train_id[i] = i

if(m3 == "y"):
	dim3 = (timestepRec3,num_bank3)
	max_pool_len3 = num_bank3 - filter_len3 + 1
	if(train_test == "1"):
		train_params3 = {    'dim':dim3, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}
	else:
		test_params3 = {    'dim':dim3, 'fs': fs, 'yaml_path':test_yaml_path,    'yaml_file':test_yaml_file,'audio_path':test_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}

if(m4 == "y"):
	dim4 = (timestepRec4,num_bank4)
	max_pool_len4 = num_bank4 - filter_len4 + 1
	if(train_test == "1"):
		train_params4 = {    'dim':dim4, 'fs': fs, 'yaml_path':train_yaml_path,    'yaml_file':train_yaml_file,'audio_path':train_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}
	else:
		test_params4 = {    'dim':dim4, 'fs': fs, 'yaml_path':test_yaml_path,    'yaml_file':test_yaml_file,'audio_path':test_audio_path,    'sample_spectrogram':sample[0:1],     'nperseg':nperseg, 'noverlap':noverlap}

audio_len = sample.shape[0]
sample_w = [1.0,1.0]

if(train_test == '1'):
	my_generator = DataGenerator(mode,"_test",sample_w,train_id, audio_len,kernel_width, **train_params,shuffle=False)
	my_generator2 = DataGenerator(mode,"_test",sample_w,train_id, audio_len,kernel_width2, **train_params2,shuffle=False)
	if(m3 == "y"):
		my_generator3 = DataGenerator(mode,"_test",sample_w,train_id, audio_len,kernel_width3, **train_params3,shuffle=False)
	if(m4 == "y"):
		my_generator4 = DataGenerator(mode,"_test",sample_w,train_id, audio_len,kernel_width4, **train_params4,shuffle=False)
else:
	my_generator = DataGenerator(mode,"_test",sample_w,test_id,audio_len,kernel_width, **test_params,shuffle=False)
	my_generator2 = DataGenerator(mode,"_test",sample_w,test_id, audio_len,kernel_width2, **test_params2,shuffle=False)
	if(m3 == "y"):
		my_generator3 = DataGenerator(mode,"_test",sample_w,test_id,audio_len,kernel_width3, **test_params3,shuffle=False)

	if(m4 == "y"):
		my_generator4 = DataGenerator(mode,"_test",sample_w,test_id,audio_len,kernel_width4, **test_params4,shuffle=False)
# Start to test
num_batch = int(my_generator.__len__())

#from Ipython.display import clear_output

def test_par(ii):

	print("Preparing %s/%s batch" %(ii+1, num_batch))
	X,y = my_generator.__getitem__(ii)
	return X,y


#num_cores = multiprocessing.cpu_count()

#X = []
#y = []

#for Xs,ys in(Parallel(n_jobs=num_cores)(delayed(test_par)(ii) for ii in range(num_batch))):
#	X.append(Xs)
#	y.append(np.flip(ys,1))

for i_batch in range(num_batch):

	print("Testing on %s/%s batch" %(i_batch+1, num_batch))

	X,y = my_generator.__getitem__(i_batch)
	X2,_= my_generator2.__getitem__(i_batch)

	#print("X[0].shape ",X[0].shape)
	#print("y.shape ",y.shape)
	y_pred_0 = model.predict(X[0], batch_size=None, verbose=0, steps=None)
	y_pred_1 = model.predict(X[1], batch_size=None, verbose=0, steps=None)
	y_pred_2 = model.predict(X[2], batch_size=None, verbose=0, steps=None)
	y_pred_0 = np.flip(y_pred_0,1)
	y_pred_1 = np.flip(y_pred_1,1)
	y_pred_2 = np.flip(y_pred_2,1)
	batch_pred = copy.deepcopy(y_pred_0)/3
	batch_pred[:,1:,:] += y_pred_1[:,0:y_pred_1.shape[1]-1,:]/3
	batch_pred[:,2:,:] += y_pred_2[:,0:y_pred_2.shape[1]-2,:]/3
	
	y_pred_0 = model2.predict(X2[0], batch_size=None, verbose=0, steps=None)
	y_pred_1 = model2.predict(X2[1], batch_size=None, verbose=0, steps=None)
	y_pred_2 = model2.predict(X2[2], batch_size=None, verbose=0, steps=None)
	y_pred_0 = np.flip(y_pred_0,1)
	y_pred_1 = np.flip(y_pred_1,1)
	y_pred_2 = np.flip(y_pred_2,1)
	batch_pred2 = copy.deepcopy(y_pred_0)/3
	batch_pred2[:,1:,:] += y_pred_1[:,0:y_pred_1.shape[1]-1,:]/3
	batch_pred2[:,2:,:] += y_pred_2[:,0:y_pred_2.shape[1]-2,:]/3
	
	if(m3 == "y"):
		X3,_= my_generator3.__getitem__(i_batch)
		y_pred_0 = model3.predict(X3[0], batch_size=None, verbose=0, steps=None)
		y_pred_1 = model3.predict(X3[1], batch_size=None, verbose=0, steps=None)
		y_pred_2 = model3.predict(X3[2], batch_size=None, verbose=0, steps=None)
		y_pred_0 = np.flip(y_pred_0,1)
		y_pred_1 = np.flip(y_pred_1,1)
		y_pred_2 = np.flip(y_pred_2,1)
		batch_pred3 = copy.deepcopy(y_pred_0)/3
		batch_pred3[:,1:,:] += y_pred_1[:,0:y_pred_1.shape[1]-1,:]/3
		batch_pred3[:,2:,:] += y_pred_2[:,0:y_pred_2.shape[1]-2,:]/3
	
	if(m4 == "y"):
		X4,_= my_generator4.__getitem__(i_batch)
		y_pred_0 = model4.predict(X4[0], batch_size=None, verbose=0, steps=None)
		y_pred_1 = model4.predict(X4[1], batch_size=None, verbose=0, steps=None)
		y_pred_2 = model4.predict(X4[2], batch_size=None, verbose=0, steps=None)
		y_pred_0 = np.flip(y_pred_0,1)
		y_pred_1 = np.flip(y_pred_1,1)
		y_pred_2 = np.flip(y_pred_2,1)
		batch_pred4 = copy.deepcopy(y_pred_0)/3
		batch_pred4[:,1:,:] += y_pred_1[:,0:y_pred_1.shape[1]-1,:]/3
		batch_pred4[:,2:,:] += y_pred_2[:,0:y_pred_2.shape[1]-2,:]/3
	
	y_comb = y[0,:,1]
	batch_size = y.shape[0]
	#print("y_comb shape:",y_comb.shape)
	if(m3 == "n"):
		y_pred_comb = (w[0]*batch_pred[0,:,1]+w[1]*batch_pred2[0,:,1])
		for j in range(1,batch_size): 
			y_comb = np.hstack((y_comb,y[j,:,1]))
			y_pred_comb =  np.hstack((y_pred_comb,(w[0]*batch_pred[j,:,1]+w[1]*batch_pred2[j,:,1])))
		y_pred_comb /= (w[0]+w[1])
	elif(m4 == "n"):	
		y_pred_comb = (w[0]*batch_pred[0,:,1]+w[1]*batch_pred2[0,:,1]+w[2]*batch_pred3[0,:,1])
		for j in range(1,batch_size): 
			y_comb = np.hstack((y_comb,y[j,:,1]))
			y_pred_comb =  np.hstack((y_pred_comb,(w[0]*batch_pred[j,:,1]+w[1]*batch_pred2[j,:,1]+w[2]*batch_pred3[j,:,1])))
		y_pred_comb /= (w[0]+w[1]+w[2])

	else:	
		y_pred_comb = (w[0]*batch_pred[0,:,1]+w[1]*batch_pred2[0,:,1]+w[2]*batch_pred3[0,:,1]+w[3]*batch_pred4[0,:,1])
		for j in range(1,batch_size): 
			y_comb = np.hstack((y_comb,y[j,:,1]))
			y_pred_comb =  np.hstack((y_pred_comb,(w[0]*batch_pred[j,:,1]+w[1]*batch_pred2[j,:,1]+w[2]*batch_pred3[j,:,1]+w[3]*batch_pred4[j,:,1])))
		y_pred_comb /= (w[0]+w[1]+w[2]+w[3])
	output_labels, predict_event_list = event_list_gen(y_pred_comb,0.6,13,28)
	dummy        , ground_truth_event_list = event_list_gen(y_comb,0.5,1,1)
	if(len(predict_event_list)>0):	
		predict_event_list = (predict_event_list+i_batch*timestepRec*batch_size)*(nperseg-noverlap)/44100        
	if(len(ground_truth_event_list)>0):
		ground_truth_event_list = (ground_truth_event_list+i_batch*timestepRec*batch_size)*(nperseg-noverlap)/44100
		
	dummy_list = np.zeros(2)
	dummy_list[1]= 0.001
	if(i_batch == 0):
		if(len(predict_event_list)>0):	
			predict_event_list_total = predict_event_list
		else:
			predict_event_list_total = copy.deepcopy(dummy_list)
		if(len(ground_truth_event_list)>0):	
			ground_truth_event_list_total = ground_truth_event_list
		else:
			ground_truth_event_list_total = copy.deepcopy(dummy_list)
	else:
		if(len(predict_event_list)>0):	
			predict_event_list_total = np.hstack((predict_event_list_total,predict_event_list))

		if(len(ground_truth_event_list)>0):
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


model_name += input("eval serial number (e.g 01):")
if(train_test == '0'):	
	np.savetxt("event_list/"+model_name+"_ens_test.txt",            predict_event_list_total, fmt='%.4f',            newline=newline_,)


	np.savetxt("event_list/"+model_name+"_ens_test_truth.txt",            ground_truth_event_list_total, fmt='%.4f',            newline=newline_,)

else:

	np.savetxt("event_list/"+model_name+"_ens_train_test.txt",            predict_event_list_total, fmt='%.4f',            newline=newline_,)


	np.savetxt("event_list/"+model_name+"_ens_train_test_truth.txt",            ground_truth_event_list_total, fmt='%.4f',            newline=newline_,)

print("Test done!")


