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
from sound_detection_utils_tf import *


train_yaml_path = "audio_set/DCASE/devtrain/10000/meta"
train_yaml_file = "mixture_recipes_devtrain_babycry.yaml"
print("loading train data ...")
train_data = read_meta_yaml(train_yaml_path +"/" + train_yaml_file)
print("loaded train data 10000, length=",len(train_data))
for i in range(4):
    filename = "mixture_recipes_devtrain_babycry_s"+str(i+1)+".yaml"
    data = train_data[i*2500:min((i+1)*2500,len(train_data))]
    print("writing into ",filename)
    with open(train_yaml_path+"/"+filename, 'w') as outfile:
        outfile.write(yaml.dump(data,default_flow_style=False))
