import os
import sys
import glob
import torch
from torch.utils import data
from model_utils import *
import numpy as np

#use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#parameters for data loader
batch_size=32
params = {'batch_size': batch_size,'shuffle': True,'num_workers': 4}
max_epochs = 500

#path to pre-trained models
pretrained_model_dir='./pretrained_models'

#path to save trained models
model_dir='./saved_models'

#path to save plots
plot_dir='./plots'

#input-output parameters
fs=16000

winsize_sec=0.04
winsize=int(winsize_sec*fs)
hopsize_sec=0.02
hopsize=int(hopsize_sec*fs)
nfft=int(2**(np.ceil(np.log2(winsize))))

input_len_sec=8
input_len=int(input_len_sec/hopsize_sec)
input_hop_sec=0.5
input_hop=int(input_hop_sec/hopsize_sec)
input_height=40

classes_dict={'voc':[1.,2.,4.,8.],'pakh':[1.,2.,4.,8.,16.],'net':[1.,2.,4.,8.,16.]}

#minimum section duration for smoothing s.t.m. estimates
min_sec_dur=5 #in seconds
min_sec_dur/=input_hop_sec

