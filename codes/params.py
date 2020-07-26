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

#path to audio dataset
datadir=''

#path to save trained models
model_dir='./saved_models'

#path to save plots
plot_dir='./plots'

#input-output parameters
fs=16000
winsize=int(0.04*fs)
hopsize=int(0.02*fs)
input_height=40
input_len=400

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

mode='voc' #'voc', 'pakh' or 'net'
classes_dict={'voc':[1.,2.,4.,8.],'pakh':[1.,2.,4.,8.,16.],'net':[1.,2.,4.,8.,16.]}
classes=classes_dict[mode]
n_classes=len(classes)

#minimum section duration for smoothing s.t.m. estimates
min_sec_dur=5 #in seconds
min_sec_dur/=input_hop_sec

#cross-validation folds for training
if datadir != '':
	songlist=os.listdir(datadir)
	labels_stm = np.load(os.path.join(datadir,'labels_stm.npy'),allow_pickle=True).item()

	fold = 0 # 0, 1 or 2
	partition = {'train':[], 'validation':[]}
	n_folds=3
	all_folds=[]
	for i_fold in range(n_folds):
		all_folds.append(np.loadtxt('./splits/%s/fold_%d.csv'%(mode,i_fold),delimiter=',',dtype='str'))

	val_fold=all_folds[fold]
	train_fold=np.array([])
	for i_fold in np.delete(np.arange(0,n_folds),fold):
		if len(train_fold)==0: train_fold=all_folds[i_fold]
		else: train_fold=np.vstack((train_fold,all_folds[i_fold]))

	for song in songlist:
		try:
			ids = glob.glob(datadir+song+'/*.pt')
		except:
			continue
		section_name='_'.join(song.split('_')[0:4])
		if section_name in ['UB_Bhrv_Sooltal_1','GB_Rageshri_Choutal_part']:
			section_name='_'.join(song.split('_')[0:4])

		if section_name in val_fold[:,0]:	partition['validation'].extend(ids)
		elif section_name in train_fold[:,0]:	partition['train'].extend(ids)


