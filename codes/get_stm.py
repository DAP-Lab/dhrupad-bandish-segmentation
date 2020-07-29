import os
import sys
import torch
from torch.utils import data
from model_utils import *
from params import *
import numpy as np
import matplotlib.pyplot as plt
import librosa

#load input audio
audio_path=sys.argv[1]
audio,fs=librosa.load(audio_path,sr=fs)

#convert to mel-spectrogram
melgram=librosa.feature.melspectrogram(audio,sr=fs,n_fft=nfft, hop_length=hopsize, win_length=winsize, n_mels=input_height, fmin=20, fmax=8000)
melgram=10*np.log10(1e-10+melgram)
melgram_chunks=makechunks(melgram,input_len,input_hop)

#load model
mode=sys.argv[2] #net / voc / pakh
classes=classes_dict[mode]
n_classes=len(classes)
model_path=os.path.join(model_dir, mode, 'saved_model_fold_0.pt')
model=build_model(input_height,input_len,n_classes).float().to(device)
model.load_state_dict(torch.load(os.path.join(model_path),map_location=device))
model.eval()

#predict s.t.m. versus time!
stm_vs_time=[]
for chunk in melgram_chunks:
	model_in=(torch.tensor(chunk).unsqueeze(0)).unsqueeze(1).float().to(device)
	model_out=model.forward(model_in)
	model_out=nn.Softmax(1)(model_out).detach().numpy()
	stm_vs_time.append(np.argmax(model_out))

#smooth predictions with a minimum section duration of 5s
stm_vs_time = smooth_boundaries(stm_vs_time,min_sec_dur)

#plot
plt.plot(np.arange(len(stm_vs_time))*0.5,stm_vs_time)
plt.yticks(np.arange(-1,6), ['']+['1','2','4','8','16']+[''])
plt.grid('on',linestyle='--',axis='y')
plt.xlabel('Time (s)',fontsize=12)
plt.ylabel('Surface tempo multiple',fontsize=12)
plt.savefig(os.path.join(plot_dir,audio_path.split('/').split('.wav')[0])+'.png')

