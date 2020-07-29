import sys
import os
import numpy as np
import librosa
from params import fs

audio_dir=sys.argv[1]
save_dir=os.path.join(audio_dir,'audio_sections')
if not os.path.exists(save_dir): os.mkdir(save_dir)

annotations=np.loadtxt('../annotations/section_boundaries_labels.csv',delimiter=',',dtype=str)

song='' #leave this line as it is
for item in annotations:
	if '_'.join(item[0].split('_')[:-1]) != song:
		song='_'.join(item[0].split('_')[:-1])
		try:
			x,fs=librosa.load(os.path.join(audio_dir,song+'.wav'),sr=fs)
		except FileNotFoundError:
			print('Audio for %s not found'%song)
			song=''
			continue

	start=int(float(item[1])*fs)
	end=int(float(item[2])*fs)
	y=x[start:end]
	librosa.output.write_wav(os.path.join(save_dir,item[0]+'.wav'),y,fs)
