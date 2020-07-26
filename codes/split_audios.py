import os
import numpy as np
import librosa

audio_dir='../audios'
annotations=np.loadtxt('../annotations/section_boundaries_labels.csv',delimiter=',',dtype=str)

song=''
for item in annotations:
	if '_'.join(item[0].split('_')[:-1]) != song:
		song='_'.join(item[0].split('_')[:-1])
		x,fs=librosa.load(os.path.join(audio_dir,song+'.wav'),sr=16000)

	start=int(float(item[1])*fs)
	end=int(float(item[2])*fs)
	y=x[start:end]
	librosa.output.write_wav(os.path.join(audio_dir.replace('/audios','/audio_sections'),item[0]+'.wav'),y,fs)
