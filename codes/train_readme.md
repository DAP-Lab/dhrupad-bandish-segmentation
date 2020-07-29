## Instructions for performing the  cross-validation experiments in the paper

1. Download all the audio files (read the Dunya python package documentation to learn how audios can be obtained using the provided MBID).
2. We need to extract and save separate audio files corresponding to sections in each audio. For this, run the ```split_audios.py``` script, providing the path to the audio directory (where the above downloaded audios are saved) as a command line argument.
   The audios will be saved in a folder called ```audio_sections``` within the folder containing the full-length audios.
3. Generate time-scaled versions({0.8, 0.84, 0.88, ..., 1.2}) of all the section audios using [rubberband](https://breakfastquay.com/rubberband/), and save the audio files as ```<filename>_t.wav``` where *t* is the time-scaling factor.
4. Run ```extract_features_labels.py``` with the following as cmd line arguments: 
  * ```audio_dir```: path to ```audio_sections```
  * ```save_dir```: path to save extracted features and labels
  * ```mode```: *net* or *pakh* or *voc* indicating what the source in the audio is. This is needed to adjust the set of output classes and data augmentation parameters accordingly.
5. Finally, run the training script ```train.py```, with the following cmd line arguments:
  * ```data_dir```: path to the extracted features and labels
  * ```mode```: *net* or *pakh* or *voc* indicating what the source in the audio is
  * ```fold```: cross-validation fold number out of *0*, *1*, *2*
6. Trained models and loss curves get saved in the locations indicated in the ```params.py``` file (*plot_dir*, *model_dir*)

### Note: Assumed file name format
* For concert audio: ```<Artist>_<Raga>_<Tala>.wav```
* For extracted audio sections: ```<Artist>_<Raga>_<Tala>_<Section #>.wav```
* For time-scaled section audios: ```<Artist>_<Raga>_<Tala>_<Section #>_<scaling-factor>.wav```
