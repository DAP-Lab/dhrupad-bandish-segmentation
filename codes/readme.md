## Contents
* [get_stm.py](get_stm.py) - to predict s.t.m. values on a test audio
* [model_utils.py](model_utils.py) - contains the data loader and model definition classes
* [params.py](params.py) - contains parameters and some code for data loading (is imported by the train script)
* [split_audios.py](split_audios.py) - to split the full-length recordings into sections from the annotations
* [extract_features_labels.py](extract_features_labels.py) - to generate and save the log-mel-spectrograms and labels of each section's audio
* [train.py](train.py) - the main training script <br><br>

* [splits/](splits/) - contains the list of sections in each fold for all 3 cases - vocals, pakhawaj and net, in separate folders
* [saved_models/](saved_models/) - trained models from each fold for all 3 cases in separate folders
