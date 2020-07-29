# Dhrupad Vocal Bandish Segmentation
This repository contains annotations of metric cycle (*tala*) boundaries and of surface tempo based structural segments of the *bandish* portions of a Dhrupad vocal concert dataset, and codes for automatic surface tempo multiple estimation and structural segmentation. The repository is linked to the following publications: </br> </br>
```
Rohit M. A., Vinutha T. P., Preeti Rao. “Structural Segmentation of Dhrupad Vocal Bandish Audio 
based on Tempo”, 21st International Society for Music Information Retrieval Conference,
Montréal, Canada, 2020
```
</br>
```
Rohit M A, Preeti Rao. “Structural Segmentation of Dhrupad Vocal Bandish Audio 
based on Tempo”, Unpublished technical report, 2020
```

The annotations were created manually by one of the authors in consultation with a musician. Trained models are made available to obtain predictions on any test audio. Training scripts are also provided to reproduce the results reported in the paper.

### Contents
* [annotations](./annotations) - annotations for all the audios used in the paper </br>
* [codes](./codes) - scripts to use the trained models to obtain surface tempo multiple estimates on a test audio, as well as to reproduce the cross-validation results </br>

* [audio_samples](./audio_samples) - mixture and source separated audio clips corresponding to sections *(a)* and *(b)* of Figure 2 in the paper
* [docs](./docs) - PDF files of the submitted paper and the technical report 

More details on the annotation format and running the codes can be found in the respective folders.

### Audio dataset
The sources for all the audios used in the work are listed in the file [dataset_sources.pdf](./annotations/dataset_sources.pdf). Some are available on YouTube, while others are from the CompMusic Dunya <sup>[1](#fn1)</sup> collection and can be obtained through the Dunya API <sup>[2](#fn2)</sup> using the provided MusicBrainz IDs <sup>[3](#fn3)</sup>. </br>

***

<a name="fn1"><sup>1</sup></a>[https://dunya.compmusic.upf.edu/](https://dunya.compmusic.upf.edu/) </br>
<a name="fn2"><sup>2</sup></a>[https://dunya.compmusic.upf.edu/developers/](https://dunya.compmusic.upf.edu/developers/) </br>
<a name="fn3"><sup>3</sup></a>[https://musicbrainz.org/doc/MusicBrainz_Identifier](https://musicbrainz.org/doc/MusicBrainz_Identifier) </br>
