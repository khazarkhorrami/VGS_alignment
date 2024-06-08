

# Evaluation of Word-Object Alignment in Visually Grounded Speech Processing

This repository contains the code for audio-visual alignment analysis in cross-modal attention-based visually grounded speech models.

## Reference

If you use this code, please cite the following paper:

**Paper:** "Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models" by Khazar Khorrami and Okko Räsänen, presented at Interspeech 2021.

**Link to the paper:** [https://www.isca-archive.org/interspeech_2021/khorrami21_interspeech.pdf](https://www.isca-archive.org/interspeech_2021/khorrami21_interspeech.pdf)


# Audio-visual alignment tensor

<img src="https://github.com/khazarkhorrami/VGS_alignment/assets/33454475/989ff7a1-0c78-4d34-a65c-7cd76e332b78" alt="tensor" width="800" height="500">



## Implementation details

All code is developed and tested in keras 2.3.1 using Tensorflow 1.15.0. Example Anaconda environment can be prepared as below:

conda create --name VGSalignment python=3.7

source activate VGSalignment

conda install -c anaconda tensorflow-gpu=1.15.0

conda install -c anaconda spyder

conda install -c conda-forge librosa

pip install opencv-python

conda install -c anaconda nltk

conda install -c anaconda gensim

***

To train the model, you need to first download MSCOCO and SPEECH-COCO datasets and store them in "../data" folder.

To run "testdata_setup", you need to download "GoogleNews-vectors-negative300.bin" model and store it in "../data" folder.

***


## features_extraction

This folder contains all necessary functions for reading data, extracting audio and visual features and saving them to "../feature" folder after shuffling and chunking features. The output features (provided in chunks) are then served to model training and validation by indicating their names ("set_of_train_files" and "set_of_validation_files" ) in model configuration script.  

"extract_audio_features.py" extracts log mel-band energis for spoken captions.

"extract_visual_features.py" extracts vgg16 features for images.



## model

This folder contains all the python files needed for training, validation, and testing the various model versions.
The training and validation losses as well as validation recalls at each epoch are saved as 'valtrainloss.mat'. At the same time, model weights are saved as model_weights.h5 file and updated in each epoch. To resume the training processe for a specific model, you need to set "use_pretrained" as "True" in configuration file, to enable the code to use previousely saved results and weights at starting point. 

## testdata_setup

As for test data, validation set of MSCOCO and SPEECH-COCO are used. The data is processed in 6 successive steps. Output of each step is used in next step/steps. So, you may want to run the codes in same order from 0 to 6. At last step, testdata nouns are obtained as a set of object labels and saved along with their timestamps.


## scores

This path includes functions needed to measure alignment and glancing scores as described in the paper.

Before running the code for score measurements, hidden layer weights (i.e., AudioVisual-tensor, e.g. example_tensor.mat) must be obtained and saved under the folder "../model/AVtensors" as an array of dimensions (N_samples ,time_frames, pixel_h*pixel_w). 


