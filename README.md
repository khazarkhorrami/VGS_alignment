

# VGS_alignment

Audio-Visual alignment analysis in a cross-modal attention -based VGS model.


Code for the paper "Evaluation of Audio-Visual alignments in Visually Gropunded Speech Modelos" by Khazar Khorrami and Okko Räsänen, presented at interspeech 2021.

Link to the paper: https://arxiv.org/pdf/2108.02562.pdf

## Implementation details

All code is developed and tested in keras using Tensorflow 1.15.0. Example Anaconda environment can be prepared as below:

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

"extract_audio_features.py" extracts log mel-band energis for spoken captions.

"extract_visual_features.py" extracts vgg16 features for images.

## model

This folder contains all the python files needed for training, validation, and testing the various model versions.
The training and validation losses as well as validation recalls at each epoch are saved as 'valtrainloss.mat'.

## testdata_setup

As for test data, validation set of MSCOCO and SPEECH-COCO are used. The data is processed in 6 successive steps. Output of each step is used in next step/steps. So, you may want to run the codes in same order. At last step, testdata nouns are obtained as a set of object labels and saved along with their timestamps.


## scores

This path includes functions needed to measure alignment and glancing scores as described in the paper.


