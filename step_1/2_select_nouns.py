#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:15:12 2020

@author: hxkhkh
"""
# Selecting and saving nouns in each utterance of test set

# At this step we went through all spoken captions in our validation (test) set,
# for each caption, we study set of words of that capton, and using nltk tool detec the nouns out of all words presented in each sentence.
# saving the indexes of nouns, we extracted corresponding onset times and offset times of each noun.
# As output we saved the dictionary with three instances of caption nouns, caption noun onsets, and caption noun offsets. 
###############################################################################
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.6
# sess = tf.Session(config=config) 

###############################################################################
import pickle
import numpy
import scipy.io
import nltk

path_out = '/worktmp/hxkhkh/project_3/output/step_1/2/'
path_in = '/worktmp/hxkhkh/project_3/output/step_1/1/'

file_in = 'word_data.mat'
file_out = 'noun_data.mat'
###############################################################################
#                           Calling Processed data                            #
############################################################################### 

wordData = scipy.io.loadmat(path_in + file_in,
                     variable_names=['wavfile_words','wavfile_words_onsets', 'wavfile_words_offsets','wavfile_caption'])  

wavfile_words = wordData['wavfile_words'][0]
wavfile_words_onsets = wordData['wavfile_words_onsets'][0]
wavfile_words_offsets = wordData['wavfile_words_offsets'][0]



wavfile_words_onsets = [item[0] for item in wavfile_words_onsets] 
wavfile_words_offsets = [item[0] for item in wavfile_words_offsets]  
temp = wavfile_words
wavfile_words = []
for utterance in temp:
    correct_utterance = []
    for phoneme in utterance:
        correct_ph  = phoneme.strip()
        correct_ph = correct_ph.lower()
        correct_utterance.append(correct_ph)
    wavfile_words.append(correct_utterance)
   
###############################################################################
#                           speech check function                            #
############################################################################### 

def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return noun_indexes
    
###############################################################################
#                           specch check function                            #
###############################################################################

saved_indexes = []

#tok = nltk.pos_tag (wavfile_words[0])
# candidate_utterance = wavfile_words[34942]
# candidate_nouns_ind = detec_nouns (candidate_utterance)
# candidate_nouns = [candidate_utterance[item] for item in candidate_nouns_ind]
# print(candidate_utterance)
# print(candidate_nouns)

wavfile_nouns = []
wavfile_nouns_onsets = []
wavfile_nouns_offsets = []

for counter_utterance  in range(len(wavfile_words)):
    
    candidate_utterance = wavfile_words[counter_utterance]
    candidate_utterance_onsets = wavfile_words_onsets[counter_utterance]
    candidate_utterance_offsets = wavfile_words_offsets[counter_utterance]
    
    candidate_nouns_ind = detec_nouns (candidate_utterance)
    if candidate_nouns_ind : 
        candidate_nouns = [candidate_utterance[item] for item in candidate_nouns_ind]
        candidate_onsets = [candidate_utterance_onsets[item] for item in candidate_nouns_ind]
        candidate_offsets = [candidate_utterance_offsets[item] for item in candidate_nouns_ind]
        
        wavfile_nouns.append(candidate_nouns)
        wavfile_nouns_onsets.append(candidate_onsets)
        wavfile_nouns_offsets.append(candidate_offsets)
    
    else:
        
        wavfile_nouns.append([])
        wavfile_nouns_onsets.append([])
        wavfile_nouns_offsets.append([])
scipy.io.savemat(path_out + file_out, {'wavfile_nouns':wavfile_nouns,'wavfile_nouns_onsets':wavfile_nouns_onsets,
                                                             'wavfile_nouns_offsets':wavfile_nouns_offsets})

