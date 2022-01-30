
"""

This file reads word_data obtained ate step_1
and does followings
    1) selects and saves nouns in each utterance of test set
    
# At this step we went through all spoken captions in our validation (test) set,
# corresponding onset times and offset times of each noun is extracted.
# Output: a dictionary with three instances of caption nouns, caption noun onsets, and caption noun offsets. 

"""

###############################################################################

import os
import scipy.io
import nltk

###############################################################################


path_in = "../../testdata/1/"
file_in = 'word_data.mat'

path_out = "../../testdata/2/"
file_out = 'noun_data.mat'

###############################################################################

 
def detec_nouns (input_words):
    tok = nltk.pos_tag(input_words)
    noun_indexes = [counter for counter in range(len(tok)) if tok[counter][1] =='NN' or tok[counter][1] =='NNS' ]
    return noun_indexes


def getting_noun_data():
    
    # Calling word data   
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
       
    wavfile_nouns = []
    wavfile_nouns_onsets = []
    wavfile_nouns_offsets = []
    
    for counter_utterance  in range(len(wavfile_words)):
        
        candidate_utterance = wavfile_words[counter_utterance]
        candidate_utterance_onsets = wavfile_words_onsets[counter_utterance]
        candidate_utterance_offsets = wavfile_words_offsets[counter_utterance]
        
        candidate_nouns_ind = detec_nouns (candidate_utterance)
        candidate_nouns = [candidate_utterance[item] for item in candidate_nouns_ind]
        candidate_onsets = [candidate_utterance_onsets[item] for item in candidate_nouns_ind]
        candidate_offsets = [candidate_utterance_offsets[item] for item in candidate_nouns_ind]
        
        wavfile_nouns.append(candidate_nouns)
        wavfile_nouns_onsets.append(candidate_onsets)
        wavfile_nouns_offsets.append(candidate_offsets)
        
     
    scipy.io.savemat(path_out + file_out, {'wavfile_nouns':wavfile_nouns,'wavfile_nouns_onsets':wavfile_nouns_onsets,
                                                                 'wavfile_nouns_offsets':wavfile_nouns_offsets})

if __name__ == '__main__':
    
    os.makedirs(path_out, exist_ok=1)
    getting_noun_data ()