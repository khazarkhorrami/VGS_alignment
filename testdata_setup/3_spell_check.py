
"""

This file reads noun_data obtained at step 2 and does followings
    1) checks utterance captions for possible spelling errors
    2) removes the utterances with any spelling error from test data

"""

###############################################################################

import os
import scipy.io
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/.../GoogleNews-vectors-negative300.bin', binary=True)

###############################################################################

path_in =  "../../testdata/2/"
path_out = "../../testdata/3/"

###############################################################################

def spell_checking():
    data = scipy.io.loadmat(path_in + 'noun_data.mat', variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])
    
    wavfile_nouns = data ['wavfile_nouns'][0]
    wavfile_nouns_onsets = data ['wavfile_nouns_onsets'][0]
    wavfile_nouns_offsets = data ['wavfile_nouns_offsets'] [0]
      
    wavfile_nouns_onsets = [item for item in wavfile_nouns_onsets] 
    wavfile_nouns_offsets = [item for item in wavfile_nouns_offsets]  
    
    temp = wavfile_nouns
    wavfile_nouns = []
    for utterance in temp:
        correct_utterance = []
        for phoneme in utterance:
            correct_ph  = phoneme.strip()
            correct_utterance.append(correct_ph)
        wavfile_nouns.append(correct_utterance)   

    # saving index of error-free utterances
       
    ind_accepted = []    
    for counter_utterance, candidate_utterance  in enumerate(wavfile_nouns):        
        print(counter_utterance)
        try:
            model.most_similar(candidate_utterance)
            ind_accepted.append(counter_utterance)
        except:
            print("An exception occurred in word:  " + str(candidate_utterance))
            
    # saving results
    scipy.io.savemat(path_out + 'corrected_nouns_index.mat', {'ind_accepted':ind_accepted})


if __name__ == '__main__': 
    
    os.makedirs(path_out, exist_ok=1)
    spell_checking ()