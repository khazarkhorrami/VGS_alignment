
"""

This file does follwings:
    1) compares utterance nouns with image object labels and 
    2) keeps the captions nouns that are similar to image object labels
    based on a similarity above a given threshold
    3 ) saves the selected nouns for each caption

    note that the threshold is selected subjectively 
    based on histogram of similarities obtained in the previous step
    
"""

###############################################################################

import os
import numpy
import scipy.io
from nltk.tokenize import wordpunct_tokenize
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)

###############################################################################
 
thresh = 0.5

path_in_nouns = "../../testdata/2/"
file_in_nouns = 'noun_data.mat'

path_in_spellcheck = "../../testdata/3/"
file_in_spellcheck = 'corrected_nouns_index.mat'

path_in_labels = "../../testdata/4/"
file_in_labels = 'unified_labels.mat'

path_out = "../../testdata/6/"
file_out = 'sub_labels.mat'

###############################################################################

def selecting_final_test_set ():
    
    # loading the test caption annotation nouns
    data = scipy.io.loadmat(path_in_nouns + file_in_nouns, variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])
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
    
    # correct caption indexes
    
    data = scipy.io.loadmat(path_in_spellcheck + file_in_spellcheck, variable_names = ['ind_accepted'])
    
    ind_accepted = data['ind_accepted'][0]
    wavfile_nouns = [wavfile_nouns[item] for item in ind_accepted]
    
    wavfile_onsets = [wavfile_nouns_onsets[item][0] for item in ind_accepted]
    wavfile_offsets = [wavfile_nouns_offsets[item][0] for item in ind_accepted]
    
    #loading test image object labels
    
    data = scipy.io.loadmat(path_in_labels + file_in_labels, variable_names =['ref_names_all'])
    
    ref_names_all = data['ref_names_all'][0]
    
    ref_names_all = [ref_names_all[item] for item in ind_accepted]
    temp = ref_names_all
    ref_names_all = []
    for utterance in temp:
        correct_utterance = []
        for phoneme in utterance:
            correct_ph  = phoneme.strip()
            correct_utterance.append(correct_ph)
        ref_names_all.append(correct_utterance) 
    
        
    ref_names_tokenized = []
    for datacounter, labellist in enumerate(ref_names_all):
        tokenized_list = []
        for item in labellist:
            
            item_tokenized = wordpunct_tokenize(item)
            tokenized_list.append(item_tokenized)
        ref_names_tokenized.append(tokenized_list)
        
    # word2vec similarity comparison
    # note: images without any coco label are detected as 0 in all_accepted_words
    
    all_accpted_words =[]
    all_accpted_onsets =[]
    all_accpted_offsets =[]
    all_accepted_ind = []
    for counter_image, nounlist in enumerate(wavfile_nouns):
        
        print('............................................................')
        #print(nounlist)
        imagelabels_tokenized = ref_names_tokenized[counter_image]
        imagelabels = ref_names_all[counter_image]
        #print(imagelabels)
        if imagelabels_tokenized:
            accepted_ind = []
            substitute_label = []
            substitute_onset = []
            substitute_offset = []
            for counter_label, candidate_noun in enumerate(nounlist):
            
                sim = [model.n_similarity([candidate_noun],item) for item in imagelabels_tokenized]
                #print(sim)
                max_sim = numpy.max(sim)
                if max_sim >= thresh:
                    
                    selected_label = imagelabels [numpy.argmax(sim)]
                    substitute_label.append(selected_label)
                    substitute_onset.append(wavfile_onsets[counter_image][counter_label])
                    substitute_offset.append(wavfile_offsets[counter_image][counter_label])
                    accepted_ind.append(counter_label)
            
        else:
            substitute_label = 0
            accepted_ind = []
            substitute_onset = []
            substitute_offset = []
            
        all_accpted_words.append(substitute_label)
        all_accpted_onsets.append(substitute_onset)
        all_accpted_offsets.append(substitute_offset)
        all_accepted_ind.append(accepted_ind)
        
    # saving the results   
    scipy.io.savemat(path_out + file_out , {'all_accpted_words':all_accpted_words,'all_accpted_onsets':all_accpted_onsets , 'all_accpted_offsets':all_accpted_offsets,
                                                   'all_accepted_ind':all_accepted_ind})


if __name__ == '__main__':
    
    os.makedirs(path_out, exist_ok=1)
    selecting_final_test_set()