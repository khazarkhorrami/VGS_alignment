# import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.6
# sess = tf.Session(config=config) 

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

###############################################################################
# this file first loads test data nouns 
# and then select corrected ones based on spellcheck applied in step 2
# next it compares the test caption words with the corresponding image object labels gathered in step 4 (unifying labels) using word2vec similarity function
# it keeps the caption words that are similar to image object labels based on a defined similarity threshold
# based on above comparison, it removes unsimilar words from caption words and substitues the accepted words with object image labels
###############################################################################

thresh = 0.5

path_in_nouns = '/worktmp/hxkhkh/project_3/output/step_1/2/'
file_in_nouns = 'noun_data.mat'

path_in_spellcheck = '/worktmp/hxkhkh/project_3/output/step_1/3/'
file_in_spellcheck = 'corrected_nouns_index.mat'

path_in_labels = '/worktmp/hxkhkh/project_3/output/step_1/4/'
file_in_labels = 'unified_labels.mat'

path_out = '/worktmp/hxkhkh/project_3/output/step_1/6/'
file_out = 'sub_labels.mat'
###############################################################################

###############################################################################

import numpy
import scipy.io
from nltk.tokenize import wordpunct_tokenize
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp/hxkhkh/project_temp/step_9/GoogleNews-vectors-negative300.bin', binary=True)

############################################################################### loading the test caption annotation nouns

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

############################################################################### correct caption indexes

data = scipy.io.loadmat(path_in_spellcheck + file_in_spellcheck, variable_names = ['ind_accepted'])

ind_accepted = data['ind_accepted'][0]
wavfile_nouns = [wavfile_nouns[item] for item in ind_accepted]

wavfile_onsets = [wavfile_nouns_onsets[item][0] for item in ind_accepted]
wavfile_offsets = [wavfile_nouns_offsets[item][0] for item in ind_accepted]
###############################################################################loading test image object labels

data = scipy.io.loadmat(path_in_labels + file_in_labels, variable_names =['ref_names_all'])

ref_names_all = data['ref_names_all'][0]

#ref_names_all = [ref_names_all[item] for item in ind_accepted]
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
    
############################################################################### function for word2vec similarity comparison
#.............................................................................. loop over all data
# note: images without any coco label are detected as 0 in all_accepted_words

all_accpted_words =[]
all_accpted_onsets =[]
all_accpted_offsets =[]
all_accepted_ind = []
for counter_image, nounlist in enumerate(wavfile_nouns):
    
    print('............................................................')
    print(nounlist)
    imagelabels_tokenized = ref_names_tokenized[counter_image]
    imagelabels = ref_names_all[counter_image]
    
    if imagelabels_tokenized:
        accepted_ind = []
        substitute_label = []
        substitute_onset = []
        substitute_offset = []
        for counter_label, candidate_noun in enumerate(nounlist):
        
            sim = [model.n_similarity([candidate_noun],item) for item in imagelabels_tokenized]
            print(sim)
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
    
#.............................................................................. saving the results
    
scipy.io.savemat(path_out + file_out , {'all_accpted_words':all_accpted_words,'all_accpted_onsets':all_accpted_onsets , 'all_accpted_offsets':all_accpted_offsets,
                                               'all_accepted_ind':all_accepted_ind})