
###############################################################################
# this file first loads test data nouns 
# and then select corrected ones based on spellcheck applied in step 2
# next it compares the test caption words with the corresponding image object labels gathered in step 4 (unifying labels) using word2vec similarity function
# it keeps the caption words that are similar to image object labels based on a defined similarity threshold
# based on above comparison, it removes unsimilar words from caption words and substitues the accepted words with object image labels
###############################################################################


# import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.6
# sess = tf.Session(config=config) 

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

###############################################################################
path_in_nouns = '/worktmp/hxkhkh/project_3/output/step_1/2/'
file_in_nouns = 'noun_data.mat'

path_in_spellcheck = '/worktmp/hxkhkh/project_3/output/step_1/3/'
file_in_spellcheck = 'corrected_nouns_index.mat'

path_in_labels = '/worktmp/hxkhkh/project_3/output/step_1/4/'
file_in_labels = 'unified_labels.mat'

path_out = '/worktmp/hxkhkh/project_3/output/step_1/5/'
file_out = 'w2vthresh'

###############################################################################

import numpy
import scipy.io

from nltk.tokenize import wordpunct_tokenize
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp/hxkhkh/project_temp/step_9/GoogleNews-vectors-negative300.bin', binary=True)
############################################################################### loading the test caption annotation nouns



data = scipy.io.loadmat(path_in_nouns + file_in_nouns, variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])

wavfile_nouns = data ['wavfile_nouns'][0]

temp = wavfile_nouns
wavfile_nouns = []
for utterance in temp:
    correct_utterance = []
    for phoneme in utterance:
        correct_ph  = phoneme.strip()
        correct_utterance.append(correct_ph)
    wavfile_nouns.append(correct_utterance) 

############################################################################### correct caption indexes base on spell check

data = scipy.io.loadmat(path_in_spellcheck + file_in_spellcheck, variable_names = ['ind_accepted'])

ind_accepted = data['ind_accepted'][0]

wavfile_nouns_corrected = [wavfile_nouns[item] for item in ind_accepted]

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

#.............................................................................. this is used to convert labels with multiple words (e.g. dinning table) to tokenized formet)  

ref_names_tokenized = []
for datacounter, labellist in enumerate(ref_names_all):
    tokenized_list = []
    for item in labellist:
        
        item_tokenized = wordpunct_tokenize(item)
        tokenized_list.append(item_tokenized)
    ref_names_tokenized.append(tokenized_list)

    
############################################################################### word2vec similarity comparison


#.............................................................................. loop over all data

# note: images without any coco label are detected as 0 in all_accepted_words

all_sims =[]
all_maxsims = []
count_nolabel = []
for testimcounter, nounlist in enumerate(wavfile_nouns_corrected):
    
    #print('............................................................')
    #print(nounlist)
    imagelabels_tokenized = ref_names_tokenized[testimcounter]
    imagelabels = ref_names_all[testimcounter]
    
    if imagelabels_tokenized:
        
        for candidate_nouncounter, candidate_noun in enumerate(nounlist):
        
            sim = [model.n_similarity([candidate_noun],item) for item in imagelabels_tokenized]
            #print(sim)
            max_sim = numpy.max(sim)
            #argmax_sim = numpy.argmax(sim)
            
            all_sims.extend(sim)
            all_maxsims.append(max_sim)
            
    else:
        #print('........ at image number    ' + str(testimcounter) + '    no label is find')
        count_nolabel.append(testimcounter)
    
#.............................................................................. saving the results
    
scipy.io.savemat(path_out + file_out + '.mat', {'all_sims':all_sims, 'all_maxsims':all_maxsims})

#.............................................................................. plotting the histogram results
from matplotlib import pyplot as plt

plt.figure(figsize=[14,21])
plt.subplot(3,2,1)
plt.hist(all_sims , bins = 10) 
plt.ylabel('10 bins \n\n counts')
plt.title('all similarities')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(3,2,2)
plt.hist(all_maxsims , bins = 10) 
plt.title('maximum similarities')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(3,2,3)
plt.hist(all_sims , bins = 20) 
plt.ylabel('20 bins \n\n counts')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(3,2,4)
plt.hist(all_maxsims , bins = 20) 
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(3,2,5)
plt.hist(all_sims , bins = 50) 
plt.xlabel('\n word2vec similarity')
plt.ylabel('50 bins \n\n counts')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(3,2,6)
plt.hist(all_maxsims , bins = 50) 
plt.xlabel('\n word2vec similarity')
plt.xticks(numpy.arange(0,1.1,0.1))

plt.savefig(path_out + file_out + '.pdf')


plt.figure(figsize=[14,7])
plt.subplot(1,2,1)
plt.title('all similarities')
plt.hist(all_sims , bins = 50) 
plt.xlabel('\n word2vec similarity')
plt.ylabel('\n counts\n')
plt.xticks(numpy.arange(0,1.1,0.1))
plt.subplot(1,2,2)
plt.title('maximum similarities')
plt.hist(all_maxsims , bins = 50) 
plt.xlabel('\n word2vec similarity')
plt.xticks(numpy.arange(0,1.1,0.1))

plt.savefig(path_out + file_out + '_final' + '.pdf')