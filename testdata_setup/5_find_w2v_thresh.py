
"""

This file first loads test data nouns (output of step 2), and does followings
    1) selects corrected ones based on spellcheck applied in step 3, 
    2) compares the test caption nouns with the corresponding image object labels
     using word2vec similarity function
    3) saves all similarites
    4) plots the histogram of similarities for all noun-label pairs

this histogram is used to detect a proper threshold for noun-label similarity


"""

###############################################################################

import os
import numpy
import scipy.io
from nltk.tokenize import wordpunct_tokenize
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', binary=True)

###############################################################################

path_in_nouns = "../../testdata/2/"
file_in_nouns = 'noun_data.mat'

path_in_spellcheck = "../../testdata/3/"
file_in_spellcheck = 'corrected_nouns_index.mat'

path_in_labels = "../../testdata/4/"
file_in_labels = 'unified_labels.mat'

path_out = "../../testdata/5/"
file_out = 'w2vthresh'

###############################################################################

def getting_w2v_similarities():
    
    # loading the test caption annotation nouns
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
    
    # correct caption indexes base on spell check   
    data = scipy.io.loadmat(path_in_spellcheck + file_in_spellcheck, variable_names = ['ind_accepted'])
    ind_accepted = data['ind_accepted'][0]
    wavfile_nouns_corrected = [wavfile_nouns[item] for item in ind_accepted]
    
    # loading test image object labels    
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
    
    # to convert labels with multiple words (e.g. dinning table) to tokenized formet)      
    ref_names_tokenized = []
    for datacounter, labellist in enumerate(ref_names_all):
        tokenized_list = []
        for item in labellist:
            
            item_tokenized = wordpunct_tokenize(item)
            tokenized_list.append(item_tokenized)
        ref_names_tokenized.append(tokenized_list)
    
        
    # word2vec similarity comparison 
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
    
    #  plotting the histogram results
    
    
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


if __name__ == '__main__':
    
    os.makedirs(path_out, exist_ok=1)
    getting_w2v_similarities()