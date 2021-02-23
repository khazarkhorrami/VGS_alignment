import numpy
import scipy.io
import pickle
###############################################################################
# this file first loads test data nouns + WI images 
# and then select corrected ones based on spellcheck applied in step 2
# next it calculates object mask within time interval of spoken word label and saves WI masks
###############################################################################
path_in_nouns = '/worktmp/hxkhkh/project2/outputs/step_6/step_1/'
path_in_corrected_ind = '/worktmp/hxkhkh/project2/outputs/step_6/step_2/'

path_in_tensor = '/worktmp/hxkhkh/project2/outputs/step_5/hidden/'
file_in_tensor = 'SI_CNN2_v2.mat'

path_in_processed_nouns = '/worktmp/hxkhkh/project2/outputs/step_6/step_5/'

path_out = '/worktmp/hxkhkh/project2/outputs/step_7/step_1/'
file_out = 'sub_labels_masks_SI'
###############################################################################
############################################################################### output of word captions, onsets and offsets
data = scipy.io.loadmat(path_in_nouns + 'noun_data.mat', variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])

#wavfile_nouns = data ['wavfile_nouns'][0]
wavfile_nouns_onsets = data ['wavfile_nouns_onsets'][0]
wavfile_nouns_offsets = data ['wavfile_nouns_offsets'] [0]

# obj = wavfile_nouns
# wavfile_nouns = []
# wavfile_nouns = [item for item in obj]

obj = wavfile_nouns_onsets
wavfile_nouns_onsets = []
wavfile_nouns_onsets = [item for item in obj]  

obj = wavfile_nouns_offsets
wavfile_nouns_offsets = []
wavfile_nouns_offsets = [item for item in obj]  

#..............................................................................
# temp = wavfile_nouns
# wavfile_nouns = []
# for utterance in temp:
#     correct_utterance = []
#     for phoneme in utterance:
#         correct_ph  = phoneme.strip()
#         correct_utterance.append(correct_ph)
#     wavfile_nouns.append(correct_utterance) 

############################################################################### output of spell correction

data = scipy.io.loadmat(path_in_corrected_ind + 'corrected_nouns_index.mat', variable_names = ['ind_accepted'])

ind_accepted = data['ind_accepted'][0]

#wavfile_nouns_corrected = [wavfile_nouns[item] for item in ind_accepted]
wavfile_onsets_corrected = [wavfile_nouns_onsets[item][0] for item in ind_accepted]
wavfile_offsets_corrected = [wavfile_nouns_offsets[item][0] for item in ind_accepted]

#del wavfile_nouns
del wavfile_nouns_onsets
del wavfile_nouns_offsets

wavfile_nouns_onsets = [item/8 for item in wavfile_onsets_corrected]
wavfile_nouns_offsets = [item/8 for item in wavfile_offsets_corrected]

del wavfile_onsets_corrected
del wavfile_offsets_corrected

# next step: to remove empty lists from set of selected nouns/onsets/offsets
############################################################################### loading the model output

data = scipy.io.loadmat(path_in_tensor + file_in_tensor  , variable_names = ['out_layer'])

out_layer = data['out_layer']
out_layer = out_layer [ind_accepted]

############################################################################### loading substitue nouns and coressponding indexes
data = scipy.io.loadmat(path_in_processed_nouns + 'sub_labels.mat', variable_names = ['all_accpted_words','all_accepted_ind'])

#all_subnouns = data ['all_accpted_words'][0]
all_subinds = data ['all_accepted_ind'][0]

all_detected_masks = []
for counter_image in range(len(all_subinds)):
    #simulated_nouns = all_subnouns [counter_image]
    simulated_nouns_ind = all_subinds[counter_image]
    
    #print(simulated_nouns)
    print(simulated_nouns_ind)
    print('...............................')
    im_masks = []
    if len(simulated_nouns_ind): # if not empty
        print(simulated_nouns_ind)
        simulated_nouns_ind = simulated_nouns_ind[0]
        caption_onsets =  numpy.asarray(wavfile_nouns_onsets[counter_image],dtype='int')
        caption_offsets =  numpy.asarray(wavfile_nouns_offsets[counter_image],dtype='int')
        
        for counter_candidates, ind_word in enumerate(simulated_nouns_ind):
            #print(ind_word)
            
            word_onset = caption_onsets[ind_word]
            word_offset = caption_offsets[ind_word] + 1
            mask_detected = out_layer[counter_image,  word_onset:word_offset , : ]
            im_masks.append((mask_detected))
            # word_label_list = simulated_nouns[counter_candidates]
            # if len(word_label_list)==1:
            #     label = word_label_list[0].strip()
            # elif len(word_label_list)==2:
            #     label =  word_label_list[0].strip() + ' '+ word_label_list[1].strip()
            #print('....' + label+ '....')
        
    all_detected_masks.append((im_masks))
    

#scipy.io.savemat(out_path + 'sub_labels_masks.mat', {'all_WA_masks':all_WA_masks})
#numpy.save(out_path + 'sub_labels_masks.mat' , all_WI_masks )



file = open(path_out + file_out ,'wb')
new_dict = pickle.dump(all_detected_masks, file)
file.close()