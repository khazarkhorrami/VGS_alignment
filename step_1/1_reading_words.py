# This file reads wavfile metadata (time stamps, etc.) and do followings
# 1. words data is extracted
# 2. words locating at t >512 are removed
# 3. all words are zeropadded


import numpy
import scipy,scipy.io


path_in_jsonData = '/worktmp2/hxkhkh/laptop/work/projects/project_1/outputs/step_6/'
file_in_jsonData = 'validation_onsets.mat'
len_of_longest_sequence = 512

path_in_zp_data = '/worktmp2/hxkhkh/laptop/work/projects/project_1/outputs/step_2/validation/'                           
file_in_zp_data =  'zero_pad_val.mat' 

path_out = '/worktmp/hxkhkh/project_3/output/step_1/'
file_out = 'word_data.mat'

                         
jsonData = scipy.io.loadmat(path_in_jsonData + file_in_jsonData, 
                 variable_names=['wavfile_names','wavfile_caption','wavfile_duration','wavfile_words','wavfile_words_onsets','wavfile_words_offsets']) 

wavfile_names_json = jsonData['wavfile_names']
wavfile_names_json = [item for item in wavfile_names_json]
wavfile_caption = jsonData['wavfile_caption'] 
wavfile_duration = jsonData['wavfile_duration'][0] 


wavfile_phs = jsonData['wavfile_words'][0]
wavfile_onsets = jsonData['wavfile_words_onsets'][0]
wavfile_offsets = jsonData['wavfile_words_offsets'][0]

wavfile_onsets = [item[0]for item in wavfile_onsets] # ms
wavfile_offsets = [item[0] for item in wavfile_offsets] # ms 

temp = wavfile_phs
wavfile_phs = []
for utterance in temp:
    correct_utterance = []
    for phoneme in utterance:
        correct_ph  = phoneme.strip()
        correct_utterance.append(correct_ph)
    wavfile_phs.append(correct_utterance)
    
    
################################################################################ converting duration (s) to frame
win_hop_time = 100 # ms : 0.01 s 
wavfile_frame_lengths = [int (numpy.ceil(item * win_hop_time)) for item in wavfile_duration]
number_of_all_utterances = len(wavfile_frame_lengths)
index_long_utterances = [counter for counter, value in enumerate(wavfile_frame_lengths) if value > len_of_longest_sequence]

################################################################################  converting time (ms) to frame
    
wavfile_onsets = [numpy.ceil(utterance/10) for utterance in wavfile_onsets]        
wavfile_offsets = [numpy.ceil(utterance/10) for utterance in wavfile_offsets]

#here i selected the last item of words offset equal to onset + 2 frame
for counter,utterance in enumerate(wavfile_onsets):
    last_item = utterance[-1] + 2
    wavfile_offsets[counter] = numpy.append(wavfile_offsets[counter], last_item)
################################################################################ cutting long utterances
# for index_utt in index_long_utterances:
#     utterance = wavfile_phs[index_utt]
#     t_on = wavfile_onsets[index_utt]
#     t_off= wavfile_offsets[index_utt]
#     n_words = len(utterance)
#     for ind in n_words:
#         if t_on[ind]>len_of_longest_sequence or t_off[ind]>len_of_longest_sequence:
#             wavfile_phs
    
cut_indx = [len(utterance) for utterance in wavfile_onsets]#numpy.zeros(len(wavfile_phs_onsets))
for ut_number,utterance in enumerate(wavfile_onsets):
    for counter,item in enumerate(utterance):
        if item >= 512 or wavfile_offsets[ut_number][counter]>= 512:
            cut_indx[ut_number] = counter
            break

wavfile_onsets = [utterance[0:cut_indx[counter]] for counter,utterance in enumerate(wavfile_onsets)]        
wavfile_offsets = [utterance[0:cut_indx[counter]] for counter,utterance in enumerate(wavfile_offsets)]    
wavfile_phs  = [utterance[0:cut_indx[counter]] for counter,utterance in enumerate(wavfile_phs)]  

#.............................................. Zero Padding for audio features
pad_data = scipy.io.loadmat(path_in_zp_data + file_in_zp_data, 
                 variable_names=['zero_pad_len']) 
zero_pad_len = pad_data['zero_pad_len'][0]

#.............................................................................. adding zero padding

wavfile_onsets = [utterance + numpy.repeat(zero_pad_len[counter],len(utterance))for counter,utterance in enumerate(wavfile_onsets)]        
wavfile_offsets = [utterance + numpy.repeat(zero_pad_len[counter],len(utterance))for counter,utterance in enumerate(wavfile_offsets)]           
###############################################################################
                        # saving the results #
###############################################################################
                       
                        
scipy.io.savemat(path_out + file_out,
                     mdict={'wavfile_words':wavfile_phs,'wavfile_words_onsets':wavfile_onsets, 'wavfile_words_offsets':wavfile_offsets
                            ,'wavfile_caption':wavfile_caption})  




