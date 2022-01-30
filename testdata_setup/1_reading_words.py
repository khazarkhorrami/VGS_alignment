
"""

This file reads wavfile metadata (words and their time stamps) obtained ate step_0
and does followings
    1)  words data is extracted
    2)  words with t >512 are removed
    3)  all words are zeropadded

"""

###############################################################################

import numpy
import os
import scipy,scipy.io

###############################################################################

path_in_jsonData = "../../testdata/0/"
file_in_jsonData = 'word_onsets.mat'
len_of_longest_sequence = 512


path_out = "../../testdata/1/"
file_out = 'word_data.mat'


###############################################################################

def getting_word_data ():                        
    jsonData = scipy.io.loadmat( file_in_jsonData, 
                     variable_names=['wavfile_names','wavfile_caption','wavfile_duration','wavfile_words','wavfile_words_onsets','wavfile_words_offsets']) 
    
    wavfile_names_json = jsonData['wavfile_names']
    wavfile_names_json = [item for item in wavfile_names_json]
    wavfile_caption = jsonData['wavfile_caption'] 
    wavfile_duration = jsonData['wavfile_duration'][0] 
    
    
    wavfile_words = jsonData['wavfile_words'][0]
    wavfile_onsets = jsonData['wavfile_words_onsets'][0]
    wavfile_offsets = jsonData['wavfile_words_offsets'][0]
    
    wavfile_onsets = [item[0]for item in wavfile_onsets] # ms
    wavfile_offsets = [item[0] for item in wavfile_offsets] # ms 
    
    temp = wavfile_words
    wavfile_words = []
    for utterance in temp:
        correct_utterance = []
        for phoneme in utterance:
            correct_ph  = phoneme.strip()
            correct_utterance.append(correct_ph)
        wavfile_words.append(correct_utterance)
        
        
    # converting duration (s) to frame
    
    win_hop_time = 100 # ms : 0.01 s 
    wavfile_frame_lengths = [int (numpy.ceil(item * win_hop_time)) for item in wavfile_duration]
    #number_of_all_utterances = len(wavfile_frame_lengths)
    #index_long_utterances = [counter for counter, value in enumerate(wavfile_frame_lengths) if value > len_of_longest_sequence]
    zero_pad_len = [numpy.maximum(len_of_longest_sequence - item , 0 ) for item in wavfile_frame_lengths]
    
    # converting onset time (ms) to frame
        
    wavfile_onsets = [numpy.ceil(utterance/10) for utterance in wavfile_onsets]        
    wavfile_offsets = [numpy.ceil(utterance/10) for utterance in wavfile_offsets]
    
    # here i selected the last item of words offset equal to onset + 2 frame since
    # the information for the offset of last words are missing in Json file
    for counter,utterance in enumerate(wavfile_onsets):
        last_item = utterance[-1] + 2
        wavfile_offsets[counter] = numpy.append(wavfile_offsets[counter], last_item)
        
    # cutting long utterances
        
    cut_indx = [len(utterance) for utterance in wavfile_onsets]
    for ut_number,utterance in enumerate(wavfile_onsets):
        for counter,item in enumerate(utterance):
            if item >= 512 or wavfile_offsets[ut_number][counter]>= 512:
                cut_indx[ut_number] = counter
                break
    
    wavfile_onsets = [utterance[0:cut_indx[counter]] for counter,utterance in enumerate(wavfile_onsets)]        
    wavfile_offsets = [utterance[0:cut_indx[counter]] for counter,utterance in enumerate(wavfile_offsets)]    
    
       
    # adding zero padding
    
    wavfile_onsets = [utterance + numpy.repeat(zero_pad_len[counter],len(utterance))for counter,utterance in enumerate(wavfile_onsets)]        
    wavfile_offsets = [utterance + numpy.repeat(zero_pad_len[counter],len(utterance))for counter,utterance in enumerate(wavfile_offsets)] 
              
    # saving the results
                                                  
    scipy.io.savemat(path_out + file_out,
                         mdict={'wavfile_words':wavfile_words,'wavfile_words_onsets':wavfile_onsets, 'wavfile_words_offsets':wavfile_offsets
                                ,'wavfile_caption':wavfile_caption})  
    
    
    
if __name__ == '__main__': 
    
    os.makedirs(path_out, exist_ok=1)
    getting_word_data ()
