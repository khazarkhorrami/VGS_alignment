
"""

This file reads data file names from SPEECHCOCO and MSCOCO data directories,
and does the following :
    1) function "processing_data_list" stores names of images and c
    orresponding spoken captions files in 'processed_data_list.mat' file.
    
    2) function "reading_word_onsets" reads onset of words and stotres them in
    'word_onsets.mat' file.

"""

###############################################################################

import os
import scipy,scipy.io
import numpy as np
import json

###############################################################################

images_path = "../../data/MSCOCO/val2014/val2014"
speech_path = "../../data/SPEECH-COCO/val2014/val2014/wav"
json_path = '../../data/SPEECH-COCO/val2014/val2014/json'

path_out = "../../testdata/0/"

###############################################################################

def processing_data_list():
    
    # preparing data list
    
    list_of_images = os.listdir(images_path)
    list_of_images.sort()
    
    wav_files = []
    image_ids = []
    counter = 0
    for folder in list_of_images:
        data_dir = os.path.join(images_path, folder)
        image_id =  data_dir [-16:-4]
        image_id = str(np.int(image_id)) 
        image_ids.append(image_id)
    #    print((image_id + '_'))
    #    print('..............')
    #    print(counter)
        counter += 1   
        for f_name in os.listdir(speech_path):
            if f_name.startswith(image_id+ '_'):
                wav_files.append(f_name)
                
    
    # processing data list
    
    list_of_wav_files = []
    list_of_wav_counts = []
     
    for image_counter,image_id in enumerate(image_ids):
    #    print(image_counter)
    #    print('...'+ image_id[0] + '...')
    
        wav_of_image = []
        wav_count_image = 0
          
        for wav_file in wav_files:
            if wav_file.startswith(image_id+ '_'):
                wav_of_image.append(wav_file)
                wav_count_image += 1
    
        list_of_wav_files.append(wav_of_image)
        list_of_wav_counts.append(wav_count_image)
    
    
    # saving the results for image/caption file names
    
    image_ids = np.array(image_ids, dtype=np.object)
    list_of_images = np.array(list_of_images, dtype=np.object)
    list_of_wav_files = np.array(list_of_wav_files, dtype=np.object)
    scipy.io.savemat(path_out + 'processed_data_list.mat',
                     {'list_of_wav_files':list_of_wav_files,'list_of_wav_counts':list_of_wav_counts,
                      'list_of_images':list_of_images,'image_id_all':image_ids})    
    return wav_files


# reading word onsets from json files
def reading_word_onsets (wav_files):
    wavfile_names = []
    wavfile_caption = []
    wavfile_duration = []
    
    wavfile_words = []
    wavfile_words_onsets = []
    
    for filename in wav_files:
        json_file = json_path + '/' + filename[0:-3] + 'json'
       
        words_onsets = []
        words = []
    
        with open(json_file, 'r') as f:
            
            datastore = json.load(f)
            
            wavfile_names.append(datastore['wavFilename'])                
            caption = datastore['synthesisedCaption']
            duration = datastore['duration']    
            timecode = datastore['timecode']
                    
            for item in timecode:
                #print(item)
                if item[1] == 'WORD':
                    words_onsets.append(item[0])
                    words.append(item[2])
                    
                    
        wavfile_caption.append(caption)
        wavfile_duration.append(duration)
        wavfile_words.append(words)
        wavfile_words_onsets.append(words_onsets) 
        
    #...................... offset times ..........................................
        
    wavfile_words_offsets = []
    for item in wavfile_words_onsets:
        words_offsets =  item[1:]
        wavfile_words_offsets.append(words_offsets)
     
    #...................... saving the results ....................................
          
    scipy.io.savemat(path_out + 'word_onsets.mat',
                 mdict={'wavfile_names':wavfile_names,'wavfile_caption':wavfile_caption,'wavfile_duration':wavfile_duration,
                        'wavfile_words':wavfile_words,'wavfile_words_onsets':wavfile_words_onsets, 'wavfile_words_offsets':wavfile_words_offsets}) 
    

    
    
if __name__ == '__main__':
    
    os.makedirs(path_out, exist_ok=1)
    # to obtain and save wave file names (speech files)
    wav_file_names = processing_data_list()
    # to extract word onsets for given wave files (speech files)
    reading_word_onsets (wav_file_names)
           