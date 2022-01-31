
########################## initial configuration ##############################

data_dir = '../../data/'
feature_dir =  '../../features/SPOKEN-COCO' 

processing_train_data = False
processing_val_data = True 
using_shuffled_indices = False
chunk_len=10000
dataset = 'SPOKEN-COCO' 

extracting_audio_features = False
extracting_visual_features = False

# Audio features parameters
number_of_mel_bands = 40     
window_len_in_ms = 0.025
window_hop_in_ms = 0.01
sr_target = 16000
# Visual features parameters
vgg_layer_name = 'block5_conv3'    

###############################################################################

import os
from extract_audio_features import  get_spokencoco_wavnames, save_chunked_logmels
from extract_visual_features import get_spokencoco_imagenames, save_chunked_vggs
from read_file_names import data_chunker


if __name__ == '__main__':
      
    if dataset == 'SPOKEN-COCO':
        
        os.makedirs(feature_dir, exist_ok=1)
        # reading image file names
        all_image_files, data_path , output_path = get_spokencoco_imagenames (data_dir , feature_dir , processing_train_data , processing_val_data , using_shuffled_indices)
        # reading wav file names
        all_wav_files, all_wav_files_counts, data_path , output_path = get_spokencoco_wavnames (data_dir, feature_dir , processing_train_data , processing_val_data , using_shuffled_indices) 
        

        if extracting_audio_features:          
            # Extracting and saving audio features (one chunk at a time)       
            chuncked_data = data_chunker (all_image_files , chunk_len)    
            for counter_chunk, data_chunk in  enumerate(chuncked_data):
                output_name =  output_path + str(counter_chunk) 
                save_chunked_logmels (data_chunk, data_path, output_name ,vgg_layer_name)
                
        if extracting_visual_features :                 
            # Extracting and saving visual features (one chunk at a time)       
            chuncked_data = data_chunker (all_image_files , chunk_len)    
            for counter_chunk, data_chunk in  enumerate(chuncked_data):
                output_name =  output_path + str(counter_chunk) 
                save_chunked_vggs (data_chunk, data_path, output_name ,vgg_layer_name)
