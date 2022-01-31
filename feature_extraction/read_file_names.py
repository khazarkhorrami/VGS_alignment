import json
import os
import numpy


def read_json_data(file_path , file_name):
    
    input_file = os.path.join (file_path , file_name)
    with open(input_file) as f:
        dict_data = json.load(f)
    data = dict_data['data']
    return data


def read_image_info(file_path , file_name_val , file_name_train):
        
    all_images = []
    data_val = read_json_data(file_path , file_name_val)
    for i in range(len(data_val)):
        all_images.append(data_val[i]['image'])
    data_train = read_json_data(file_path , file_name_train)
    for i in range(len(data_train)):
        all_images.append(data_train[i]['image'])
        
    return all_images



def read_caption_info(file_path , file_name_val , file_name_train):
        
    all_captions = []
    data_val = read_json_data(file_path , file_name_val)
    for i in range(len(data_val)):
        all_captions.append(data_val[i]['captions'])
    data_train = read_json_data(file_path , file_name_train)
    for i in range(len(data_train)):
        all_captions.append(data_train[i]['captions'])
        
    return all_captions


# splitting data to train/val 2014
def split_data (data_path):
    
    file_name_val = os.path.join( data_path , 'SpokenCOCO_val.json' ) 
    file_name_train = os.path.join( data_path , 'SpokenCOCO_train.json' )
    
    all_images = read_image_info(data_path , file_name_val , file_name_train)
    all_captions = read_caption_info(data_path , file_name_val , file_name_train)
    train_images = []
    val_images = []
    
    train_indices = []
    val_indices = []
        
    for counter_indice , item_image in enumerate(all_images):
        if item_image.startswith('val'):
            val_images.append(item_image)
            val_indices.append(counter_indice)
        else:
            train_images.append(item_image)
            train_indices.append(counter_indice)
    
    
    train_captions = [all_captions[item] for item in train_indices]
    val_captions = [ all_captions[item] for item in val_indices]
    
    return train_images,val_images,train_captions,val_captions


def data_chunker (input_list , chunk_len):  
    output_list = []          
    for item in numpy.arange(0, len(input_list),chunk_len):
        output_list.append(input_list [item:item+ chunk_len])         
    return output_list



def get_wav_files (input_captions):
    wav_files = []
    wav_files_count = []
    for counter, item in enumerate(input_captions): # 1:40 000
        number_of_captions = len(item)
        wav_files_count.append(number_of_captions)
        wav_pack = []
        for counter_subitem, subitem in enumerate(item):
            wav_pack.append(subitem['wav'])
        wav_files.append(wav_pack)
            
    return wav_files, wav_files_count


if __name__ == '__main__':

    data_path =  "../../data/SPOKEN-COCO/"
    train_images,val_images,train_captions,val_captions = split_data (data_path)
    wav_files, wav_files_count = get_wav_files (val_captions)
