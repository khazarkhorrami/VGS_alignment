import pickle
import numpy 

from extract_audio_features import data_chunker, split_data

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 

import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model


def calculate_vgg_features(imgs_chunk, data_path,layer_name = 'block5_conv3'):
    model = VGG16()  
    model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    #print(model.summary())
    vgg_out_all = []    
    for counter_images, image_name in enumerate(imgs_chunk): 
        print(counter_images)
                
        image_original = cv2.imread(data_path + image_name)
        image_resized = cv2.resize(image_original,(224,224))
        image_input_vgg = preprocess_input(image_resized.reshape((1, 224, 224, 3)))
        vgg_out = model.predict(image_input_vgg)
        vgg_out_reshaped = numpy.squeeze(vgg_out)
        vgg_out_all.append(vgg_out_reshaped)
    return vgg_out_all

    
def save_chunked_vggs (imgs_chunk, data_path, out_file_name ,layer_name = 'block5_conv3' ):
    image_features = calculate_vgg_features(imgs_chunk, data_path,layer_name)
    save_pickle (image_features, out_file_name )

    
def get_spokencoco_imagenames ( process_train_data , process_val_data , use_shuffle_indices)   :        
        json_path =  "../../data/SPOKEN-COCO/" 
        data_path = "../../data/MSCOCO/"
        feature_path = "../../features/SPOKEN-COCO/"              
        train_imgs,val_imgs,train_caps,val_caps = split_data (json_path)   
          
        if process_train_data:
            input_images = train_imgs
            out_file_name = 'vggb5conv3_train_ch'
        elif process_val_data:
            input_images = val_imgs
            out_file_name = 'vggb5conv3_val_ch'           
        vgg_path = feature_path + out_file_name
        
        if use_shuffle_indices:
            inds_shuffled = numpy.load(feature_path + 'shuffle_indices.npy')                  
            input_images_shuffled = [input_images[item] for item in inds_shuffled]
            del input_images
            input_images = input_images_shuffled
              
        return input_images, data_path , vgg_path  



def save_pickle (input_list, filename):
   outfile = open(filename,'wb')
   pickle.dump(input_list ,outfile , protocol=pickle.HIGHEST_PROTOCOL)
   outfile.close()   


    

if __name__ == '__main__':
           
       
    processing_train_data = False
    processing_val_data = True 
    using_shuffled_indices = False
    chunk_len=10000
    dataset = 'SPOKEN-COCO'
    
    vgg_layer_name = 'block5_conv3'
       
    if dataset == 'SPOKEN-COCO':
        # reading file names
        all_image_files, data_path , output_path = get_spokencoco_imagenames ( processing_train_data , processing_val_data , using_shuffled_indices)          
        # Extracting and saving audio features (one chunk at a time)       
        chuncked_data = data_chunker (all_image_files , chunk_len)    
        for counter_chunk, data_chunk in  enumerate(chuncked_data):
            output_name =  output_path + str(counter_chunk) 
            save_chunked_vggs (data_chunk, data_path, output_name ,vgg_layer_name)
    



