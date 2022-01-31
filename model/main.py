###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)


###############################################################################

     
import config as cfg

model_name = cfg.paths['model_name']
model_subname = cfg.paths['model_subname']
model_dir = cfg.paths['modeldir']
feature_dir = cfg.paths['featuredir']
visual_feature_name = cfg.paths['visual_feature_name']
audio_feature_name = cfg.paths['audio_feature_name']

use_pretrained = cfg.model_settings['use_pretrained']
training_mode = cfg.model_settings['training_mode']
evaluating_mode = cfg.model_settings['evaluating_mode']
saveing_mode = cfg.model_settings['save_model']
save_best_recall = cfg.model_settings['save_best_recall']
save_best_loss = cfg.model_settings['save_best_loss']
find_recall = cfg.model_settings['find_recall']
number_of_epochs = cfg.model_settings['number_of_epochs']
   
number_of_captions_per_image = cfg.feature_settings['n_caps_per_image']
training_chunks = cfg.feature_settings['set_of_train_files']
validation_chunks = cfg.feature_settings['set_of_validation_files']
length_sequence = cfg.feature_settings['length_sequence']
Xshape = cfg.feature_settings['Xshape']
Yshape = cfg.feature_settings['Yshape']
input_dim=[Xshape,Yshape] 
 

###############################################################################

from train import train_validate

feature_name = [ audio_feature_name, visual_feature_name ]
training_params = [ number_of_captions_per_image,  length_sequence ]
validation_params = [ find_recall, save_best_recall ]
action_parameters = [ number_of_epochs , training_mode, evaluating_mode, saveing_mode ]

run_training_and_validation = train_validate (model_name, model_subname, input_dim, model_dir, feature_dir, feature_name, training_chunks, validation_chunks, training_params, validation_params, action_parameters, use_pretrained )    
run_training_and_validation()  
