     
import configuration as cfg


model_name = cfg.paths.model_name
modeldir = cfg.paths.modeldir
featuredir = cfg.paths.featuredir
visual_feature_name = cfg.paths.visual_feature_name
audio_feature_name = cfg.paths.audio_feature_name

use_pretrained = cfg.model_settings.use_pretrained
training_mode = cfg.model_settings.training_mode
evaluating_mode = cfg.model_settings.evaluating_mode
saving_mode = cfg.model_settings.saving_mode
save_best_recall = cfg.model_settings.save_best_recall
save_best_loss = cfg.model_settings.save_best_loss
find_recall = cfg.model_settings.find_recall
number_of_epochs = cfg.model_settings.number_of_epochs
   
n_caps_per_image = cfg.feature_settings.n_caps_per_image
set_of_train_files = cfg.feature_settings.set_of_train_files
set_of_validation_files = cfg.feature_settings.set_of_validation_files
length_sequence = cfg.feature_settings.length_sequence
Xshape = cfg.feature_settings.Xshape
Yshape = cfg.feature_settings.Yshape
  
###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)

import scipy.io
from  train_validate import initialize_model, save_model , train_model, evaluate_model

if model_name == 'CNNatt':
    from CNNatt import build_vgsmodel
elif model_name == 'CNN0':   
    from CNN0 import build_vgsmodel



if __name__ == '__main__':

    #  building the model 
    
    vgs_model, embedding_audio , embedding_visual =  build_vgsmodel (Xshape, Yshape , modeldir)

    # initializing the model 
    
    allepochs_valloss,allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator = initialize_model (vgs_model , modeldir ,use_pretrained )

    # training and validation loop  
    
    for epoch in range(number_of_epochs):
        
        print('.............................................  epoch ' + str(epoch ) )
        
        if training_mode: 
            epoch_trainloss = train_model (vgs_model, featuredir ,visual_feature_name , audio_feature_name, set_of_train_files, n_caps_per_image, length_sequence)
            allepochs_trainloss.append(epoch_trainloss)
        if evaluating_mode:                
            epoch_recall_av, epoch_recall_va , epoch_valloss =  evaluate_model(vgs_model, featuredir , modeldir , visual_feature_name , audio_feature_name,set_of_validation_files, find_recall, length_sequence) 
            allepochs_valloss.append(epoch_valloss) 
            if find_recall:                   
                all_avRecalls.append(epoch_recall_av)
                all_vaRecalls.append(epoch_recall_va)
            # saving the model
            if saving_mode:        
                save_model(vgs_model , modeldir, val_indicator,recall_indicator , save_best_loss , save_best_recall , epoch_valloss , epoch_recall_av, epoch_recall_va )
                scipy.io.savemat(modeldir + 'valtrainloss.mat', 
                         {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })
   
                
        vgs_model.save_weights('%smodel_weights_lastepoch.h5' % modeldir)        
        print ('............................................    validation_loss at this epoch =    ' + str(epoch_valloss))
        print ('............................................    recall_av at this epoch =    ' + str(epoch_recall_av))
        print ('............................................    recall_va at this epoch =    ' + str(epoch_recall_va))
        