     
###################### initial configuration  #################################

model_name = 'CNNatt'
modeldir = '/models/' + model_name + '/'
featuredir = '/features/coco/SPOKEN-COCO/train/' 
visual_feature_name = 'vggb5conv3_'
audio_feature_name = 'logmel_'

use_pretrained = True
training_mode = False
evaluating_mode = True
saving_mode = True
save_best_recall = False
save_best_loss = True
find_recall = True

number_of_epochs = 200
   
n_caps_per_image = 5
set_of_train_files = ['train_ch0', 'train_ch1', 'train_ch2', 'train_ch3', 'train_ch4', 'train_ch5', 'train_ch6', 'train_ch7']
set_of_validation_files = ['train_ch8']

length_sequence = 512
Xshape = (length_sequence,40)
Yshape = (14,14,512)
    
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
        