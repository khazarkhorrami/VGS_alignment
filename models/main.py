import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config) 

import scipy.io
from CNNatt import build_vgsmodel
from  train_validate import initialize_model, train_model, evaluate_model

if __name__ == '__main__':
    
    ################################################################################  initial configuration  
    modeldir = '/'
    featuredir = '/' 
    visual_feature_name = 'vggb5conv3_'
    audio_feature_name = 'logmel_'
    
    use_pretrained = False
    save_best_recall = False
    save_best_loss = True
    find_recall = True
    number_of_epochs = 200
   
    n_of_caps_per_image = 5
    set_of_train_files = ['train_ch0', 'train_ch1', 'train_ch2', 'train_ch3', 'train_ch4', 'train_ch5', 'train_ch6', 'train_ch7']
    set_of_validation_files = ['train_ch8']
    
    length_sequence = 512
    X_shape = (length_sequence,40)
    Y_shape = (14,14,512)
    vgs_model, embedding_audio , embedding_visual =  build_vgsmodel (X_shape, Y_shape , modeldir)

    ############################################################################### initializing model 
    
    allepochs_valloss,allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator = initialize_model (vgs_model , modeldir ,use_pretrained )

    ################################################################################   training and validation loop    
    for epoch in range(number_of_epochs):
        print('......................... epoch .............................' + str(epoch ) )
        #train
        epoch_trainloss = train_model (vgs_model, featuredir ,visual_feature_name , audio_feature_name, set_of_train_files, n_of_caps_per_image, length_sequence)
        allepochs_trainloss.append(epoch_trainloss)
        #evaluate                 
        epoch_recall_av, epoch_recall_va , epoch_valloss =  evaluate_model(vgs_model, featuredir ,  visual_feature_name , audio_feature_name,set_of_validation_files, find_recall, length_sequence) 
        allepochs_valloss.append(epoch_valloss) 
        if find_recall:                   
            all_avRecalls.append(epoch_recall_av)
            all_vaRecalls.append(epoch_recall_va)
        # save  
        if save_best_loss:
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                weights = vgs_model.get_weights()
                vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % modeldir)
                
        if save_best_recall:
            epoch_recall = ( epoch_recall_av + epoch_recall_va ) / 2
            if epoch_recall >= recall_indicator: 
                recall_indicator = epoch_recall
                weights = vgs_model.get_weights()
                vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % modeldir)
                
        vgs_model.save_weights('%smodel_weights_lastepoch.h5' % modeldir)        
        print ('............................................    validation_loss at this epoch =    ' + str(epoch_valloss))
        print ('............................................    recall_av at this epoch =    ' + str(epoch_recall_av))
        print ('............................................    recall_va at this epoch =    ' + str(epoch_recall_va))
        scipy.io.savemat(modeldir + 'valtrainloss.mat', 
                     {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })
    # .............................................................................
