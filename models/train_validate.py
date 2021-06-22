import numpy
import scipy.io
import scipy.spatial as ss

from prepare_train_validation_data import prepare_triplet_data
from CNNatt import build_vgsmodel, build_audiomodel, build_visualmodel

def calculate_recallat10( embedding_1,embedding_2, sampling_times, number_of_all_audios, pool):   
    recall_all = []
    recallat = 10  
    for trial_number in range(sampling_times):      
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=pool)       
        vec_1 = [embedding_1[item] for item in data_ind]
        vec_2 = [embedding_2[item] for item in data_ind]           
        distance_utterance = ss.distance.cdist( vec_1 , vec_2 ,  'cosine') # 1-cosine
       
        r = 0
        for n in range(pool):
            ind_1 = n #random.randrange(0,number_of_audios)                   
            distance_utterance_n = distance_utterance[n]            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_1)*1)   
        recall_all.append(r)
        del distance_utterance  
        
    return recall_all


def train_model (vgs_model, featuredir,  visual_feature_name , audio_feature_name, set_of_train_chunks, number_of_captions_per_image , length_sequence):
    set_of_input_chunks = set_of_train_chunks
    for chunk_counter, chunk_name in enumerate(set_of_input_chunks): 
        for i_caption in range(number_of_captions_per_image):
            Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet_data (featuredir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , length_sequence)          
            print('.......... train chunk ..........' + str(chunk_counter))
            print('.......... audio caption ........' + str(i_caption))            
            history = vgs_model.fit([Ydata_triplet, Xdata_triplet ], bin_triplet, shuffle=False, epochs=1,batch_size=120)   
            del Xdata_triplet
        del Ydata_triplet      
    final_trainloss = history.history['loss'][0]
    return final_trainloss


def evaluate_model (vgs_model, featuredir,  visual_feature_name , audio_feature_name, set_of_validation_chunks , find_recall , length_sequence) :
    epoch_cum_val = 0
    epoch_cum_recall_av = 0
    epoch_cum_recall_va = 0
    i_caption = numpy.random.randint(0, 5)
    set_of_input_chunks = set_of_validation_chunks 
    for chunk_counter, chunk_name in enumerate(set_of_input_chunks):
        print('.......... validation chunk ..........' + str(chunk_counter))
        Ydata_triplet, Xdata_triplet, bin_triplet =  prepare_triplet_data (featuredir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , length_sequence)
        val_chunk = vgs_model.evaluate( [Ydata_triplet,Xdata_triplet ],bin_triplet,batch_size=120)    
        epoch_cum_val += val_chunk                  
        #..................................................................... Recall
        if find_recall:
            Ydata = Ydata_triplet[0::3]
            Xdata = Xdata_triplet[0::3]
            
            audio_model = build_audiomodel (X_shape, Y_shape , modeldir)
            visual_model = build_visualmodel (X_shape, Y_shape , modeldir)
            
            number_of_samples = len(Ydata)
            visual_embeddings = visual_model.predict(Ydata)
            visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1)  
            
            audio_embeddings = audio_model.predict(Xdata) 
            audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)
            ########### calculating Recall@10                    
            poolsize =  1000
            number_of_trials = 100
            recall_av_vec = calculate_recallat10( audio_embeddings_mean,visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize )          
            recall_va_vec = calculate_recallat10( audio_embeddings_mean,visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize ) 
            recall10_av = numpy.mean(recall_av_vec)/(poolsize)
            recall10_va = numpy.mean(recall_va_vec)/(poolsize)
            epoch_cum_recall_av += recall10_av
            epoch_cum_recall_va += recall10_va               
            del Xdata, audio_embeddings
            del Ydata, visual_embeddings            
        del Xdata_triplet,Ydata_triplet
        
    final_recall_av = epoch_cum_recall_av / (chunk_counter + 1 ) 
    final_recall_va = epoch_cum_recall_va / (chunk_counter + 1 ) 
    final_valloss = epoch_cum_val/ (chunk_counter + 1 )  
    
    return final_recall_av, final_recall_va , final_valloss

def initialize_model (vgs_model , modeldir ,use_pretrained ) :
    if use_pretrained:
        vgs_model.load_weights(modeldir + 'model_weights.h5')
    
        data = scipy.io.loadmat(modeldir + 'valtrainloss.mat' , variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
        allepochs_valloss = data['allepochs_valloss'][0]
        allepochs_trainloss = data['allepochs_trainloss'][0]
        all_avRecalls = data['all_avRecalls'][0]
        all_vaRecalls = data['all_vaRecalls'][0]
        
        allepochs_valloss = numpy.ndarray.tolist(allepochs_valloss)
        allepochs_trainloss = numpy.ndarray.tolist(allepochs_trainloss)
        all_avRecalls = numpy.ndarray.tolist(all_avRecalls)
        all_vaRecalls = numpy.ndarray.tolist(all_vaRecalls)
        recall_indicator = numpy.max(allepochs_valloss)
        val_indicator = numpy.min(allepochs_valloss)
    else:
        allepochs_valloss = []
        allepochs_trainloss = []
        all_avRecalls = []
        all_vaRecalls = []
        recall_indicator = 0
        val_indicator = 1000
        
    return allepochs_valloss,allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator
    
if __name__ == '__main__':
    
    modeldir = ''
    featuredir = ''
    visual_feature_name = ''
    audio_feature_name = ''
    
    use_pretrained = False
    save_best_recall = False
    save_best_loss = True
    find_recall = True
    number_of_epochs = 200
   
    n_of_caps_per_image = 5
    set_of_train_chunks = []
    set_of_validation_chunks = []
    
    length_sequence = 512
    X_shape = (length_sequence,40)
    Y_shape = (14,14,512)
    vgs_model, embedding_audio , embedding_visual =  build_vgsmodel (X_shape, Y_shape , modeldir)
    
    ############################################################################### initializing model
    
    allepochs_valloss,allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator = initialize_model (vgs_model , modeldir ,use_pretrained )
    
    ################################################################################  training and validation loop   
    
    for epoch in range(number_of_epochs):
        print('......................... epoch .............................' + str(epoch ) )
        #train
        epoch_trainloss = train_model (vgs_model,featuredir, visual_feature_name , audio_feature_name, set_of_train_chunks, n_of_caps_per_image, length_sequence)       
        #evaluate                 
        epoch_recall_av, epoch_recall_va , epoch_valloss =  evaluate_model(vgs_model, featuredir ,  visual_feature_name , audio_feature_name, set_of_validation_chunks, find_recall, length_sequence) 
