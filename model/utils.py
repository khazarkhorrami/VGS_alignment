import numpy
import pickle

import scipy.spatial as ss
from tensorflow.keras import backend as K


def loadXdata (filename, len_of_longest_sequence , i_cap):
    infile = open(filename ,'rb')
    logmel = pickle.load(infile)
    infile.close()
    logmel_i = [item[i_cap] for item in logmel]
    Xdata = preparX (logmel_i, len_of_longest_sequence ,)
    del logmel
    return Xdata
    
def loadYdata (filename):
    infile = open(filename ,'rb')
    vgg = pickle.load(infile)
    infile.close()
    Ydata = preparY(vgg)
    del vgg 
    return Ydata

    
def preparX (dict_logmel, len_of_longest_sequence):
    number_of_audios = numpy.shape(dict_logmel)[0]
    number_of_audio_features = numpy.shape(dict_logmel[0])[1]
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       logmel_item = dict_logmel[k]
       logmel_item = logmel_item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(logmel_item):, :] = logmel_item
    return X


def preparY (dict_vgg):
    Y = numpy.array(dict_vgg)    
    return Y


def make_bin_target (n_sample):
    target = []
    for group_number in range(n_sample):    
        target.append(1)
        target.append(0)
        target.append(0)
        
    return target


def randOrder(n_t):
    random_order = numpy.random.permutation(int(n_t))
    random_order_X = numpy.random.permutation(int(n_t))
    random_order_Y = numpy.random.permutation(int(n_t))
    
    data_orderX = []
    data_orderY = []     
    for group_number in random_order:
        
        data_orderX.append(group_number)
        data_orderY.append(group_number)
        
        data_orderX.append(group_number)
        data_orderY.append(random_order_Y[group_number])
        
        data_orderX.append(random_order_X[group_number])
        data_orderY.append(group_number)
        
    return data_orderX,data_orderY


def prepare_triplet_data (featuredir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , length_sequence):
     #...................................................................... Y 
    filename = featuredir + visual_feature_name + chunk_name 
    Ydata = loadYdata(filename)
    n_samples = len(Ydata)
   
    #.................................................................. X
    filename = featuredir + audio_feature_name + chunk_name
    Xdata = loadXdata(filename , length_sequence , i_caption) 
    orderX,orderY = randOrder(n_samples)
    bin_triplet = numpy.array(make_bin_target(n_samples)) 
    Ydata_triplet = Ydata[orderY]
    Xdata_triplet = Xdata[orderX]
    return Ydata_triplet, Xdata_triplet, bin_triplet   
        

       
def triplet_loss(y_true,y_pred):    
    margin = 0.1
    Sp = y_pred[0::3]
    Si = y_pred[1::3]
    Sc = y_pred[2::3]      
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0) 


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
            
