
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config) 

import numpy
import pickle
import numpy
import scipy.io
import scipy.spatial as ss
# initial configuration
length_sequence = 512 
number_of_epochs = 200
name_of_train_chunks = ['train_ch0', 'train_ch1', 'train_ch2', 'train_ch3', 'train_ch4', 'train_ch5', 'train_ch6', 'train_ch7']
name_of_validation_chunks = ['train_ch8']

modeldir = ''
featuredir = ''
use_pretrained = False


############################################################################### 
                        
###############################################################################

from keras import backend as K
import keras

from keras.layers import Lambda

from keras.models import Model
from keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import  MaxPooling1D,  Conv1D,Conv2D
from keras.layers import Softmax, Permute, AveragePooling1D, Concatenate
dropout_size = 0.3
#output_visual_size = numpy.shape(Y)[1]

activation_C='relu'
activation_R='tanh'

X_shape = (512,40)
Y_shape = (14,14,512)


#.............................................................................. Audio Network
audio_sequence = Input(shape=X_shape)


forward1 = Conv1D(128,5,padding="same",activation=activation_C,name = 'conv1')(audio_sequence)
dr1 = Dropout(dropout_size)(forward1)
bn1 = BatchNormalization(axis=-1)(dr1)


forward2 = Conv1D(256,11,padding="same",activation=activation_C,name = 'conv2')(bn1)
dr2 = Dropout(dropout_size)(forward2)
bn2 = BatchNormalization(axis=-1)(dr2)
 
pool2 = MaxPooling1D(3,strides = 2, padding='same')(bn2)


forward3 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv3')(pool2)
dr3 = Dropout(dropout_size)(forward3)
bn3 = BatchNormalization(axis=-1)(dr3) 

pool3 = MaxPooling1D(3,strides = 2,padding='same')(bn3)


forward4 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv4')(pool3)
dr4 = Dropout(dropout_size)(forward4)
bn4 = BatchNormalization(axis=-1)(dr4) 
pool4 = MaxPooling1D(3,strides = 2,padding='same')(bn4)


forward5 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv5')(pool4)
dr5 = Dropout(dropout_size)(forward5)
bn5 = BatchNormalization(axis=-1,name='audio_branch')(dr5) 
out_audio_channel = bn5

#.............................................................................. Visual Network

visual_sequence = Input(shape=Y_shape)
visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)

forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)

out_visual_channel = Reshape([196,512],name='reshape_visual')(bn_visual)

############################################################################### combining audio and visual channels



out_audio = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
out_visual = Dense(512,activation='linear',name='dense_visual')(out_visual_channel) 


out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(out_audio)
out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(out_visual)

#.............................................................................. 

I = out_visual
A = out_audio

mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_matchmap')

def custom_layer(tensor):
    x= tensor   
    sisa = K.mean( (K.mean(x, axis=1)), axis=-1)
    sisa = Reshape([1],name='reshape_misa')(sisa)
    return sisa

lambda_layer = Lambda(custom_layer, name="final_layer")(mapIA)

model = Model(inputs=[visual_sequence, audio_sequence], outputs = lambda_layer )


############################################################################### for using previously trained model
allepochs_valloss = []
allepochs_trainloss = []
all_avRecalls = []
all_vaRecalls = []
recall_indicator = 0
val_indicator = 1000


if use_pretrained:
    model.load_weights(modeldir + 'model_weights.h5')

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

###############################################################################
                      # Custom loss function #
###############################################################################                      
def mycustomloss(y_true,y_pred):
    
    margin = 0.1

    Sp = y_pred[0::3]

    Si = y_pred[1::3]

    Sc = y_pred[2::3]
    
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0)                      
                                            
###############################################################################

lossfunction = mycustomloss

model.compile(loss=lossfunction, optimizer= keras.optimizers.Adam(lr=1e-04))
print(model.summary())
model.save('%smodel' % modeldir)

###############################################################################


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


###############################################################################
def calculate_recallat10( embedding_1,embedding_2, sampling_times,  number_of_all_audios, pool):   
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


def model_training ():
    for chunk_counter, chunk_name in enumerate(name_of_train_chunks): 
        #...................................................................... Y train
        filename = featuredir + 'vggb5conv3_' + chunk_name 
        print(filename)
        Ydata = loadYdata(filename)
        n_samples = len(Ydata)
         
        for i_caption in range(5):            
            #.................................................................. X train
            filename = featuredir + 'logmel_'+ chunk_name
            Xdata = loadXdata(filename , length_sequence , i_caption)
            
            n_samples = Ydata.shape[0]
            orderX,orderY = randOrder(n_samples)
            bin_triplet = numpy.array(make_bin_target(n_samples))
             
            print('......................... epoch .............................' + str(epoch ) )
            print('......................... chunk train .................................' + str(chunk_counter))
            print('......................... audio_class .................................' + str(i_caption)) 
            
            history = model.fit([Ydata[orderY], Xdata[orderX] ] , bin_triplet, shuffle=False, epochs=1,batch_size=120)   
            del Xdata
        del Ydata      
    final_trainloss = history.history['loss'][0]
    return final_trainloss


def model_evaluating ():
    epoch_cum_val = 0
    epoch_cum_recall_av = 0
    epoch_cum_recall_va = 0
     #shuffle([q for q in range(n_chunks)])
    for chunk_counter, chunk_name in enumerate(name_of_validation_chunks): 
        print('......................... chunk validation .................................' + str(chunk_counter))
        #.................................................................. Y
        filename = featuredir + 'vggb5conv3_' + chunk_name
        Ydata = loadYdata(filename)
        n_samples = len(Ydata)
        visual_embeddings = new_visual_model.predict(Ydata)
        visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1)      
        #.................................................................. X
        i_caption = numpy.random.randint(0, 5)
        filename = featuredir + 'logmel_'+ chunk_name
        Xdata = loadXdata(filename , length_sequence , i_caption)
        audio_embeddings = new_audio_model.predict(Xdata) 
        audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)              
        orderX,orderY = randOrder(n_samples)
        bin_triplet = numpy.array(make_bin_target(n_samples)) 
        val_chunk = model.evaluate( [Ydata[orderY],Xdata[orderX] ],bin_triplet,batch_size=120)    
        epoch_cum_val += val_chunk                  
        
        ########### calculating Recall@10                    
        poolsize =  1000
        recall_av_vec = calculate_recallat10( audio_embeddings_mean,visual_embeddings_mean, 100,  n_samples , poolsize ) 
        recall_va_vec = calculate_recallat10( audio_embeddings_mean,visual_embeddings_mean, 100,  n_samples , poolsize ) 
        recall10_av = numpy.mean(recall_av_vec)/(poolsize)
        recall10_va = numpy.mean(recall_va_vec)/(poolsize)
        epoch_cum_recall_av += recall10_av
        epoch_cum_recall_va += recall10_va               
        del Xdata
        del audio_embeddings
        del Ydata           
        del visual_embeddings
              
    final_recall_av = epoch_cum_recall_av / (chunk_counter + 1 ) 
    final_recall_va = epoch_cum_recall_va / (chunk_counter + 1 ) 
    final_valloss = epoch_cum_val/ (chunk_counter + 1 )  
    
    return final_recall_av, final_recall_va , final_valloss
###############################################################################
                        # defining the new Audio model #
###############################################################################

new_audio_model = Model(inputs=audio_sequence,outputs=out_audio)

for n in range(13):
    new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
   
new_audio_model.layers[13].set_weights(model.layers[14].get_weights())
new_audio_model.layers[14].set_weights(model.layers[16].get_weights())
new_audio_model.layers[15].set_weights(model.layers[18].get_weights()) 
new_audio_model.layers[16].set_weights(model.layers[20].get_weights()) 
new_audio_model.layers[17].set_weights(model.layers[22].get_weights()) 
new_audio_model.layers[18].set_weights(model.layers[24].get_weights()) 
new_audio_model.layers[19].set_weights(model.layers[26].get_weights())
new_audio_model.layers[20].set_weights(model.layers[28].get_weights()) # out audio

print(new_audio_model.summary())
 

###############################################################################
                        # defining the new Visual model #
###############################################################################
new_visual_model = Model(inputs=visual_sequence,outputs=out_visual)
 
new_visual_model.layers[0].set_weights(model.layers[13].get_weights()) # input layer
new_visual_model.layers[1].set_weights(model.layers[15].get_weights())
new_visual_model.layers[2].set_weights(model.layers[17].get_weights()) 
new_visual_model.layers[3].set_weights(model.layers[19].get_weights()) 
new_visual_model.layers[4].set_weights(model.layers[21].get_weights()) 
new_visual_model.layers[5].set_weights(model.layers[23].get_weights()) 
new_visual_model.layers[6].set_weights(model.layers[25].get_weights())
new_visual_model.layers[7].set_weights(model.layers[27].get_weights())# out visual

print(new_visual_model.summary())


###############################################################################
                        # INITIAL VALIDATION #
###############################################################################
                       
initial_recall_av, initial_recall_va , initial_valloss =  model_evaluating()      

################################################################################  TRAINING LOOP  

for epoch in range(number_of_epochs):
    print('......................... epoch .............................' + str(epoch ) )
    #train
    epoch_trainloss = model_training ()
    allepochs_trainloss.append(epoch_trainloss)
    #evaluate                 
    epoch_recall_av, epoch_recall_va , epoch_valloss =  model_evaluating()                         
    all_avRecalls.append(epoch_recall_av)
    all_vaRecalls.append(epoch_recall_va)
    allepochs_valloss.append(epoch_valloss) 
    
    if epoch_valloss <= val_indicator: 
            val_indicator = epoch_valloss
            weights = model.get_weights()
            model.set_weights(weights)
            model.save_weights('%smodel_weights.h5' % modeldir)
    
    model.save_weights('%smodel_weights_lastepoch.h5' % modeldir)        
    print ('............................................    validation_loss at this epoch =    ' + str(epoch_valloss))
    print ('............................................    recall_av at this epoch =    ' + str(epoch_recall_av))
    print ('............................................    recall_va at this epoch =    ' + str(epoch_recall_va))
    scipy.io.savemat(modeldir + 'valtrainloss.mat', 
                 {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })
# .............................................................................

