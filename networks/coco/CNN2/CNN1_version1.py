#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:16:44 2019

@author: khorrami
"""

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config) 


#machine_path = '/worktmp/hxkhkh/'
#machine_path = '/worktmp/khorrami/'
machine_path = '/scratch/hxkhkh/'

modeldir = machine_path +  'project2/current/outputs/models/coco/CNN1/v1/'
datadir = machine_path +  'features/coco/old/threechunks/'

###############################################################################
# The main code starts here
###############################################################################
import pickle
import numpy
import scipy.io
import scipy.spatial as ss
############################################################################### 
                        # CNN Modelwith ATTENTION #
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

X_shape = (512,24)
Y_shape = (14,14,512)

#.............................................................................. Audio Network
#.............................................................................. Audio Network
audio_sequence = Input(shape=X_shape)

# layer 1

forward1 = Conv1D(64,1,padding="same",activation=activation_C,name = 'conv1')(audio_sequence)
dr1 = Dropout(dropout_size)(forward1)
bn1 = BatchNormalization(axis=2)(dr1)

# layer 2
forward2 = Conv1D(128,11,padding="same",activation=activation_C,name = 'conv2')(bn1)
dr2 = Dropout(dropout_size)(forward2)
bn2 = BatchNormalization(axis=2)(dr2) 
pool2 = MaxPooling1D(3,strides = 2, padding='same')(bn2)

# layer 3
forward3 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv3')(pool2)
dr3 = Dropout(dropout_size)(forward3)
bn3 = BatchNormalization(axis=2)(dr3) 
pool3 = MaxPooling1D(3,strides = 2,padding='same')(bn3)

# layer 4
forward4 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv4')(pool3)
dr4 = Dropout(dropout_size)(forward4)
bn4 = BatchNormalization(axis=2)(dr4) 
pool4 = MaxPooling1D(3,strides = 2,padding='same')(bn4)

# layer 5
forward5 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv5')(pool4)
dr5 = Dropout(dropout_size)(forward5)
bn5 = BatchNormalization(axis=2,name='audio_branch')(dr5) 
out_audio_channel = bn5

#.............................................................................. Visual Network

visual_sequence = Input(shape=Y_shape)
forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same",name = 'conv_visual')(visual_sequence)
dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
bn_visual = BatchNormalization(axis=-1)(dr_visual)

out_visual_channel = Reshape([196,512],name='reshape_visual')(bn_visual)


###############################################################################  Attention I for query Audio
# which part of image gets more attention based on audio query.

keyImage = out_visual_channel
valueImage = out_visual_channel
queryAudio = out_audio_channel

scoreI = keras.layers.dot([queryAudio,keyImage], normalize=False, axes=-1,name='scoreI')
weightI = Dense(196,activation='sigmoid')(scoreI)

valueImage = Permute((2,1))(valueImage)
attentionI = keras.layers.dot([weightI, valueImage], normalize=False, axes=-1,name='attentionI')

poolAttI = AveragePooling1D(512, padding='same')(attentionI)
poolqueryAudio = AveragePooling1D(512, padding='same')(queryAudio)

outAttAudio = Concatenate(axis=-1)([poolAttI, poolqueryAudio])
outAttAudio = Reshape([1024],name='reshape_out_attAudio')(outAttAudio)

###############################################################################  Attention A  for query Image
# which part of Audio gets more attention based on image query.
keyAudio = out_audio_channel
valueAudio = out_audio_channel
queryImage = out_visual_channel

scoreA = keras.layers.dot([queryImage,keyAudio], normalize=False, axes=-1,name='scoreA')
weightA = Dense(64,activation='sigmoid')(scoreA)

valueAudio = Permute((2,1))(valueAudio)
attentionA = keras.layers.dot([weightA, valueAudio], normalize=False, axes=-1,name='attentionA')

poolAttA = AveragePooling1D(512, padding='same')(attentionA)
poolqueryImage = AveragePooling1D(512, padding='same')(queryImage)

outAttImage = Concatenate(axis=-1)([poolAttA, poolqueryImage])
outAttImage = Reshape([1024],name='reshape_out_attImage') (outAttImage)

############################################################################### combining audio and visual channels
#out_audio = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
#out_visual = Dense(512,activation='linear',name='dense_visual')(out_visual_channel)

out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(outAttAudio)
out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(outAttImage)
 
I = out_visual
A = out_audio

mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_final')

model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA )

############################################################################### for using previously trained model
#model.load_weights(modeldir + 'model_weights.h5')
###############################################################################
                      # Custom loss function #
###############################################################################
                      
def mycustomloss(y_true,y_pred):
    
    margin = 0.2 

    Sp = y_pred[0::3]

    Si = y_pred[1::3]

    Sc = y_pred[2::3]
    
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0)                      
                      
                      
###############################################################################

lossfunction = mycustomloss#'mean_squared_error'

#myoptimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9)
model.compile(loss=lossfunction,optimizer= keras.optimizers.Adam(lr=1e-04))
model.save('%smodel' % modeldir)

print(model.summary())

############################################################################### Binary target
def make_bin_target (n_sample):
    target = []

    for group_number in range(n_sample):    
        target.append(1)
        target.append(0)
        target.append(0)
        
    return target

   
###############################################################################
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

###############################################################################
def calculate_recallat10( audio_embedd,visual_embedd, sampling_times,  number_of_all_audios, pool):
    
    recall_all = []
    recallat = 10
    
    for trial_number in range(sampling_times):
        
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=pool)
        
        a_embedd = [audio_embedd[item] for item in data_ind]
        v_embedd = [visual_embedd[item] for item in data_ind]
            
        distance_utterance = ss.distance.cdist( a_embedd , v_embedd ,  'cosine') # 1-cosine
        
        r = 0
        for n in range(pool):
            #print('###############################################################.....' + str(n))
            ind_audio = n #random.randrange(0,number_of_audios)
                    
            distance_utterance_n = distance_utterance[n] 
            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_audio)*1)
    
        recall_all.append(r)
        del distance_utterance
        
    return recall_all
###############################################################################
                        # defining the new Audio model #
###############################################################################

new_audio_model = Model(inputs=audio_sequence,outputs=out_audio_channel)

for n in range(14):
    new_audio_model.layers[n].set_weights(model.layers[n].get_weights())
   
new_audio_model.layers[14].set_weights(model.layers[15].get_weights())
new_audio_model.layers[15].set_weights(model.layers[17].get_weights())
new_audio_model.layers[16].set_weights(model.layers[19].get_weights()) 
new_audio_model.layers[17].set_weights(model.layers[21].get_weights()) 
new_audio_model.layers[18].set_weights(model.layers[23].get_weights()) 


print(new_audio_model.summary())
 

###############################################################################
                        # defining the new Visual model #
###############################################################################
new_visual_model = Model(inputs=visual_sequence,outputs=out_visual_channel)
 
new_visual_model.layers[0].set_weights(model.layers[14].get_weights()) # input layer
new_visual_model.layers[1].set_weights(model.layers[16].get_weights())
new_visual_model.layers[2].set_weights(model.layers[18].get_weights()) 
new_visual_model.layers[3].set_weights(model.layers[20].get_weights()) 
new_visual_model.layers[4].set_weights(model.layers[22].get_weights()) 


print(new_visual_model.summary())

##############################################################################

##############################################################################
###############################################################################
###############################################################################

                            ### TRAINING Procedure ###


############################################################################### 
############################################################################### metadata
meta_data = scipy.io.loadmat(datadir + 'metaData.mat', 
                 variable_names=['n_train','n_val','n_chunks'])
n_train = meta_data['n_train'][0][0]#69700#numpy.shape(X_train1)[0]

n_val = meta_data['n_val'][0][0]
n_val = n_val*5

n_chunks = meta_data['n_chunks'][0][0]
#n_train = 5000
#n_val =  1000
###############################################################################
## .............................................................................  Fitting 
#from sklearn.utils import shuffle
recall_indicator = 0
val_indicator = 1000 
allepochs_valloss = [] 
allepochs_trainloss = [] 
allavRecalls = []
###############################################################################
                        # INITIAL VALIDATION #
############################################################################### 

#................................................ loading validation data (vgg)
filename = machine_path + 'features/coco/old/threechunks/valY_ch0'
infile = open(filename,'rb')
Y_val = pickle.load(infile)
infile.close()

#........................................... loading validation data (logmels)
filename = machine_path + 'features/coco/old/threechunks/valX3_ch0'
infile = open(filename ,'rb')
X_val = pickle.load(infile)
infile.close()

X_val = numpy.array(X_val)
Y_val = numpy.array(Y_val)            
Y_val = numpy.reshape(Y_val,[Y_val.shape[0],14,14,512])

number_of_audios = X_val.shape[0]

############################################################################### # Finding validation loss #
                        
n_val = Y_val.shape[0]
valorderX,valorderY = randOrder(n_val)
bin_val_triplet = numpy.array(make_bin_target(n_val))

val_primary = model.evaluate( [Y_val[valorderY],X_val[valorderX] ],bin_val_triplet,batch_size=120) 
print('......................... val primary ... = ' + str(val_primary))
 
allepochs_valloss.append(val_primary)
model.save_weights(modeldir + 'epoch0'  + '_weights.h5')

################################################################################ Finding Recall #

audio_embeddings = new_audio_model.predict(X_val) 
visual_embeddings = new_visual_model.predict(Y_val)

# average embedding over data direction  (SISA)        
audio_embeddings = numpy.mean(audio_embeddings, axis = 1) 
visual_embeddings = numpy.mean(visual_embeddings, axis = 1)

poolsize =  1000
recall_vec = calculate_recallat10( audio_embeddings,visual_embeddings, 50,  number_of_audios , poolsize )

recall10 = numpy.mean(recall_vec)/(poolsize)
print('###############################################################...recall@10 is = ' + str(recall10) )       
allavRecalls.append(recall10) 
    
################################################################################ deleting validation data    
del Y_val
del X_val
del audio_embeddings
del visual_embeddings



for epoch in range(200):
      
    #indx_audio_class = [k for k in range(5)] #shuffle([k for k in range(5)])        
    for audio_class in range(5):
        #audio_class = indx_audio_class[i]
        
        indx_chunks = [q for q in range(n_chunks)] #shuffle([q for q in range(n_chunks)])
        for chunk in range(n_chunks):            
            
            #...................................................................... Y train
            
            filename = datadir + 'trainY' + '_ch' + str(chunk)
            infile = open(filename ,'rb')
            Y_train = pickle.load(infile)
            infile.close()
                      
            #...................................................................... X train
            
            filename = datadir + 'trainX' + str(audio_class) + '_ch' + str(chunk)
            infile = open(filename,'rb')
            X_train = pickle.load(infile)
            infile.close()
            
            #...................................................................... Reshaping
                            
            X_train = numpy.array(X_train)
            Y_train = numpy.array(Y_train)
            Y_train = numpy.reshape(Y_train,[Y_train.shape[0],14,14,512])
                       
            #...................................................................... triplet index
            
            trainorderX,trainorderY = randOrder(Y_train.shape[0])
            bin_train_triplet = numpy.array(make_bin_target(Y_train.shape[0]))
            
            
            print('......................... epoch .............................' + str(epoch ) )
            print('......................... audio_class .................................' + str(audio_class)) 
            print('......................... chunk train .................................' + str(chunk))
            
            
            history = model.fit([Y_train[trainorderY], X_train[trainorderX] ] , bin_train_triplet, shuffle=False, epochs=1,batch_size=120)   
            del Y_train,X_train 
            
    allepochs_trainloss.append(history.history['loss'][0])
    del history
                        
    epoch_cum_val = 0
    epoch_cum_recall = 0
    for audio_class in range(5):
        
    
        indx_chunks = [q for q in range(n_chunks)] #shuffle([q for q in range(n_chunks)])
        for chunk_val in range(n_chunks):        
        #.............................................................................. Y validation
            filename = datadir + 'valY' + '_ch' + str(chunk_val)            
            infile = open(filename,'rb')
            Y_val = pickle.load(infile)
            infile.close()
            
            #.............................................................................. X validation
            filename = datadir + 'valX' + str(audio_class) + '_ch' + str(chunk_val) 
            infile = open(filename ,'rb')
            X_val = pickle.load(infile)
            infile.close()
            
            #...................................................................... Reshaping
            X_val = numpy.array(X_val)
            Y_val = numpy.array(Y_val)
                        
            Y_val = numpy.reshape(Y_val,[Y_val.shape[0],14,14,512])
            #...................................................................... triplet index
            valorderX,valorderY = randOrder(Y_val.shape[0])
            bin_val_triplet = numpy.array(make_bin_target(Y_val.shape[0]))
            print('......................... chunk validation .................................' + str(chunk_val))    
            val_chunk = model.evaluate( [Y_val[valorderY],X_val[valorderX] ],bin_val_triplet,batch_size=120)     
  
            epoch_cum_val += val_chunk           
            
            ###############################################################################
                        # Finding Recall #
            ###############################################################################
            audio_embeddings = new_audio_model.predict(X_val) 
            visual_embeddings = new_visual_model.predict(Y_val)
            
            audio_embeddings = numpy.mean(audio_embeddings, axis = 1) 
            visual_embeddings = numpy.mean(visual_embeddings, axis = 1)
            
            poolsize =  1000
            recall_vec = calculate_recallat10( audio_embeddings,visual_embeddings, 50,  number_of_audios , poolsize )
        
            recall10 = numpy.mean(recall_vec)/(poolsize)

            epoch_cum_recall += recall10  
             
            ###############################################################################
            del Y_val,X_val 
            
            del audio_embeddings
            del visual_embeddings
            ###############################################################################
            ############################################################################### saving the best model
         
            
    epoch_recall = epoch_cum_recall / 15
    epoch_val = epoch_cum_val/15  
                          
      
    allavRecalls.append(epoch_recall)
    allepochs_valloss.append(epoch_val) 
    
    if epoch_recall >= recall_indicator: 
            recall_indicator = epoch_recall
            weights = model.get_weights()
            model.set_weights(weights)
            model.save_weights('%smodel_weights.h5' % modeldir)
    
    model.save_weights('%smodel_weights_lastepoch.h5' % modeldir)        
    print ('............................................    validation_loss at this epoch =    ' + str(epoch_val))
    print ('............................................    recall at this epoch =    ' + str(epoch_recall))
    scipy.io.savemat(modeldir + 'valtrainloss.mat', 
                 {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'allavRecalls':allavRecalls})
# .............................................................................
###############################################################################
########################   Testing  ###########################################
############################################################################### 
chunk_val = 1
audio_class = 1
#.............................................................................. Y validation
filename = datadir + 'valY' + '_ch' + str(chunk_val)            
infile = open(filename,'rb')
Y_val = pickle.load(infile)
infile.close()

#.............................................................................. X validation
filename = datadir + 'valX' + str(audio_class) + '_ch' + str(chunk_val) 
infile = open(filename ,'rb')
X_val = pickle.load(infile)
infile.close()

#...................................................................... Reshaping
X_val = numpy.array(X_val)
Y_val = numpy.array(Y_val)
            
Y_val = numpy.reshape(Y_val,[Y_val.shape[0],14,14,512])


X_val = X_val[0:1000]
Y_val = Y_val[0:1000]
valorderX,valorderY = randOrder(Y_val.shape[0])

X_val = X_val [valorderX]
Y_val = Y_val [valorderY] 
#......................................................................
            
layer_name = 'dot_final'  

intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

out_final = intermediate_layer_model.predict([Y_val,X_val])


layer_name = 'out_visual'  

intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
out_visual_output = intermediate_layer_model.predict([Y_val,X_val])
out_visual_output = numpy.reshape(out_visual_output , -1)

# out_visual_output_reshape = numpy.reshape(out_visual_output, (-1,14,14,512))
# out_visual_output_reshape_1 = out_visual_output_reshape[12,:,:,:]
# import matplotlib as plt 
# plt.pyplot.imshow(numpy.mean(out_visual_output_reshape_1, axis=-1))

layer_name = 'out_audio' 
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
out_audio_output = intermediate_layer_model.predict([Y_val,X_val])
out_audio_output = numpy.reshape(out_audio_output , -1)

print(numpy.min(out_final))
print(numpy.max(out_final))
print(numpy.min(out_audio_output))
print(numpy.max(out_audio_output))
print(numpy.min(out_visual_output))
print(numpy.max(out_visual_output))



layer_name = 'conv5'  
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
out_temp = intermediate_layer_model.predict([Y_val,X_val])
out_temp = numpy.reshape(out_temp , -1)
print(numpy.min(out_temp))
print(numpy.max(out_temp))
# .............................................................................

#modeldir = '/worktmp/khorrami/work/projects/project_2/outputs/step_4/models/test/version1/'
import scipy.io
import numpy
trainval = scipy.io.loadmat(modeldir+'valtrainloss.mat',variable_names= {'allepochs_valloss','allepochs_trainloss'}) 
   
import matplotlib as plt
plt.pyplot.figure(figsize=(28,18))

fig, ax = plt.pyplot.subplots()

var = trainval['allepochs_trainloss'][0]
number_of_trained_epochs = int(len(var) / 15)
average_loss = []
for item in numpy.arange(number_of_trained_epochs ):    
    temp = var[item*15 : (item+1)*15]
    average_loss.append(numpy.mean(temp))
average_loss_train = average_loss    
 
var = trainval['allepochs_valloss'][0]
number_of_trained_epochs = int(len(var) / 15)
average_loss = []
for item in numpy.arange(number_of_trained_epochs ):    
    temp = var[item*15 : (item+1)*15]
    average_loss.append(numpy.mean(temp))
average_loss_val = average_loss  
   
ax.plot(average_loss_train, label = 'train')
ax.plot(average_loss_val, label ='validation')
plt.pyplot.yticks(numpy.arange(0,2,0.1))
plt.pyplot.xticks(numpy.arange(0,number_of_trained_epochs ,2), fontsize=8)
plt.pyplot.grid()
plt.pyplot.title(' DAVEnet model (CNN0) with Adam (lr=1e-4)')
ax.legend()

plt.pyplot.savefig(modeldir+'loss_plot',format='pdf')












