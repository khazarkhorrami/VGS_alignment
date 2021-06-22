from keras import backend as K
import keras
from keras.layers import Lambda
from keras.models import Model
from keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import  MaxPooling1D,  Conv1D,Conv2D
from keras.layers import Softmax, Permute, AveragePooling1D, Concatenate


def build_vgsmodel (X_shape, Y_shape , modeldir):
    
    dropout_size = 0.3
    activation_C='relu'

    ########################################################################### Audio Network
    audio_sequence = Input(shape=X_shape)
    
    # layer 1  
    forward1 = Conv1D(128,5,padding="same",activation=activation_C,name = 'conv1')(audio_sequence)
    dr1 = Dropout(dropout_size)(forward1)
    bn1 = BatchNormalization(axis=2)(dr1)
    
    # layer 2
    forward2 = Conv1D(256,11,padding="same",activation=activation_C,name = 'conv2')(bn1)
    dr2 = Dropout(dropout_size)(forward2)
    bn2 = BatchNormalization(axis=2)(dr2) 
    pool2 = MaxPooling1D(3,strides = 2, padding='same')(bn2)
    
    # layer 3
    forward3 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv3')(pool2)
    dr3 = Dropout(dropout_size)(forward3)
    bn3 = BatchNormalization(axis=2)(dr3) 
    pool3 = MaxPooling1D(3,strides = 2,padding='same')(bn3)
    
    # layer 4
    forward4 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv4')(pool3)
    dr4 = Dropout(dropout_size)(forward4)
    bn4 = BatchNormalization(axis=2)(dr4) 
    pool4 = MaxPooling1D(3,strides = 2,padding='same')(bn4)
    
    # layer 5
    forward5 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv5')(pool4)
    dr5 = Dropout(dropout_size)(forward5)
    bn5 = BatchNormalization(axis=2,name='audio_branch')(dr5) 
    out_audio_channel = bn5

    embedding_audio = out_audio_channel
    ########################################################################### Visual Network
    
    visual_sequence = Input(shape=Y_shape)
    forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same",name = 'conv_visual')(visual_sequence)
    dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
    bn_visual = BatchNormalization(axis=-1)(dr_visual)
    
    out_visual_channel = Reshape([196,512],name='reshape_visual')(bn_visual)
    
    embedding_visual = out_visual_channel
    ###############################################################################  Attention I for query Audio
    # checks which part of image gets more attention based on audio query.   
    keyImage = out_visual_channel
    valueImage = out_visual_channel
    queryAudio = out_audio_channel
    
    scoreI = keras.layers.dot([queryAudio,keyImage], normalize=False, axes=-1,name='scoreI')
    weightID = Dense(196,activation='sigmoid')(scoreI)
    weightI = Softmax(name='weigthI')(scoreI)
    
    valueImage = Permute((2,1))(valueImage)
    attentionID = keras.layers.dot([weightID, valueImage], normalize=False, axes=-1,name='attentionID')
    attentionI = keras.layers.dot([weightI, valueImage], normalize=False, axes=-1,name='attentionI')
    
    poolAttID = AveragePooling1D(512, padding='same')(attentionID)
    poolAttI = AveragePooling1D(512, padding='same')(attentionI)
    
    poolqueryAudio = AveragePooling1D(512, padding='same')(queryAudio)
    
    outAttAudio = Concatenate(axis=-1)([poolAttI,poolAttID, poolqueryAudio])
    outAttAudio = Reshape([1536],name='reshape_out_attAudio')(outAttAudio)

    
    ###############################################################################  Attention A  for query Image
    # checks which part of audio gets more attention based on image query.
    keyAudio = out_audio_channel
    valueAudio = out_audio_channel
    queryImage = out_visual_channel
    
    scoreA = keras.layers.dot([queryImage,keyAudio], normalize=False, axes=-1,name='scoreA')
    weightAD = Dense(64,activation='sigmoid')(scoreA)
    weightA = Softmax(name='weigthA')(scoreA)
    
    valueAudio = Permute((2,1))(valueAudio)
    attentionAD = keras.layers.dot([weightAD, valueAudio], normalize=False, axes=-1,name='attentionAD')
    attentionA = keras.layers.dot([weightA, valueAudio], normalize=False, axes=-1,name='attentionA')
    
    poolAttAD = AveragePooling1D(512, padding='same')(attentionAD)
    poolAttA = AveragePooling1D(512, padding='same')(attentionA)
    
    poolqueryImage = AveragePooling1D(512, padding='same')(queryImage)
    
    outAttImage = Concatenate(axis=-1)([poolAttA,poolAttAD, poolqueryImage])
    outAttImage = Reshape([1536],name='reshape_out_attImage') (outAttImage)
    
    ############################################################################### combining audio and visual channels
    #out_audio = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
    #out_visual = Dense(512,activation='linear',name='dense_visual')(out_visual_channel) 
    
    out_audio = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(outAttAudio)
    out_visual = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(outAttImage)
     
    I = out_visual
    A = out_audio
    
    mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_final')
    
    vgs_model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA )
    vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
    print(vgs_model.summary())
    vgs_model.save('%smodel' % modeldir)
    
    return vgs_model, embedding_audio , embedding_visual 

                      
def triplet_loss(y_true,y_pred):
    
    margin = 0.1

    Sp = y_pred[0::3]

    Si = y_pred[1::3]

    Sc = y_pred[2::3]
    
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0)                      
                                            

def custom_layer(tensor):
    x= tensor   
    sisa = K.mean( (K.mean(x, axis=1)), axis=-1)
    sisa = Reshape([1],name='reshape_misa')(sisa)
    return sisa


def build_audiomodel (X_shape, Y_shape , modeldir):
    vgs_model, embedding_audio , embedding_visual = build_vgsmodel (X_shape, Y_shape , modeldir)
    audio_sequence = Input(shape=X_shape)
    audio_model = Model(inputs=audio_sequence,outputs=embedding_audio)
    
    for n in range(14):
        audio_model.layers[n].set_weights(vgs_model.layers[n].get_weights())
       
    audio_model.layers[14].set_weights(vgs_model.layers[15].get_weights())
    audio_model.layers[15].set_weights(vgs_model.layers[17].get_weights())
    audio_model.layers[16].set_weights(vgs_model.layers[19].get_weights()) 
    audio_model.layers[17].set_weights(vgs_model.layers[21].get_weights()) 
    audio_model.layers[18].set_weights(vgs_model.layers[23].get_weights()) 
    
    print(audio_model.summary())
 

def build_visualmodel (X_shape, Y_shape , modeldir):
    vgs_model, embedding_audio , embedding_visual = build_vgsmodel (X_shape, Y_shape , modeldir)
    visual_sequence = Input(shape=Y_shape)
    visual_model = Model(inputs=visual_sequence,outputs=embedding_visual)
     
    visual_model.layers[0].set_weights(vgs_model.layers[14].get_weights()) # input layer
    visual_model.layers[1].set_weights(vgs_model.layers[16].get_weights())
    visual_model.layers[2].set_weights(vgs_model.layers[18].get_weights()) 
    visual_model.layers[3].set_weights(vgs_model.layers[20].get_weights()) 
    visual_model.layers[4].set_weights(vgs_model.layers[22].get_weights()) 
    
    print(visual_model.summary())

if __name__ == '__main__':
       
    length_sequence = 512
    modeldir = ''  
    X_shape = (length_sequence,40)
    Y_shape = (14,14,512)  
    
    vgs_model, embedding_audio , embedding_visual =  build_vgsmodel (X_shape, Y_shape , modeldir)
    audio_model = build_audiomodel (X_shape, Y_shape , modeldir)
    visual_model = build_visualmodel (X_shape, Y_shape , modeldir)
