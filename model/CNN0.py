from keras import backend as K
import keras
from keras.layers import Lambda
from keras.models import Model
from keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization
from keras.layers import  MaxPooling1D,  Conv1D,Conv2D


def build_audio_model (X_shape,modeldir ):
    
    dropout_size = 0.3
    activation_C='relu'

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
    
    dn1 = Dense(512,activation='linear',name='dense_audio')(bn5) 
    norm1 = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(dn1)
    
    out_audio_channel = norm1 
    audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
    
    return audio_sequence , out_audio_channel , audio_model

    
def build_visual_model (Y_shape,modeldir ): 
    dropout_size = 0.3
    visual_sequence = Input(shape=Y_shape)
    visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
    
    forward_visual = Conv2D(512,(3,3),strides=(1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
    dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
    bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
    
    resh1 = Reshape([196,512],name='reshape_visual')(bn_visual) 
    
    dn1 = Dense(512,activation='linear',name='dense_visual')(resh1) 
    norm1 = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(dn1)
    
    out_visual_channel = norm1
    visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
    return visual_sequence , out_visual_channel , visual_model
    

def build_vgsmodel (X_shape, Y_shape , modeldir):
    
    audio_sequence , out_audio_channel , audio_model = build_audio_model (X_shape,modeldir )
    visual_sequence , out_visual_channel , visual_model =  build_visual_model (Y_shape,modeldir )  
    
    # combining audio and visual channels     
    I = out_visual_channel
    A = out_audio_channel    
    mapIA = keras.layers.dot([I,A],axes=-1,normalize = True,name='dot_matchmap')  
    lambda_layer = Lambda(custom_layer, name="final_layer")(mapIA)
    vgs_model = Model(inputs=[visual_sequence, audio_sequence], outputs = lambda_layer )
    vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
    print(vgs_model.summary())
    vgs_model.save('%smodel' % modeldir)
    
    return vgs_model, A , I   
 
                  
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


def assign_audiomodel (audio_model, vgs_model):
    
    for n in range(13):
        audio_model.layers[n].set_weights(vgs_model.layers[n].get_weights())
       
    audio_model.layers[13].set_weights(vgs_model.layers[14].get_weights())
    audio_model.layers[14].set_weights(vgs_model.layers[16].get_weights())
    audio_model.layers[15].set_weights(vgs_model.layers[18].get_weights()) 
    audio_model.layers[16].set_weights(vgs_model.layers[20].get_weights()) 
    audio_model.layers[17].set_weights(vgs_model.layers[22].get_weights()) 
    audio_model.layers[18].set_weights(vgs_model.layers[24].get_weights()) 
    audio_model.layers[19].set_weights(vgs_model.layers[26].get_weights())
    audio_model.layers[20].set_weights(vgs_model.layers[28].get_weights()) # out audio

 

def assign_visualmodel (visual_model,vgs_model):
     
    visual_model.layers[0].set_weights(vgs_model.layers[13].get_weights()) # input layer
    visual_model.layers[1].set_weights(vgs_model.layers[15].get_weights())
    visual_model.layers[2].set_weights(vgs_model.layers[17].get_weights()) 
    visual_model.layers[3].set_weights(vgs_model.layers[19].get_weights()) 
    visual_model.layers[4].set_weights(vgs_model.layers[21].get_weights()) 
    visual_model.layers[5].set_weights(vgs_model.layers[23].get_weights()) 
    visual_model.layers[6].set_weights(vgs_model.layers[25].get_weights())
    visual_model.layers[7].set_weights(vgs_model.layers[27].get_weights())# out visual




if __name__ == '__main__':
       
    length_sequence = 512
    modeldir = ''  
    X_shape = (length_sequence,40)
    Y_shape = (14,14,512)  
    
    vgs_model, embedding_audio , embedding_visual =  build_vgsmodel (X_shape, Y_shape , modeldir)
    audio_sequence , out_audio_channel , audio_model = build_audio_model (X_shape,modeldir )
    visual_sequence , out_visual_channel , visual_model =  build_visual_model (Y_shape,modeldir )
    assign_audiomodel(audio_model, vgs_model)
    assign_visualmodel(visual_model,vgs_model)
    
    