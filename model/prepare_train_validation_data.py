import numpy
import pickle

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
        
if __name__ == '__main__':
    length_sequence = 512 
    number_of_epochs = 200
    number_of_caps_per_image = 5
    set_of_train_chunks = []
    set_of_validation_chunks = []   
    visual_feature_name = ''
    audio_feature_name = ''  
    featuredir = ''
    use_pretrained = False
    set_of_input_chunks = set_of_train_chunks
    for chunk_counter, chunk_name in enumerate(set_of_input_chunks): 
        for i_caption in range(number_of_caps_per_image):
            Ydata, Xdata, bin_triplet = prepare_triplet_data (featuredir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , length_sequence)
       
        