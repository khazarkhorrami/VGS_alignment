
import numpy
import scipy.io


def get_input_vectors (file_indices, file_metadata, file_labels,  file_nouns, file_AVtensor):

    data = scipy.io.loadmat(file_indices , variable_names = ['ind_accepted'])
    ind_accepted = data['ind_accepted'][0]
    number_of_images = len(ind_accepted)
    #.....................
    # hidden layer weights (AV-tensor) is saved as an array of dimensions (N_samples, time_frames, pixel_h*pixel_w) .
    data = scipy.io.loadmat( file_AVtensor , variable_names = ['out_layer'])
    tensor_input = data['out_layer']
    tensor_input = tensor_input [ind_accepted]
    tensor_input = tensor_input    
    #.....................
    
    data = scipy.io.loadmat(file_metadata, variable_names=['list_of_images','image_id_all'])
    all_image_ids = data ['image_id_all'][0]
    all_image_ids = [ item[0].strip() for item in all_image_ids] 
    all_image_ids = [all_image_ids[item] for item in ind_accepted]
    #.....................
    
    data = scipy.io.loadmat(file_nouns, variable_names = ['all_accpted_words','all_accepted_ind','all_accpted_onsets','all_accpted_offsets'])
    all_nouns = data ['all_accpted_words'][0]
    all_inds = data ['all_accepted_ind'][0]
    all_onsets = data ['all_accpted_onsets'][0]
    all_offsets = data ['all_accpted_offsets'][0]
    all_nouns = [ [subitem.strip() for subitem in imitem if subitem ] for imitem in all_nouns ]
    
    #.....................
    
    data = scipy.io.loadmat(file_labels, variable_names =['ref_names_all'])
    all_reference_names = data['ref_names_all'][0]
    all_reference_names = [all_reference_names[item] for item in ind_accepted]
    all_reference_names = [ [phoneme.strip() for phoneme in utterance] for utterance in all_reference_names ]
    return tensor_input, number_of_images, all_image_ids, all_inds, all_nouns,all_onsets, all_offsets , all_reference_names


def initialize_output_vectors (n_categories):
    all_sa_scores =  numpy.zeros([n_categories, 1])
    all_ta_scores =  numpy.zeros([n_categories, 1])
    all_meta_info =  numpy.zeros([n_categories, 3])
    
    allrand_sa_scores =  numpy.zeros([n_categories, 1])
    allrand_ta_scores =  numpy.zeros([n_categories, 1])
   
    
    cm_detection = numpy.zeros([n_categories + 1 ,n_categories])
    cm_object_area = numpy.zeros([81, 2])
    return all_sa_scores,all_ta_scores,all_meta_info, allrand_sa_scores, allrand_ta_scores, cm_detection, cm_object_area

def prepare_all_tensors (tensor_input, softmax, res_source_t , res_source_h, res_source_w):
    
    if softmax:
        tensor_input_SA = softmax_spatial(tensor_input) 
        tensor_input_TA = softmax_temporal(tensor_input) 
    else:
        tensor_input_SA = tensor_input 
        tensor_input_TA = tensor_input 
        
    tensor_input_SA = numpy.reshape(tensor_input_SA, [tensor_input.shape[0], res_source_t , res_source_h, res_source_w])    
    tensor_input_TA = numpy.reshape(tensor_input_TA, [tensor_input.shape[0], res_source_t , res_source_h, res_source_w])
    
    return tensor_input_SA , tensor_input_TA


def prepare_item_tensors (tensor_input_SA, tensor_input_TA, counter_image ,scale_t , scale_h, scale_w ):
    
    tensor_in_sa  = tensor_input_SA[counter_image]
    tensor_in_ta  = tensor_input_TA[counter_image]
    
    tensor_in_sa =  numpy.clip(tensor_in_sa, 0, numpy.max(tensor_in_sa))
    tensor_in_ta =   numpy.clip(tensor_in_ta, 0, numpy.max(tensor_in_ta))
    
    tensor_in_sa = upsample_3D (tensor_in_sa,scale_t , scale_h, scale_w)
    tensor_in_ta = upsample_3D (tensor_in_ta, scale_t , scale_h, scale_w)
    
    return tensor_in_sa , tensor_in_ta   


def softmax(input_vec):
    output_vec = []
    for item in input_vec:
        e_x = numpy.exp(item - numpy.max(item))
        output_vec.append(e_x / e_x.sum())
    return numpy.array(output_vec)

def softmax_spatial (tensor_in):
    tensor_out = []
    for counter , tensor_av in enumerate(tensor_in):
        tensor_av_soft = softmax(tensor_av)
        tensor_out.append(tensor_av_soft)    
    return numpy.array(tensor_out)

def softmax_temporal (tensor_in):
    tensor_out = []
    for counter , tensor_av in enumerate(tensor_in):
        tensor_va = numpy.transpose(tensor_av)
        tensor_va_soft = softmax(tensor_va)
        tensor_av_soft = numpy.transpose(tensor_va_soft)
        tensor_out.append(tensor_av_soft)    
    return numpy.array(tensor_out)


def upsample_3D (input_tensor,scale_T , scale_H, scale_W):
    tensor_detected_uptime = numpy.repeat(input_tensor,scale_T, axis=0)
    output_tensor = numpy.repeat (numpy.repeat(tensor_detected_uptime,scale_W, axis=2)  , scale_H , axis=1)
    return output_tensor

# def save_results (filename, all_sa_scores,all_ta_scores,allrand_ta_scores, allrand_sa_scores,cm_detection,cm_object_area):
#     scipy.io.savemat(filename , {'all_sa_scores':all_sa_scores, 'all_ta_scores':all_ta_scores, 
#                                                       'allrand1_ta_scores': allrand_ta_scores, 'allrand1_sa_scores': allrand_sa_scores ,
#                                                       'all_meta_info':all_meta_info, 'nan_check':nan_check,
#                                                       'cm_detection':cm_detection, 'cm_object_area':cm_object_area})
                                             
