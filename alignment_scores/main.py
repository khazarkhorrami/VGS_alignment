import os

###################### initial configuration  #################################

path_project = '/'


path_in_AVtensor = os.path.join(path_project , 'tensors/../')
file_in_AVtensor = '...'

path_out = os.path.join(path_project , '/')
file_out = 'glaning_' + file_in_AVtensor 


path_in_metadata  = os.path.join(path_project , '../coco_validation/')
file_in_metadata = 'processed_data_list_val.mat'

path_in_corrected_ind = os.path.join(path_project , '/')
file_in_corrected_ind = 'corrected_nouns_index.mat'

path_in_labels = os.path.join(path_project , '/')
file_in_labels = 'unified_labels.mat'

path_in_processed_nouns = os.path.join(path_project , '/')
file_in_processed_nouns = 'sub_labels.mat'


#.............................................................................. input parameters
n_categories = 80

softmax = True

res_target_h = 224    
res_target_w = 224
res_target_t = 512

res_source_h = 14
res_source_w = 14
res_source_t = 64

scale_t = int(res_target_t /res_source_t)
scale_h = int(res_target_h /res_source_h)
scale_w = int(res_target_w /res_source_w)

###############################################################################

import numpy
from alignment_scores import  compute_spatial_alignment, compute_temporal_alignment, compute_spatial_alignment_other


file_indices = path_in_corrected_ind + file_in_corrected_ind
file_AVtensor = path_in_AVtensor + file_in_AVtensor + '.mat'
file_metadata = path_in_metadata + file_in_metadata
file_nouns = path_in_processed_nouns + file_in_processed_nouns
file_labels = path_in_labels + file_in_labels
#.............................................................................. loading input files



#.............................................................................. output data
from prepare_tensors import get_input_vectors, initialize_output_vectors, prepare_all_tensors, prepare_item_tensors,save_results
###############################################################################

if __name__ == '__main__':
    
    
    tensor_input, number_of_images, all_image_ids, all_inds, all_nouns,all_onsets, all_offsets, all_reference_names = get_input_vectors (file_indices, file_AVtensor, file_metadata, file_nouns)
    all_sa_scores,all_ta_scores,all_meta_info, allrand_sa_scores, allrand_ta_scores, cm_detection, cm_rand, cm_object_area = initialize_output_vectors (n_categories)  
    tensor_input_SA , tensor_input_TA = prepare_all_tensors (tensor_input, softmax)
    
    nan_check = []
    error_check = []
    for counter_image in range(number_of_images):
        
        print(' image ..........................' , str(counter_image))
    
        #..........................................................................
        imageID = int(all_image_ids[counter_image])
        
        tensor_in_sa , tensor_in_ta = prepare_item_tensors (tensor_input_SA, tensor_input_TA, counter_image )
        tensor_rand = numpy.random.rand(512,224,224)
        
        subinds = all_inds[counter_image]   
        ref_names = all_reference_names[counter_image]    
          
        if len(subinds): 
            
            subind = subinds[0] 
            t_onsets_image = all_onsets[counter_image][0]
            t_offsets_image = all_offsets[counter_image][0]
            for counter_label in range(len(subinds)): 
                
                label_result = []           
                subnoun = all_nouns[counter_image][counter_label] 
                
                    
                if subnoun in ref_names: 
                    
                    #..............................................................................  Spatial Alignment
                    t_onset_sa = int(t_onsets_image[counter_label])
                    t_offset_sa = int(t_offsets_image[counter_label])
                    t_duration = t_offset_sa - t_onset_sa
                    
                    m_null = numpy.ones([res_target_h,res_target_w]) 
                    score_sa, area_GT , col, m_null_tp  = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_in_sa, subnoun, imageID, m_null)
                    
                    m_null = numpy.ones([res_target_h,res_target_w]) 
                    scorerand_sa, arearand_GT , col, m_null_r = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_rand, subnoun, imageID, m_null)
                    
                    
                    #.............................................................................. Temporal Alignment
                    t_onset_ta = int(numpy.maximum (t_onsets_image[counter_label]  , 0 ))
                    t_offset_ta = int(numpy.minimum (t_offsets_image[counter_label] , 512))               
                    t_duration_extra = t_offset_ta - t_onset_ta
                    
                    score_ta = compute_temporal_alignment (t_onset_ta , t_offset_ta ,tensor_in_ta, subnoun, imageID)  
                    
                    scorerand_ta = compute_temporal_alignment (t_onset_ta , t_offset_ta , tensor_rand,subnoun, imageID)
                    
                    # print(score_sa)
                    # print(score_ta)
                    if score_sa > 1 or score_ta > 1 :
                        print('.......................................................................................')
                        error_check.append(counter_image)
                    
                    #.............................................................................. Saving results
                    # very exceptionally nan might occure because of upsampling and thresholding of GT mask
                    elif ~numpy.isnan(score_sa) and ~numpy.isnan(score_ta) :
                    
                        all_ta_scores[col,0] = all_ta_scores[col,0] + score_ta
                        all_sa_scores[col,0] = all_sa_scores[col,0] + score_sa
                        
                        all_meta_info [col,0] = all_meta_info [col,0] + area_GT
                        all_meta_info [col,1] = all_meta_info [col,1] + t_duration
                        all_meta_info [col,2] = all_meta_info [col,2] + 1
                        
                        allrand_ta_scores[col,0] = allrand_ta_scores[col,0] + scorerand_ta
                        allrand_sa_scores[col,0] = allrand_sa_scores[col,0] + scorerand_sa
                        
                        cm_detection[col,col] = cm_detection[col,col] + score_sa
                        cm_rand[col,col] = cm_rand[col,col] + scorerand_sa
                        
                        # confusion matrix 
                        m_null_other = m_null_tp
                        for annitem in ref_names:
                            score_s, area_GT , row , m_null_label = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_in_sa, annitem, imageID , m_null_tp)
                            m_null_other = m_null_other * m_null_label
                            score_r, area_GT , row , m_null = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_rand, annitem, imageID ,m_null_tp)
                            
                            if ~numpy.isnan(score_s) :
                                cm_detection[row,col] = cm_detection[row,col] + score_s
                                cm_rand[row,col] = cm_rand[row,col] + score_r
                                
                                cm_object_area [row,0] = cm_object_area [row,0] + area_GT
                                cm_object_area [row,1] = cm_object_area [row,0] + 1
                        score_s_other, area_other = compute_spatial_alignment_other (t_onset_sa, t_offset_sa ,tensor_in_sa, m_null_other)
                        score_r_other, area_other = compute_spatial_alignment_other (t_onset_sa, t_offset_sa ,tensor_rand, m_null_other)
                        
                        if ~numpy.isnan(score_s_other):
                            cm_detection[80,col] = cm_detection[80,col] + score_s_other
                            cm_object_area [80,0] = cm_object_area [80,0] + area_other
                            cm_object_area [80,1] = cm_object_area [80,0] + 1
                            
                            cm_rand[80,col] = cm_rand[80,col] + score_r_other
                            
                    else:
                        nan_check.append(counter_image)

    filename = path_out + file_out + '_T.mat'
    save_results (filename, all_sa_scores,all_ta_scores,allrand_ta_scores, allrand_sa_scores,cm_detection,cm_object_area)


