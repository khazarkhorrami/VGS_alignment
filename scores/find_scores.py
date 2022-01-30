from prepare_tensors import get_input_vectors, initialize_output_vectors, prepare_all_tensors, prepare_item_tensors,save_results
from aligning_scores import compute_AS_object , compute_AS_word , compute_AS_object_other , compute_GS_object , compute_GS_word , compute_GS_object_other
import numpy
import os



def find_glancing_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax):
    
    tensor_input, number_of_images, all_image_ids, all_inds, all_nouns,all_onsets, all_offsets, all_reference_names = get_input_vectors (file_indices, file_AVtensor, file_metadata, file_nouns)
    all_OD_scores,all_WD_scores,all_meta_info, allrand_OD_scores, allrand_WD_scores, cm_detection, cm_object_area = initialize_output_vectors (n_categories)  
    tensor_input_SA , tensor_input_TA = prepare_all_tensors (tensor_input, softmax)
    
    nan_check = []
    error_check = []
    for counter_image in range(number_of_images):
        
        print(' image ..........................' , str(counter_image))
    
        #..........................................................................
        imageID = int(all_image_ids[counter_image])
        
        tensor_in_OD , tensor_in_WD = prepare_item_tensors (tensor_input_SA, tensor_input_TA, counter_image )
        tensor_rand = numpy.random.rand(res_target_t,res_target_h,res_target_w)
        
        subinds = all_inds[counter_image]   
        ref_names = all_reference_names[counter_image]    
          
        if len(subinds): 
            
            t_onsets_image = all_onsets[counter_image][0]
            t_offsets_image = all_offsets[counter_image][0]
            for counter_label in range(len(subinds)): 
                
                subnoun = all_nouns[counter_image][counter_label] 
                
                    
                if subnoun in ref_names: 
                    
                    #..............................................................................  object detection
                    t_onset_OD = int(t_onsets_image[counter_label])
                    t_offset_OD = int(t_offsets_image[counter_label])
                    t_duration = t_offset_OD - t_onset_OD
                    
                    m_null = numpy.ones([res_target_h,res_target_w]) 
                    score_OD, area_GT , col, m_null_tp  = compute_GS_object  (t_onset_OD, t_offset_OD , res_target_w , res_target_h, tensor_in_OD, subnoun, imageID, m_null) 
                    
                    m_null = numpy.ones([res_target_h,res_target_w])
                    scorerand_OD, arearand_GT , col, m_null_r = compute_GS_object (t_onset_OD, t_offset_OD , res_target_w , res_target_h, tensor_rand, subnoun, imageID, m_null)               
                    
                    #.............................................................................. word detection
                    t_onset_WD = int(numpy.maximum (t_onsets_image[counter_label]  , 0 ))
                    t_offset_WD = int(numpy.minimum (t_offsets_image[counter_label] , 512))               
                    #t_duration_extra = t_offset_WD - t_onset_WD
                    
                    score_WD = compute_GS_word (t_onset_WD , t_offset_WD, res_target_w , res_target_h ,tensor_in_WD, subnoun, imageID)  
                    
                    scorerand_WD = compute_GS_word (t_onset_WD , t_offset_WD, res_target_w , res_target_h , tensor_rand,subnoun, imageID)
                    
                    # print(score_sa)
                    # print(score_ta)
                    if score_OD > 1 or score_WD > 1 :
                        print('.......................................................................................')
                        error_check.append(counter_image)
                    
                    #.............................................................................. Saving results
                    # very exceptionally nan might occure because of upsampling and thresholding of GT mask
                    elif ~numpy.isnan(score_OD) and ~numpy.isnan(score_WD) :
                    
                        all_WD_scores[col,0] = all_WD_scores[col,0] + score_WD
                        all_OD_scores[col,0] = all_OD_scores[col,0] + score_OD
                        
                        all_meta_info [col,0] = all_meta_info [col,0] + area_GT
                        all_meta_info [col,1] = all_meta_info [col,1] + t_duration
                        all_meta_info [col,2] = all_meta_info [col,2] + 1
                        
                        allrand_WD_scores[col,0] = allrand_WD_scores[col,0] + scorerand_WD
                        allrand_OD_scores[col,0] = allrand_OD_scores[col,0] + scorerand_OD
                        
                        cm_detection[col,col] = cm_detection[col,col] + score_OD
                        
                        #.............................................................................. confusion matrix 
                        m_null_other = m_null_tp
                        for annitem in ref_names:
                            score_s, area_GT , row , m_null_label = compute_GS_object (t_onset_OD, t_offset_OD , res_target_w , res_target_h ,tensor_in_OD, annitem, imageID , m_null_tp)
                            m_null_other = m_null_other * m_null_label                            
                            
                            if ~numpy.isnan(score_s) :
                                cm_detection[row,col] = cm_detection[row,col] + score_s                                
                                
                                cm_object_area [row,0] = cm_object_area [row,0] + area_GT
                                cm_object_area [row,1] = cm_object_area [row,0] + 1
                        score_s_other, area_other = compute_GS_object_other(t_onset_OD, t_offset_OD , res_target_w , res_target_h, tensor_in_OD, m_null_other) 
                        score_r_other, area_other = compute_GS_object_other(t_onset_OD, t_offset_OD , res_target_w , res_target_h, tensor_rand, m_null_other)
                        
                        if ~numpy.isnan(score_s_other):
                            cm_detection[80,col] = cm_detection[80,col] + score_s_other
                            cm_object_area [80,0] = cm_object_area [80,0] + area_other
                            cm_object_area [80,1] = cm_object_area [80,0] + 1
                            
                    else:
                        nan_check.append(counter_image)

    filename = path_out + file_out + '_T.mat'
    save_results (filename, all_OD_scores,all_WD_scores,allrand_WD_scores, allrand_OD_scores,cm_detection,cm_object_area)





def find_alignment_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax):
    
    tensor_input, number_of_images, all_image_ids, all_inds, all_nouns,all_onsets, all_offsets, all_reference_names = get_input_vectors (file_indices, file_AVtensor, file_metadata, file_nouns)
    all_OD_scores,all_WD_scores,all_meta_info, allrand_OD_scores, allrand_WD_scores, cm_detection, cm_object_area = initialize_output_vectors (n_categories)  
    tensor_input_SA , tensor_input_TA = prepare_all_tensors (tensor_input, softmax)
    
    nan_check = []
    error_check = []
    for counter_image in range(number_of_images):
        
        print(' image ..........................' , str(counter_image))
    
        #..........................................................................
        imageID = int(all_image_ids[counter_image])
        
        tensor_in_OD , tensor_in_WD = prepare_item_tensors (tensor_input_SA, tensor_input_TA, counter_image )
        tensor_rand = numpy.random.rand(res_target_t,res_target_h,res_target_w)
        
        subinds = all_inds[counter_image]   
        ref_names = all_reference_names[counter_image]    
          
        if len(subinds): 
            
            t_onsets_image = all_onsets[counter_image][0]
            t_offsets_image = all_offsets[counter_image][0]
            for counter_label in range(len(subinds)): 
                
                subnoun = all_nouns[counter_image][counter_label] 
                
                    
                if subnoun in ref_names: 
                    
                    #..............................................................................  object detection
                    t_onset_OD = int(t_onsets_image[counter_label])
                    t_offset_OD = int(t_offsets_image[counter_label])
                    t_duration = t_offset_OD - t_onset_OD
                    
                    m_null = numpy.ones([res_target_h,res_target_w]) 
                    score_OD, area_GT , col, m_null_tp  = compute_AS_object (t_onset_OD, t_offset_OD, res_target_w , res_target_h ,tensor_in_OD, subnoun, imageID, m_null)
                    
                    m_null = numpy.ones([res_target_h,res_target_w]) 
                    scorerand_OD, arearand_GT , col, m_null_r = compute_AS_object (t_onset_OD, t_offset_OD, res_target_w , res_target_h ,tensor_rand, subnoun, imageID, m_null)                  
                    
                    #.............................................................................. word detection
                    t_onset_WD = int(numpy.maximum (t_onsets_image[counter_label]  , 0 ))
                    t_offset_WD = int(numpy.minimum (t_offsets_image[counter_label] , 512))               
                    #t_duration_extra = t_offset_WD - t_onset_WD
                    
                    score_WD = compute_AS_word (t_onset_WD , t_offset_WD, res_target_w , res_target_h ,tensor_in_WD, subnoun, imageID)  
                    
                    scorerand_WD = compute_GS_word (t_onset_WD , t_offset_WD, res_target_w , res_target_h , tensor_rand,subnoun, imageID)
                    
                    # print(score_sa)
                    # print(score_ta)
                    if score_OD > 1 or score_WD > 1 :
                        print('.......................................................................................')
                        error_check.append(counter_image)
                    
                    #.............................................................................. Saving results
                    # very exceptionally nan might occure because of upsampling and thresholding of GT mask
                    elif ~numpy.isnan(score_OD) and ~numpy.isnan(score_WD) :
                    
                        all_WD_scores[col,0] = all_WD_scores[col,0] + score_WD
                        all_OD_scores[col,0] = all_OD_scores[col,0] + score_OD
                        
                        all_meta_info [col,0] = all_meta_info [col,0] + area_GT
                        all_meta_info [col,1] = all_meta_info [col,1] + t_duration
                        all_meta_info [col,2] = all_meta_info [col,2] + 1
                        
                        allrand_WD_scores[col,0] = allrand_WD_scores[col,0] + scorerand_WD
                        allrand_OD_scores[col,0] = allrand_OD_scores[col,0] + scorerand_OD
                        
                        cm_detection[col,col] = cm_detection[col,col] + score_OD
                        
                        #.............................................................................. confusion matrix 
                        m_null_other = m_null_tp
                        for annitem in ref_names:
                            score_s, area_GT , row , m_null_label = compute_AS_object (t_onset_OD, t_offset_OD, res_target_w , res_target_h ,tensor_in_OD, annitem, imageID , m_null_tp)
                            m_null_other = m_null_other * m_null_label                            
                            
                            if ~numpy.isnan(score_s) :
                                cm_detection[row,col] = cm_detection[row,col] + score_s                                
                                
                                cm_object_area [row,0] = cm_object_area [row,0] + area_GT
                                cm_object_area [row,1] = cm_object_area [row,0] + 1
                        score_s_other, area_other = compute_AS_object_other(t_onset_OD, t_offset_OD, res_target_w , res_target_h ,tensor_in_OD, m_null_other)
                        score_r_other, area_other = compute_AS_object_other(t_onset_OD, t_offset_OD, res_target_w , res_target_h ,tensor_rand, m_null_other)
                        
                        if ~numpy.isnan(score_s_other):
                            cm_detection[80,col] = cm_detection[80,col] + score_s_other
                            cm_object_area [80,0] = cm_object_area [80,0] + area_other
                            cm_object_area [80,1] = cm_object_area [80,0] + 1
                            
                    else:
                        nan_check.append(counter_image)

    filename = path_out + file_out + '_T.mat'
    save_results (filename, all_OD_scores,all_WD_scores,allrand_WD_scores, allrand_OD_scores,cm_detection,cm_object_area)
    

if __name__ == '__main__':

    ###################### initial configuration  #################################
    
    path_project = ''
    
    path_in = os.path.join(path_project , 'input_files')
    path_out = os.path.join(path_project , 'output_files')
    
    
    path_in_AVtensor = os.path.join(path_in,'tensors')
    file_in_AVtensor = ''
    
    path_in_metadata = "../../testdata/0/"
    file_in_metadata = 'processed_data_list.mat'
    
    path_in_corrected_ind = "../../testdata/3/"
    file_in_corrected_ind = 'corrected_nouns_index.mat'
    
    path_in_labels = "../../testdata/4/"
    file_in_labels = 'unified_labels.mat'
    
    path_in_processed_nouns = "../../testdata/6/"
    file_in_processed_nouns = 'sub_labels.mat'
    
    find_GS = False
    
    if find_GS:
        file_out = 'GS_' + file_in_AVtensor 
    else:
        file_out = 'AS_' + file_in_AVtensor 
    
    
    # input parameters  
    softmax = True
    
    n_categories = 80
    
    res_target_h = 224    
    res_target_w = 224
    res_target_t = 512
    
    res_source_h = 14
    res_source_w = 14
    res_source_t = 64
    
    scale_t = int(res_target_t /res_source_t)
    scale_h = int(res_target_h /res_source_h)
    scale_w = int(res_target_w /res_source_w)
    
    file_indices = os.path.join(path_in_corrected_ind , file_in_corrected_ind) 
    file_AVtensor = os.path.join(path_in_AVtensor , file_in_AVtensor + '.mat') 
    file_metadata = os.path.join(path_in_metadata , file_in_metadata)
    file_nouns = os.path.join(path_in_processed_nouns , file_in_processed_nouns) 
    file_labels = os.path.join(path_in_labels , file_in_labels)
    
    
    ###############################################################################

    find_alignment_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax)
    find_glancing_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax)
    

