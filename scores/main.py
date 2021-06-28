import os
from aligning_scores import find_alignment_scores, find_glancing_scores

###################### initial configuration  #################################

path_project = ''

path_in = os.path.join(path_project , 'input_files')
path_out = os.path.join(path_project , 'output_files')


path_in_AVtensor = os.path.join(path_in,'tensors')
file_in_AVtensor = '...'

file_in_metadata = 'processed_data_list_val.mat'
file_in_corrected_ind = 'corrected_nouns_index.mat'

path_in_labels = os.path.join(path_project , '/')
file_in_labels = 'unified_labels.mat'

path_in_processed_nouns = os.path.join(path_project , '/')
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

file_indices = os.path.join(path_in,file_in_corrected_ind) 
file_AVtensor = os.path.join(path_in_AVtensor , file_in_AVtensor + '.mat') 
file_metadata = os.path.join(path_in,file_in_metadata)
file_nouns = os.path.join(path_in,file_in_processed_nouns) 
file_labels = os.path.join(path_in,file_in_labels)


###############################################################################

if __name__ == '__main__':
    find_alignment_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax)
    find_glancing_scores(file_indices,file_AVtensor,file_metadata,file_nouns,n_categories,softmax)
    
