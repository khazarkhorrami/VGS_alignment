
import config as cfg
from find_scores import find_alignment_scores, find_glancing_scores

###################### initial configuration  #################################

# input files 

file_indices = cfg.paths['file_indices']
file_metadata =  cfg.paths['file_metadata']
file_nouns = cfg.paths['file_nouns']
file_labels = cfg.paths['file_labels']
file_AVtensor = cfg.paths['file_AVtensor']
path_output = cfg.paths['path_output']


# input parameters

find_GS = cfg.score_settings['find_GS']
softmax = cfg.score_settings['softmax']
n_categories = cfg.score_settings['n_categories']

res_target_h = cfg.score_settings['res_target_h']    
res_target_w = cfg.score_settings['res_target_w']
res_target_t = cfg.score_settings['res_target_t']

res_source_h = cfg.score_settings['res_source_h']
res_source_w = cfg.score_settings['res_source_w']
res_source_t = cfg.score_settings['res_source_t']

scale_t = int(res_target_t /res_source_t)
scale_h = int(res_target_h /res_source_h)
scale_w = int(res_target_w /res_source_w)

###############################################################################

if __name__ == '__main__':

    files = [file_indices,file_metadata,file_labels,file_nouns,file_AVtensor,path_output]
    parameters_1 = [softmax,n_categories]
    parameters_2 = [res_target_h,res_target_w,res_target_t,res_source_h,res_source_w,res_source_t]
    
    if find_GS:
        find_glancing_scores(files, parameters_1, parameters_2)
    else:
        find_alignment_scores(files, parameters_1, parameters_2)
    
    
