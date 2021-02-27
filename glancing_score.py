# This file takes input tensot T(time*h*w) and computes 2 alignments scores
# 1. spatial alignment of visual object (how well the model can locate the object within image in a given time window )
# 2. temporal alignment of spoken word (how well the model can locate the word within spoken utterance at given object mask)


#import pickle
import numpy
import scipy.io
import cv2
import os

path_project = '/worktmp/hxkhkh/project_3/'

path_in_metadata  = os.path.join(path_project , 'outputs/step_0/coco_validation/')
file_in_metadata = 'processed_data_list_val.mat'

path_in_corrected_ind = os.path.join(path_project , 'outputs/step_1/3/')
file_in_corrected_ind = 'corrected_nouns_index.mat'

path_in_labels = os.path.join(path_project , 'outputs/step_1/4/')
file_in_labels = 'unified_labels.mat'

path_in_processed_nouns = os.path.join(path_project , 'outputs/step_1/6/')
file_in_processed_nouns = 'sub_labels.mat'

path_in_AVtensor = os.path.join('/worktmp/hxkhkh/project2/' ,'outputs/step_5/hidden/')
file_in_AVtensor = 'SI_CNN2_v2'


path_out = os.path.join(path_project , 'outputs/step_2/')
file_out = 'glancing_' + file_in_AVtensor 
#.............................................................................. input parameters

res_target_h = 224    
res_target_w = 224
res_target_t = 512

res_source_h = 14
res_source_w = 14
res_source_t = 64

scale_t = int(res_target_t /res_source_t)
scale_h = int(res_target_h /res_source_h)
scale_w = int(res_target_w /res_source_w)

#..............................................................................

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

def compute_GT_mask (label,imID,imH,imW):      
    anncatind = cats_names.index(label)
    anncatID = cats_id[anncatind]
    annIds_imgs = coco.getAnnIds( imgIds=imID,catIds=anncatID, iscrowd=False)
    anns = coco.loadAnns(annIds_imgs)
    mask_annitem = numpy.zeros([imH,imW])
    for item in anns: # constructing true mask by ading all mask items
        mask_temp = coco.annToMask(item )
        mask_annitem = mask_annitem + mask_temp
           
    return mask_annitem, anncatind

def process_GT_mask(mask_annitem_in):
    mask_annitem = cv2.resize(mask_annitem_in, (res_target_w,res_target_h))
    mask_annitem_out = 1*(mask_annitem>=0.5)  
    return mask_annitem_out

def upsample_3D (input_tensor,scale_T , scale_H, scale_W):
    tensor_detected_uptime = numpy.repeat(input_tensor,scale_T, axis=0)
    output_tensor = numpy.repeat (numpy.repeat(tensor_detected_uptime,scale_W, axis=2)  , scale_H , axis=1)
    return output_tensor

def normalize_across_pixels (input_mask):
    norm_factor = numpy.sum(input_mask)
    output_mask = input_mask/ norm_factor
    return output_mask

def normalize_across_time(input_tensor):
    norm_factor = numpy.sum(input_tensor)
    output_tensor = input_tensor/ norm_factor
    return output_tensor

def compute_sa_score(mask_1, mask_2):
    mask_overlap = mask_1*mask_2
    sa_score = round ( numpy.sum(mask_overlap) , 3)
    return sa_score


def compute_spatial_alignment (t_on, t_off,tensor_AV_image,sub_noun, imID):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    mask_av_subnoun = numpy.sum(tensor_av_subnoun , axis = 0)
    mask_av_subnoun_norm = normalize_across_pixels (mask_av_subnoun)
    mask_GT, label_id = compute_GT_mask (sub_noun,imID,imH,imW) 
    mask_GT = process_GT_mask (mask_GT)
    score_sa = compute_sa_score(mask_GT, mask_av_subnoun_norm)
    s_GT = numpy.sum(mask_GT)
    return score_sa, s_GT , label_id

def compute_temporal_alignment (t_on, t_off,tensor_AV_image, sub_noun, imID):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    mask_GT, label_id = compute_GT_mask (sub_noun,imID,imH,imW) 
    mask_GT = process_GT_mask (mask_GT) 
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis , : , : ], 512, axis = 0)
    tensor_av_subnoun = tensor_AV_image * tensor_GT
    mask_av_subnoun = numpy.sum(tensor_av_subnoun , axis = (1,2))    
    mask_av_norm = normalize_across_time (mask_av_subnoun)
    score_ta = numpy.sum ( mask_av_norm[t_on:t_off]  )
    score_ta = round(score_ta, 3)
    return score_ta
                
#.............................................................................. loading input files

data = scipy.io.loadmat(path_in_corrected_ind + file_in_corrected_ind , variable_names = ['ind_accepted'])
ind_accepted = data['ind_accepted'][0]
number_of_images = len(ind_accepted)
#.....................

data = scipy.io.loadmat(path_in_AVtensor + file_in_AVtensor + '.mat'  , variable_names = ['out_layer'])
tensor_input = data['out_layer']
tensor_input = tensor_input [ind_accepted]
tensor_input = tensor_input    
#.....................

data = scipy.io.loadmat(path_in_metadata + file_in_metadata, variable_names=['list_of_images','image_id_all'])
all_image_ids = data ['image_id_all'][0]
all_image_ids = [ item[0].strip() for item in all_image_ids] 

#.....................

data = scipy.io.loadmat(path_in_processed_nouns + file_in_processed_nouns, variable_names = ['all_accpted_words','all_accepted_ind','all_accpted_onsets','all_accpted_offsets'])
all_nouns = data ['all_accpted_words'][0]
all_inds = data ['all_accepted_ind'][0]
all_onsets = data ['all_accpted_onsets'][0]
all_offsets = data ['all_accpted_offsets'][0]
all_nouns = [ [subitem.strip() for subitem in imitem if subitem ] for imitem in all_nouns ]

#.....................

data = scipy.io.loadmat(path_in_labels + file_in_labels, variable_names =['ref_names_all'])
all_reference_names = data['ref_names_all'][0]
all_reference_names = [all_reference_names[item] for item in ind_accepted]
all_reference_names = [ [phoneme.strip() for phoneme in utterance] for utterance in all_reference_names ]

#.....................

from pycocotools.coco import COCO
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/worktmp/hxkhkh/data/coco/MSCOCO/'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)
cats = coco.loadCats(coco.getCatIds())
cats_id = [item['id'] for item in cats]
cats_names = [item['name']for item in cats]


#.............................................................................. output data

all_sa_scores =  numpy.zeros([80, 1])
all_ta_scores =  numpy.zeros([80, 1])
all_meta_info =  numpy.zeros([80, 3])

allrand_sa_scores =  numpy.zeros([80, 1])
allrand_ta_scores =  numpy.zeros([80, 1])
##############################################################################

                                    # MAIN
                                    
###############################################################################

tensor_input_SA = softmax_spatial(tensor_input)
tensor_input_SA = numpy.reshape(tensor_input_SA, [tensor_input.shape[0], res_source_t , res_source_h, res_source_w])

tensor_input_TA = softmax_temporal(tensor_input)
tensor_input_TA = numpy.reshape(tensor_input_TA, [tensor_input.shape[0], res_source_t , res_source_h, res_source_w])


for counter_image in range(number_of_images):
    
    print(' image ..........................' , str(counter_image))

    #..........................................................................
    imageID = int(all_image_ids[counter_image])
    
    tensor_in_sa  = tensor_input_SA[counter_image]
    tensor_in_sa = upsample_3D (tensor_in_sa,scale_t , scale_h, scale_w)
    
    tensor_in_ta  = tensor_input_TA[counter_image]
    tensor_in_ta = upsample_3D (tensor_in_ta, scale_t , scale_h, scale_w)
    
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
                
                score_sa, area_GT , col = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_in_sa, subnoun, imageID)
                #print(score_sa)
                
                scorerand_sa, arearand_GT , col = compute_spatial_alignment (t_onset_sa, t_offset_sa ,tensor_rand, subnoun, imageID)
                #print(scorerand_sa)
                #.............................................................................. Temporal Alignment
                t_onset = int(numpy.maximum (t_onsets_image[counter_label] - 50 , 0 ))
                t_offset = int(numpy.minimum (t_offsets_image[counter_label]+ 50 , 512))
                
                t_duration_extra = t_offset- t_onset
                
                score_ta = compute_temporal_alignment (t_onset, t_offset,tensor_in_ta, subnoun, imageID)
                #print(score_ta)
                
                scorerand_ta = compute_temporal_alignment (t_onset, t_offset, tensor_rand,subnoun, imageID)
                #print(scorerand_ta)
                
                #.............................................................................. Saving results
                
                all_ta_scores[col,0] = all_ta_scores[col,0] + score_ta
                all_sa_scores[col,0] = all_sa_scores[col,0] + score_sa
                
                all_meta_info [col,0] = all_meta_info [col,0] + area_GT
                all_meta_info [col,1] = all_meta_info [col,1] + t_duration
                all_meta_info [col,2] = all_meta_info [col,2] + 1
                
                allrand_ta_scores[col,0] = allrand_ta_scores[col,0] + scorerand_ta
                allrand_sa_scores[col,0] = allrand_sa_scores[col,0] + scorerand_sa
                
scipy.io.savemat(path_out + file_out + '.mat' , {'all_sa_scores':all_sa_scores, 'all_ta_scores':all_ta_scores, 
                                                 'allrand_ta_scores': allrand_ta_scores, 'allrand_sa_scores': allrand_sa_scores ,
                                                 'all_meta_info':all_meta_info})                
                
# from matplotlib import pyplot as plt
# plt.imshow(numpy.sum (tensor_in_ta[t_onset:t_offset , :, : ] , axis = 0 ))                
(t_on, t_off,tensor_AV_image, imID) = (t_onset_sa, t_offset_sa ,tensor_in_sa, imageID)