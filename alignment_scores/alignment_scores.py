# This file takes input tensot T(time*h*w) and computes 2 alignments scores
# 1. object detection alignment (how well the model can locate the object within image in a given time window )
# 2. word detection alignment (how well the model can locate the word within spoken utterance at given object mask)


#import pickle
import numpy
import scipy.io
import cv2
import os

path_project = '/'


path_in_AVtensor = os.path.join(path_project , 'tensors/.../')
file_in_AVtensor = '...'

path_out = os.path.join(path_project , '../')
file_out = 'aligning_' + file_in_AVtensor 


path_in_metadata  = os.path.join(path_project , '../coco_validation/')
file_in_metadata = 'processed_data_list_val.mat'

path_in_corrected_ind = os.path.join(path_project , '../')
file_in_corrected_ind = 'corrected_nouns_index.mat'

path_in_labels = os.path.join(path_project , '../')
file_in_labels = 'unified_labels.mat'

path_in_processed_nouns = os.path.join(path_project , '../')
file_in_processed_nouns = 'sub_labels.mat'


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



def compute_GT_mask (label,imID,imH,imW):      
    anncatind = cats_names.index(label)
    anncatID = cats_id[anncatind]
    annIds_imgs = coco.getAnnIds( imgIds=imID,catIds=anncatID, iscrowd=False)
    anns = coco.loadAnns(annIds_imgs)
    mask_annitem = numpy.zeros([imH,imW])
    for item in anns: # constructing true mask by ading all mask items
        mask_temp = coco.annToMask(item )
        mask_annitem =mask_annitem + mask_temp 
    mask_annitem = cv2.resize(mask_annitem, (res_target_w,res_target_h))           
    return mask_annitem, anncatind




def normalize_across_pixels (input_tensor):
    norm_factor = numpy.sum(input_tensor, axis =(1,2) )
    norm_tensor = numpy.repeat(norm_factor[ : ,numpy.newaxis], input_tensor.shape[1], axis=1)
    norm_tensor = numpy.repeat(norm_tensor[:,  : , numpy.newaxis ], input_tensor.shape[2], axis=2)
    output_tensor = input_tensor/ norm_tensor
    return output_tensor

def normalize_across_time(input_tensor):
    norm_factor = numpy.sum(input_tensor, axis = 0 )
    norm_tensor = numpy.repeat(norm_factor[numpy.newaxis , : , :], input_tensor.shape[0], axis=0)
    output_tensor = input_tensor/ norm_tensor
    return output_tensor

def compute_sa_score(tensor_1, tensor_2):
    tensor_overlap = tensor_1*tensor_2
    sa_score_frames = numpy.sum(tensor_overlap , axis = (1,2))
    sa_score= round(numpy.mean(sa_score_frames) , 3)
    return sa_score

def compute_ta_score(tensor_1, tensor_2):
    tensor_overlap = tensor_1*tensor_2
    ta_score_pixels = numpy.sum(tensor_overlap , axis = 0)
    ta_score= round(numpy.mean(ta_score_pixels) , 3)
    return ta_score

def compute_spatial_alignment (t_on, t_off,tensor_AV_image, label, imID, mask_null):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    mask_GT, label_id = compute_GT_mask (label,imID,imH,imW)
    mask_GT = 1*(mask_GT>=0.01)
    s_GT = numpy.sum(mask_GT)
    mask_GT = mask_GT * mask_null
    mask_null = numpy.ones([res_target_h,res_target_w]) - (1* mask_GT >0)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
       
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    tensor_av_subnoun_norm = normalize_across_pixels (tensor_av_subnoun)
    score_sa = compute_sa_score(tensor_GT, tensor_av_subnoun_norm)
    
    return score_sa, s_GT , label_id, mask_null

def compute_temporal_alignment (t_on, t_off,tensor_AV_image, label, imID):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']   
    mask_GT, label_id = compute_GT_mask (label,imID,imH,imW)
    mask_GT = 1*(mask_GT>=0.01)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
    
    tensor_av_norm = normalize_across_time (tensor_AV_image)
    tensor_av_subnoun = tensor_av_norm[t_on:t_off,:,:]     
    score_ta = compute_ta_score(tensor_GT, tensor_av_subnoun)
    return score_ta

def compute_spatial_alignment_other (t_on, t_off ,tensor_AV_image, mask_null_other):  
    mask_GT = numpy.ones([res_target_h,res_target_w]) - (1* mask_null_other >0)
    s_GT = numpy.sum( mask_GT)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
    #print(s_GT)                   
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    tensor_av_subnoun_norm = normalize_across_pixels (tensor_av_subnoun)
    score_sa = compute_sa_score(tensor_GT, tensor_av_subnoun_norm)
    return score_sa, s_GT                

#.....................

from pycocotools.coco import COCO
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/.../data/coco/MSCOCO/'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)
cats = coco.loadCats(coco.getCatIds())
cats_id = [item['id'] for item in cats]
cats_names = [item['name']for item in cats]


#.............................................................................. output data

#.............................................................................. output data

all_sa_scores =  numpy.zeros([80, 1])
all_ta_scores =  numpy.zeros([80, 1])
all_meta_info =  numpy.zeros([80, 3])

allrand1_sa_scores =  numpy.zeros([80, 1])
allrand1_ta_scores =  numpy.zeros([80, 1])
allrand2_sa_scores =  numpy.zeros([80, 1])
allrand2_ta_scores =  numpy.zeros([80, 1])

cm_detection = numpy.zeros([81,80])
cm_rand1 = numpy.zeros([81,80])
cm_rand2 = numpy.zeros([81,80])
cm_object_area = numpy.zeros([81, 2])

nanfound = []
errorfound = []
##############################################################################


