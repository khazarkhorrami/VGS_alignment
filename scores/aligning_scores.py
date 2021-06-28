
from prepare_tensors import get_input_vectors, initialize_output_vectors, prepare_all_tensors, prepare_item_tensors,save_results
import numpy
import cv2
import os

# global variables 

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



def compute_GT_mask (label, imID, imH, imW , res_target_w , res_target_h):      
    anncatind = cats_names.index(label)
    anncatID = cats_id[anncatind]
    annIds_imgs = coco.getAnnIds( imgIds=imID,catIds=anncatID, iscrowd=False)
    anns = coco.loadAnns(annIds_imgs)
    mask_annitem = numpy.zeros([imH,imW])
    for item in anns: # constructing true mask by ading all mask items
        mask_temp = coco.annToMask(item )
        mask_annitem = mask_annitem + mask_temp
    mask_annitem = cv2.resize(mask_annitem, (res_target_w,res_target_h))       
    return mask_annitem, anncatind


def normalize_across_pixels_G (input_mask):
    norm_factor = numpy.sum(input_mask)
    output_mask = input_mask/ norm_factor
    return output_mask

def normalize_across_time_G(input_tensor):
    norm_factor = numpy.sum(input_tensor)
    output_tensor = input_tensor/ norm_factor
    return output_tensor

def object_GS(mask_1, mask_2):
    mask_overlap = mask_1*mask_2
    sa_score = round ( numpy.sum(mask_overlap) , 3)
    return sa_score


def compute_GS_object (t_on, t_off,res_target_w , res_target_h, tensor_AV_image,label, imID , mask_null):
    
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    mask_GT, label_id = compute_GT_mask (label, imID, imH, imW , res_target_w , res_target_h)
    mask_GT = 1*(mask_GT>=0.01) # very low thresh leads to many overlapping pixels between objects    
    s_GT = numpy.sum(mask_GT)
    mask_GT = mask_GT * mask_null
    mask_null = numpy.ones([res_target_h,res_target_w]) - (1* mask_GT >0)
    
                   
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    mask_av_subnoun = numpy.sum(tensor_av_subnoun , axis = 0)
    mask_av_subnoun_norm = normalize_across_pixels_G (mask_av_subnoun)   
    score_sa = object_GS(mask_GT, mask_av_subnoun_norm)   
    return score_sa, s_GT , label_id , mask_null

def compute_GS_word (t_on, t_off, res_target_w , res_target_h,tensor_AV_image, label, imID):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    mask_GT, label_id = compute_GT_mask (label, imID, imH, imW , res_target_w , res_target_h)
    mask_GT = 1*(mask_GT>=0.01)    
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis , : , : ], 512, axis = 0)
    
    tensor_av_subnoun = tensor_AV_image * tensor_GT
    mask_av_subnoun = numpy.sum(tensor_av_subnoun , axis = (1,2))    
    mask_av_norm = normalize_across_time_G (mask_av_subnoun)
    score_ta = numpy.sum ( mask_av_norm[t_on:t_off]  )
    score_ta = round(score_ta, 3)
    return score_ta

def compute_GS_object_other(t_on, t_off , res_target_w , res_target_h, tensor_AV_image, mask_null_other):  
    mask_GT = numpy.ones([res_target_h,res_target_w]) - (1* mask_null_other >0)
    s_GT = numpy.sum( mask_GT)
    #print(s_GT)                   
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    mask_av_subnoun = numpy.sum(tensor_av_subnoun , axis = 0)
    mask_av_subnoun_norm = normalize_across_pixels_G (mask_av_subnoun)   
    score_sa = object_GS(mask_GT, mask_av_subnoun_norm)   
    return score_sa, s_GT 


def normalize_across_pixels_A (input_tensor):
    norm_factor = numpy.sum(input_tensor, axis =(1,2) )
    norm_tensor = numpy.repeat(norm_factor[ : ,numpy.newaxis], input_tensor.shape[1], axis=1)
    norm_tensor = numpy.repeat(norm_tensor[:,  : , numpy.newaxis ], input_tensor.shape[2], axis=2)
    output_tensor = input_tensor/ norm_tensor
    return output_tensor


def normalize_across_time_A(input_tensor):
    norm_factor = numpy.sum(input_tensor, axis = 0 )
    norm_tensor = numpy.repeat(norm_factor[numpy.newaxis , : , :], input_tensor.shape[0], axis=0)
    output_tensor = input_tensor/ norm_tensor
    return output_tensor


def object_AS(tensor_1, tensor_2):
    tensor_overlap = tensor_1*tensor_2
    sa_score_frames = numpy.sum(tensor_overlap , axis = (1,2))
    sa_score= round(numpy.mean(sa_score_frames) , 3)
    return sa_score

def compute_ta_score_A(tensor_1, tensor_2):
    tensor_overlap = tensor_1*tensor_2
    ta_score_pixels = numpy.sum(tensor_overlap , axis = 0)
    ta_score= round(numpy.mean(ta_score_pixels) , 3)
    return ta_score

def compute_AS_object (t_on, t_off, res_target_w , res_target_h , tensor_AV_image, label, imID, mask_null):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    mask_GT, label_id = compute_GT_mask (label, imID, imH, imW , res_target_w , res_target_h)
    mask_GT = 1*(mask_GT>=0.01)
    s_GT = numpy.sum(mask_GT)
    mask_GT = mask_GT * mask_null
    mask_null = numpy.ones([res_target_h,res_target_w]) - (1* mask_GT >0)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
       
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    tensor_av_subnoun_norm = normalize_across_pixels_A (tensor_av_subnoun)
    score_sa = object_AS(tensor_GT, tensor_av_subnoun_norm)
    
    return score_sa, s_GT , label_id, mask_null

def compute_AS_word (t_on, t_off,tensor_AV_image, label, imID):
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']   
    mask_GT, label_id = compute_GT_mask (label,imID,imH,imW)
    mask_GT = 1*(mask_GT>=0.01)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
    
    tensor_av_norm = normalize_across_time_A (tensor_AV_image)
    tensor_av_subnoun = tensor_av_norm[t_on:t_off,:,:]     
    score_ta = compute_ta_score_A(tensor_GT, tensor_av_subnoun)
    return score_ta

def compute_AS_object_other (t_on, t_off ,res_target_w , res_target_h, tensor_AV_image, mask_null_other):  
    mask_GT = numpy.ones([res_target_h,res_target_w]) - (1* mask_null_other >0)
    s_GT = numpy.sum( mask_GT)
    n_frames = t_off - t_on
    tensor_GT = numpy.repeat(mask_GT[numpy.newaxis,:,:],n_frames , axis =0 )
    #print(s_GT)                   
    tensor_av_subnoun = tensor_AV_image[t_on:t_off,:,:]
    tensor_av_subnoun_norm = normalize_across_pixels_A (tensor_av_subnoun)
    score_sa = object_AS(tensor_GT, tensor_av_subnoun_norm)
    return score_sa, s_GT                


