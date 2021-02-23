import os
import numpy
import scipy.io
import cv2
import random
###############################################################################
# this file first loads test data nouns + detected image masks 
# next it compares the detected objects (within AV tensor) with the correct object masks
###############################################################################
# Note:
    # cv2 reading image output, dim = (height, width)
    # cv2 resizing image input, dim = (width, height)
     
res_target_h = 224    
res_target_w = 224
res_target_t = 512

res_source_h = 14
res_source_w = 14
res_source_t = 64


path_project = '/worktmp/hxkhkh/project2/'


path_in_corrected_ind = os.path.join(path_project , 'outputs/step_6/step_2/')
path_in_labels = os.path.join(path_project , 'outputs/step_6/step_4/')

path_in_processed_nouns = os.path.join(path_project , 'outputs/step_6/step_5/')
path_in_metadata  = os.path.join(path_project , 'outputs/step_1/validation/')

path_in_AVtensor_masks = os.path.join(path_project , 'outputs/step_7/step_1/')
file_in_AVtensor_masks = 'sub_labels_masks_SI'

out_path = os.path.join(path_project , 'outputs/step_7/step_2/')
file_out = 'info_SI_res224_softmax_best.mat'

flag_threshols = 1
threshmask = 1e-5

flag_shift = 0
###############################################################################
                        # 0. Loading coco tool #
############################################################################### loading coco package
from pycocotools.coco import COCO

dataDir='/worktmp/hxkhkh/data/coco/MSCOCO/'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)

cats = coco.loadCats(coco.getCatIds())
cats_id = [item['id'] for item in cats]
cats_names = [item['name']for item in cats]


###############################################################################
                        # 1. Loading validation image IDS #
###############################################################################

data = scipy.io.loadmat(path_in_metadata + 'processed_data_list_val.mat',
                                  variable_names=['list_of_images','image_id_all'])

image_id_all = data ['image_id_all'][0]  
        
temp = image_id_all
image_id_all = []
for item in temp:
    value = item[0].strip()
    image_id_all.append(value)
del temp

# correct caption indexes
data = scipy.io.loadmat(path_in_corrected_ind + 'corrected_nouns_index.mat', variable_names = ['ind_accepted'])
ind_accepted = data['ind_accepted'][0]
image_id_all = [image_id_all[item] for item in ind_accepted]
###############################################################################
                        # 3. loading substitue nouns and coressponding indexes #
###############################################################################

data = scipy.io.loadmat(path_in_processed_nouns + 'sub_labels.mat', variable_names = ['all_accpted_words','all_accepted_ind'])

all_subnouns = data ['all_accpted_words'][0]
all_subinds = data ['all_accepted_ind'][0]
all_labels = []

temp = all_subnouns
all_subnouns = []
for imitem in temp:
    subnouns_im = []
    for subitem in imitem:
        if subitem:
            correct_noun = subitem.strip()
            subnouns_im.append(correct_noun)
            all_labels.append(correct_noun)
    all_subnouns.append(subnouns_im)        
del temp

###############################################################################
                        # 3. loading detected masks #
###############################################################################

def softmax(input_vec):
    """Compute softmax values for each sets of scores in x."""
    output_vec = []
    for item in input_vec:
        e_x = numpy.exp(item - numpy.max(item))
        output_vec.append(e_x / e_x.sum())
    return output_vec

import pickle
filename = path_in_AVtensor_masks + file_in_AVtensor_masks
file = open(filename,'rb')
all_word_detection_masks = pickle.load(file)  


temp = all_word_detection_masks
all_word_detection_masks = []
all_word_detection_randmasks = []



for counter_image , image in enumerate(temp):
    image_item = []
    for  maskitem in image:       
        if len(maskitem): 
            maskitem = softmax(maskitem)
            image_item.append(numpy.array(maskitem))           
    all_word_detection_masks.append(image_item)
    
del temp


###############################################################################
                        # 4. loading unified image labels #
############################################################################### 
data = scipy.io.loadmat(path_in_labels + 'unified_labels.mat', variable_names =['ref_names_all'])

refnames_all = data['ref_names_all'][0]

refnames_all = [refnames_all[item] for item in ind_accepted]


temp = refnames_all
refnames_all = []
for utterance in temp:
    correct_utterance = []
    for phoneme in utterance:
        correct_ph  = phoneme.strip()
        correct_utterance.append(correct_ph)
    refnames_all.append(correct_utterance)      

###############################################################################
def upsample1 (tensor_in, scale_h,scale_w):
    tensor_out= []    
    for frame in tensor_in:
        frame_upsampled = upsamplereshape_2D(frame,scale_h,scale_w )
        tensor_out.append(frame_upsampled)
        
    tensor_out = numpy.array(tensor_out)                 
    tensor_out = numpy.sum(tensor_out, axis=0)    
    return tensor_out

def upsample2 (tensor_in, scale_h,scale_w):      
    frame_sum = numpy.sum(tensor_in,axis = 0)      
    tensor_out = upsamplereshape_2D(frame_sum,scale_h,scale_w )
    tensor_out = numpy.array(tensor_out)                     
    return tensor_out

def upsample3 (tensor_in, scale_h,scale_w):
    tensor_out= []    
    for frame in tensor_in:
        frame_upsampled = upsamplereshape_2D(frame,scale_h,scale_w )
        tensor_out.append(frame_upsampled)
        
    tensor_out = numpy.array(tensor_out)                     
    return tensor_out
                                
def upsamplereshape_2D (frame_in, scale_h,scale_w):
    out_upsampled = numpy.reshape(frame_in,[res_source_h,res_source_w])
    out_upsampled = numpy.repeat( (numpy.repeat(out_upsampled, scale_h, axis =0)), scale_w, axis=1)
    if res_target_w!=224 or res_target_h!=224:
        out_upsampled = cv2.resize(out_upsampled,(res_target_w,res_target_h))
    #out_upsampled = numpy.reshape(out_upsampled, [res_target_h,res_target_w])
    return out_upsampled

def normalize_mask (mask_in):
    norm_factor = numpy.sum(mask_in)
    mask_norm = numpy.divide(mask_in,norm_factor)              
    return mask_norm



            
def compute_GT_mask (label,imID,imH,imW):      
    anncatind = cats_names.index(label)
    anncatID = cats_id[anncatind]
    annIds_imgs = coco.getAnnIds( imgIds=imID,catIds=anncatID, iscrowd=False)
    anns = coco.loadAnns(annIds_imgs)
    mask_annitem = numpy.zeros([imH,imW])
    for item in anns: # constructing true mask by ading all mask items
        mask_temp = coco.annToMask(item )
        mask_annitem =mask_annitem + mask_temp
           
    return mask_annitem, anncatind

def process_GT_mask(mask_annitem):
    mask_annitem = cv2.resize(mask_annitem, (res_target_w,res_target_h))
    mask_annitem = 1*(mask_annitem>=0.5)
    mask_annitem_final = numpy.reshape(mask_annitem , -1)
    return mask_annitem_final

def select_best_mask(masks_detected_in,mask_GT):
    x_TP_all = []
    for mask_frame in masks_detected_in:
        mask_tp_frame = 1 *  (mask_frame*mask_GT)                     
        x_TP_all.append(numpy.sum(mask_tp_frame))

    mask_selected_arg = numpy.argmax(x_TP_all)
    mask_selected = masks_detected_in[mask_selected_arg]
    return mask_selected

###############################################################################
###############################################################################
   
#all_info = numpy.zeros([len(all_subinds), 3])
all_names = ['x','s_mask' , 'x_FN' , 'x_FN_hits','number_of_repeats']
all_labels_info = numpy.zeros([80, 5])
cm_detected = numpy.zeros([81,80])

counter_n = 0
for counter_image in range(len(image_id_all)):
    
    print('........counter image ....' , str(counter_image))

    #..........................................................................
    imID = int(image_id_all[counter_image])
    img = coco.loadImgs(imID)[0]
    imID = img['id']
    imW = img['width']
    imH = img['height']
    im_name = img ['file_name']
    #s_image = imW*imH
    
    # res_target_h = imH
    # res_target_w = imW
    #..........................................................................
    
    subinds = all_subinds[counter_image]
    subnouns = all_subnouns[counter_image]
    masks_detected_im = all_word_detection_masks[counter_image]
     
    
    ref_names = refnames_all[counter_image]
    
    image_overlaps = []
    image_results = []
    
    if len(subinds): 
        
        subind = subinds[0]        
        for counter_label in range(len(masks_detected_im)): # loop over WI items
            
            label_result = []           
            label_caption = subnouns[counter_label] # chair or # dining table
            mask_detected_label = masks_detected_im[counter_label]
            n_sub_images = numpy.shape(mask_detected_label)[0]
           
            if n_sub_images>0:                
                #.............................................................. to compute Detection mask
                if flag_shift:
                    min_tensor = numpy.min(numpy.reshape(mask_detected_label, [-1]))
                    min_tensor= abs(min_tensor)
                    shift_tensor = numpy.tile(min_tensor,[numpy.shape(mask_detected_label)[0],numpy.shape(mask_detected_label)[1]])
                    mask_detected_label =  mask_detected_label + shift_tensor
                    
                if flag_threshols:                   
                    mask_th = 1*(mask_detected_label> threshmask) 
                    mask_detected_label = mask_detected_label * mask_th
                    
                # mask_detected_label_sum = numpy.reshape(numpy.sum(mask_detected_label, axis=0),[14,14]) 
                # mask_detected_label_sum = normalize_mask (mask_detected_label_sum) 
                mask_detected_upsampled = upsample3(mask_detected_label, int(res_target_h /res_source_h) , int(res_target_w /res_source_w)) 
                
                mask_detected_norm = []
                for frame in mask_detected_upsampled:
                    frame_norm = normalize_mask (frame)
                    mask_detected_norm.append(frame_norm)
                mask_detected_norm = numpy.array(mask_detected_norm)
                
                mask_detected_final = numpy.reshape(mask_detected_norm,[n_sub_images,-1])
                #mask_detected_final = numpy.reshape(mask_detected_final ,[224,224])
                labelcatind = cats_names.index(label_caption) # 56 for chair
                labelcatID = cats_id[labelcatind] # ID=62 for chair
                
                #.............................................................. to compute GT mask
                
                if label_caption in ref_names: # first detect TP 
                    
                    mask_annitem, col = compute_GT_mask (label_caption,imID,imH,imW)                
                    mask_annitem_final = process_GT_mask (mask_annitem)                   
                    area_GT_mask = numpy.sum(mask_annitem_final)
                        
                    mask_selected_final = select_best_mask(mask_detected_final,mask_annitem_final)
                    
                    #mask_selected_final = numpy.mean(mask_detected_final, axis= 0)
                    
                    mask_tp = 1 *  (mask_selected_final*mask_annitem_final)
                    x_TP = numpy.sum(mask_tp) #s_overlap_maskWI                    
                    mask_null = numpy.ones(res_target_h*res_target_w) - (1* mask_tp >0)
                    
                    
                    x_TP = round(x_TP, 3) 
                    #print(x_TP)
                    if (x_TP > 0):
                        cm_detected[col,col] = cm_detected[col,col] + 1
                    
                    mask_fn = mask_annitem_final - mask_tp * (1* mask_tp >0)
                    x_FN = numpy.sum(mask_fn)
                    
                    
                    all_labels_info[col,0] = all_labels_info[col,0] + x_TP
                    all_labels_info[col,1] = all_labels_info[col,1] + area_GT_mask
                    all_labels_info[col,2] = all_labels_info[col,2] + x_FN
                    if x_TP==0:
                        all_labels_info[col,3] = all_labels_info[col,3] + 1                        
                    all_labels_info[col,4] = all_labels_info[col,4] + 1    

                    counter_n += 1                   
                    ref_names.remove(label_caption)  
                    
                    for annitem in ref_names: # next detect FP 
                    
                        mask_annitem, row = compute_GT_mask (annitem,imID,imH,imW)
                        mask_annitem_final = process_GT_mask (mask_annitem)
                        mask_annitem_final = mask_null * mask_annitem_final
                        
                        mask_fp = 1 *  (mask_selected_final*mask_annitem_final)
                        
                        mask_null = mask_null - (1* mask_fp >0)
                        
                        # now use null version of mask item instead of maks item                            
                        x_fp = numpy.sum(mask_selected_final*mask_annitem_final)
                        
                        if (x_fp > 0):
                            cm_detected[row,col] = cm_detected[row,col] + 1
                            
                    mask_other = mask_selected_final*mask_null
                    x_fp_other = numpy.sum(mask_other)
                    if (x_fp_other > 0):
                        cm_detected[80,col] = cm_detected[80,col] + 1
                
scipy.io.savemat(out_path + file_out , { 'all_names':all_names,'all_labels_info':all_labels_info, 'cm_detected':cm_detected})
