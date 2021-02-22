import os
import numpy
import scipy.io
import cv2
import random
###############################################################################
# this file first loads test data nouns + detected image masks 
# next it compares the detected objects (within AV tensor) with the correct object masks
###############################################################################

res_p_target = 224
res_t_target = 64
res_p_source = 14

project_path = '/worktmp/hxkhkh/project2/'

in_path_corrected_ind = os.path.join(project_path , 'outputs/step_6/step_2/')
in_path_labels = os.path.join(project_path , 'outputs/step_6/step_4/')

in_path_processed_nouns = os.path.join(project_path , 'outputs/step_6/step_5/')
metadata_path  = os.path.join(project_path , 'outputs/step_1/validation/')

in_path_AVtensor_masks = os.path.join(project_path , 'outputs/step_7/step_1/')

out_path = os.path.join(project_path , 'outputs/step_7/step_2/')

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

data = scipy.io.loadmat(metadata_path + 'processed_data_list_val.mat',
                                  variable_names=['list_of_images','image_id_all'])

image_id_all = data ['image_id_all'][0]  
        
temp = image_id_all
image_id_all = []
for item in temp:
    value = item[0].strip()
    image_id_all.append(value)
del temp

# correct caption indexes
data = scipy.io.loadmat(in_path_corrected_ind + 'corrected_nouns_index.mat', variable_names = ['ind_accepted'])
ind_accepted = data['ind_accepted'][0]
image_id_all = [image_id_all[item] for item in ind_accepted]
###############################################################################
                        # 3. loading substitue nouns and coressponding indexes #
###############################################################################

data = scipy.io.loadmat(in_path_processed_nouns + 'sub_labels.mat', variable_names = ['all_accpted_words','all_accepted_ind'])

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

import pickle
filename = in_path_AVtensor_masks + 'sub_labels_masks'
file = open(filename,'rb')
all_word_detection_masks = pickle.load(file)  


temp = all_word_detection_masks
all_word_detection_masks = []
all_word_detection_randmasks = []

threshmask = 1e-5

for counter_image , image in enumerate(temp):
    image_item = []
    image_item_rand = []
    for  maskitem in image:       
        if len(maskitem):
            maskitem_th = 1.0*(maskitem> threshmask)   
            image_item.append(maskitem_th)
         
            sum_item = int(numpy.sum(maskitem_th))           
            rand_mask_object = numpy.zeros((res_p_source*res_p_source))
            for rand_hit in range(sum_item):
                rand_mask_object[random.randint(0,(res_p_source*res_p_source - 1) )] = 1
            image_item_rand.append(rand_mask_object)
    all_word_detection_masks.append(image_item)
    all_word_detection_randmasks.append(image_item_rand)
del temp


###############################################################################
                        # 4. loading unified image labels #
############################################################################### 
data = scipy.io.loadmat(in_path_labels + 'unified_labels.mat', variable_names =['ref_names_all'])

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
def upsamplereshape_2D (output_layer, upsample_factor):
    out_upsampled = numpy.reshape(output_layer,[14,14])
    out_upsampled = numpy.repeat( (numpy.repeat(out_upsampled, upsample_factor, axis =0)), upsample_factor, axis=1)
    out_upsampled = numpy.reshape(out_upsampled, [res_p_target,res_p_target])
    return out_upsampled

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
    mask_annitem = cv2.resize(mask_annitem, (res_p_target,res_p_target))
    mask_annitem = 1*(mask_annitem>=0.5)
    mask_annitem_final = numpy.reshape(mask_annitem , -1)
    return mask_annitem_final

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
                
                mask_detected_upsampled = []
                for frame in mask_detected_label:
                    frame_upsampled = upsamplereshape_2D(frame,16 )
                    mask_detected_upsampled.append(frame_upsampled)
                    
                mask_detected_upsampled = numpy.array(mask_detected_upsampled)                 
                mask_detected_sumtime = numpy.mean(mask_detected_upsampled, axis=0)
                norm_factor = numpy.sum(mask_detected_sumtime)
                mask_detected_norm = numpy.divide(mask_detected_sumtime,norm_factor)
                
                mask_detected_final = numpy.reshape(mask_detected_norm,[-1])
                area_mask_detected = numpy.sum(mask_detected_final)
                
                labelcatind = cats_names.index(label_caption) # 56 for chair
                labelcatID = cats_id[labelcatind] # ID=62 for chair
                
                
                #.............................................................. to compute GT mask
                
                if label_caption in ref_names: # first dtect TP and discartd cells 
                    
                    mask_annitem, col = compute_GT_mask (label_caption,imID,imH,imW)                
                    mask_annitem_final = process_GT_mask (mask_annitem)                   
                    area_GT_mask = numpy.sum(mask_annitem_final)
              
                    mask_tp = 1 *  (mask_detected_final*mask_annitem_final)
                    mask_null = numpy.ones(res_p_target*res_p_target) - (1* mask_tp >0)
                    
                    
                    x_TP = numpy.sum(mask_tp) #s_overlap_maskWI
                    x_TP = round(x_TP, 3) 
                    #print(x_TP)
                    
                    mask_fn = mask_annitem_final - (1* mask_tp >0)
                    x_FN = numpy.sum(mask_fn)
                    
                    
                    all_labels_info[col,0] = all_labels_info[col,0] + x_TP
                    all_labels_info[col,1] = all_labels_info[col,1] + area_GT_mask
                    all_labels_info[col,2] = all_labels_info[col,2] + x_FN
                    if x_TP==0:
                        all_labels_info[col,3] = all_labels_info[col,3] + 1                        
                    all_labels_info[col,4] = all_labels_info[col,4] + 1    

                    counter_n += 1
                    
                    if (x_TP > 0):
                        cm_detected[col,col] = cm_detected[col,col] + 1
                                             
                    ref_names.remove(label_caption)  
                    
                    for annitem in ref_names: # next detect FP based on remaining cells (loop over other image objects)
                    
                        mask_annitem, row = compute_GT_mask (annitem,imID,imH,imW)
                        mask_annitem_final = process_GT_mask (mask_annitem)
                        mask_annitem_final = mask_null * mask_annitem_final
                        
                        mask_fp = 1 *  (mask_detected_final*mask_annitem_final)
                        
                        mask_null = mask_null - (1* mask_fp >0)
                        
                        # now use null version of mask item instead of maks item                            
                        x_fp = numpy.sum(mask_detected_final*mask_annitem_final)
                        
                        if (x_fp > 0):
                            cm_detected[row,col] = cm_detected[row,col] + 1
                            
                    mask_other = mask_detected_final*mask_null
                    x_fp_other = numpy.sum(mask_other)
                    if (x_fp_other > 0):
                        cm_detected[80,col] = cm_detected[80,col] + 1
                
scipy.io.savemat(out_path + 'all_info_WI_norm_updated2.mat', { 'all_names':all_names,'all_labels_info':all_labels_info, 'cm_detected':cm_detected})
