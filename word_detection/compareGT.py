###############################################################################
# This file computes GT timing and GT tensor for selected labels

###############################################################################
res_target_h = 224    
res_target_w = 224
res_target_t = 512

res_source_h = 14
res_source_w = 14
res_source_t = 64

import numpy
import scipy.io
import cv2


project_path = '/worktmp/hxkhkh/project2/'

path_in_corrected_ind = project_path + 'outputs/step_6/step_2/'
path_in_nouns = '/worktmp/hxkhkh/project2/outputs/step_6/step_1/'

path_in_processed_nouns = '/worktmp/hxkhkh/project2/outputs/step_6/step_5/'

path_metadata  = project_path + 'outputs/step_1/validation/'

path_in_labels = project_path +'outputs/step_6/step_4/'

path_out = project_path + 'outputs/step_8/step_0/'
file_out = 'GT_res224.mat'
###############################################################################
                        # 0. Loading spell check results #
###############################################################################

data = scipy.io.loadmat(path_in_corrected_ind + 'corrected_nouns_index.mat', variable_names = ['ind_accepted'])
ind_accepted = data['ind_accepted'][0]

###############################################################################
                        # 1. Loading validation image IDS #
###############################################################################

data = scipy.io.loadmat(path_metadata + 'processed_data_list_val.mat',
                                  variable_names=['list_of_images','image_id_all'])

image_id_all = data ['image_id_all'][0]  
        
temp = image_id_all
image_id_all = []
for item in temp:
    value = item[0].strip()
    image_id_all.append(value)
del temp


image_id_all = [image_id_all[item] for item in ind_accepted]


###############################################################################
                        # 2. all image labels #
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
                        # 1. Loading word timing data #
###############################################################################

data = scipy.io.loadmat(path_in_nouns + 'noun_data.mat', variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])

#wavfile_nouns = data ['wavfile_nouns'][0]
wavfile_nouns_onsets = data ['wavfile_nouns_onsets'][0]
wavfile_nouns_offsets = data ['wavfile_nouns_offsets'] [0]

obj = wavfile_nouns_onsets
wavfile_nouns_onsets = []
wavfile_nouns_onsets = [item for item in obj]  

obj = wavfile_nouns_offsets
wavfile_nouns_offsets = []
wavfile_nouns_offsets = [item for item in obj]  

#wavfile_nouns_corrected = [wavfile_nouns[item] for item in ind_accepted]
wavfile_onsets_corrected = [wavfile_nouns_onsets[item][0] for item in ind_accepted]
wavfile_offsets_corrected = [wavfile_nouns_offsets[item][0] for item in ind_accepted]


wavfile_nouns_onsets = [item/1 for item in wavfile_onsets_corrected]
wavfile_nouns_offsets = [item/1 for item in wavfile_offsets_corrected]


wavfile_nouns_onsets_sim = []
wavfile_nouns_offset_sim = []
all_time_masks = []

for counter_image in range(len(all_subinds)):
    #simulated_nouns = all_subnouns [counter_image]
    simulated_nouns_ind = all_subinds[counter_image]
    word_sim_on = []
    word_sim_off = []
   
    im_masks = []
    if len(simulated_nouns_ind): # if not empty
        #print(simulated_nouns_ind)
        simulated_nouns_ind = simulated_nouns_ind[0]
        caption_onsets =  numpy.asarray(wavfile_nouns_onsets[counter_image],dtype='int')
        caption_offsets =  numpy.asarray(wavfile_nouns_offsets[counter_image],dtype='int')
        
        for counter_candidates, ind_word in enumerate(simulated_nouns_ind):
            #print(ind_word)
            
            word_onset = caption_onsets[ind_word] - 0
            word_offset = caption_offsets[ind_word] + 0
            
            word_sim_on.append(word_onset)
            word_sim_off.append(word_offset)
            
            mask_word = numpy.zeros([512])
            mask_word[word_onset:word_offset]=1
            im_masks.append(mask_word)
            
    wavfile_nouns_onsets_sim.append(word_sim_on)
    wavfile_nouns_offset_sim.append(word_sim_off)
    all_time_masks.append((im_masks))

###############################################################################
                        # 4. Loading coco tool #
############################################################################### loading coco package
from pycocotools.coco import COCO

import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

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

# def upsample1 (tensor_in, scale_h,scale_w):
#     tensor_out= []    
#     for frame in tensor_in:
#         frame_upsampled = upsamplereshape_2D(frame,scale_h,scale_w )
#         tensor_out.append(frame_upsampled)
        
#     tensor_out = numpy.array(tensor_out)                 
#     tensor_out = numpy.sum(tensor_out, axis=0)    
#     return tensor_out

# def upsample2 (tensor_in, scale_h,scale_w):      
#     frame_sum = numpy.sum(tensor_in,axis = 0)      
#     tensor_out = upsamplereshape_2D(frame_sum,scale_h,scale_w )
#     tensor_out = numpy.array(tensor_out)                     
#     return tensor_out

# def upsample3 (tensor_in, scale_h,scale_w):
#     tensor_out= []    
#     for frame in tensor_in:
#         frame_upsampled = upsamplereshape_2D(frame,scale_h,scale_w )
#         tensor_out.append(frame_upsampled)
        
#     tensor_out = numpy.array(tensor_out)                     
#     return tensor_out
                                
# def upsamplereshape_2D (frame_in, scale_h,scale_w):
#     out_upsampled = numpy.reshape(frame_in,[res_source_h,res_source_w])
#     out_upsampled = numpy.repeat( (numpy.repeat(out_upsampled, scale_h, axis =0)), scale_w, axis=1)
#     if res_target_w!=224 or res_target_h!=224:
#         out_upsampled = cv2.resize(out_upsampled,(res_target_w,res_target_h))
#     #out_upsampled = numpy.reshape(out_upsampled, [res_target_h,res_target_w])
#     return out_upsampled

# def normalize_mask (mask_in):
#     norm_factor = numpy.sum(mask_in)
#     mask_norm = numpy.divide(mask_in,norm_factor)              
#     return mask_norm



            
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

# def select_best_mask(masks_detected_in,mask_GT):
#     x_TP_all = []
#     for mask_frame in masks_detected_in:
#         mask_tp_frame = 1 *  (mask_frame*mask_GT)                     
#         x_TP_all.append(numpy.sum(mask_tp_frame))

#     mask_selected_arg = numpy.argmax(x_TP_all)
#     mask_selected = masks_detected_in[mask_selected_arg]
#     return mask_selected

###############################################################################
###############################################################################
  

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
    
    
    ref_names = refnames_all[counter_image]
    
    if len(subinds): 
        
        subind = subinds[0]        
        for counter_label in range(len(subnouns)): # loop over WI items
            
            label_result = []           
            label_caption = subnouns[counter_label] # chair or # dining table
            #.............................................................. to compute GT mask
                
            if label_caption in ref_names: # first detect TP 
                
                mask_annitem, col = compute_GT_mask (label_caption,imID,imH,imW) 
                mask_annitem = process_GT_mask (mask_annitem)
                
                mask_time = all_time_masks[counter_image][counter_label]
                
                # if 1D
                tensor_annitem = numpy.repeat(mask_annitem[:, numpy.newaxis], res_target_t, axis=1)
                tensor_time = numpy.repeat(mask_time[numpy.newaxis ,  : ], res_target_h*res_target_h, axis=0)
                
                #if 2D
                # tensor_annitem = numpy.repeat(mask_annitem[:, :, numpy.newaxis], res_target_t, axis=2) # If 2D
                # tensor_time = numpy.repeat(mask_time[numpy.newaxis, :  ], res_target_w, axis=0)
                # tensor_time = numpy.repeat(tensor_time[numpy.newaxis ,:,  : ], res_target_h, axis=0)
                
                tensor_GT = tensor_annitem*tensor_time
                counter_n += 1                   
                
                
#tensor_GT = numpy.reshape(tensor_GT,[224,224,512])               
# import matplotlib.pyplot as plt
# plt.imshow(mask_annitem)
# plt.imshow(tensor_GT[:,:,240])
                
#scipy.io.savemat(path_out + file_out , { 'word_sim_off':word_sim_off,'word_sim_off':word_sim_off})   
