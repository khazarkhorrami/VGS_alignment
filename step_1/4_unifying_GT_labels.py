

# this file reads coco annotations for set of test data, and unifies masks for shared object categories
  
###############################################################################
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.6
# sess = tf.Session(config=config) 

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
###############################################################################

import numpy
import scipy.io

metadata_path  = '/worktmp/khorrami/work/projects/project_1/outputs/step_1/validation/'
in_path_corrected_ind = '/worktmp/khorrami/work/projects/project_2/outputs/step_6/step_2/'
out_path = '/worktmp/khorrami/work/projects/project_2/outputs/step_6/step_4/'
###############################################################################
                        # 1. Loading Meta Data #
###############################################################################

data = scipy.io.loadmat(metadata_path + 'processed_data_list.mat',
                                  variable_names=['list_of_images','image_id_all'])


image_id_all = data ['image_id_all'][0]  
        
temp = image_id_all
image_id_all = []
for item in temp:
    value = item[0].strip()
    image_id_all.append(value)
del temp


###############################################################################
###############################################################################
                        # 2. coco annotation #
###############################################################################

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/worktmp/khorrami/work/data/MSCOCO/'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)

cats = coco.loadCats(coco.getCatIds())

###############################################################################
                        # 3. unifying annotations # 
############################################################################### running the loop for object IDs, act names, and supercat names

ref_names_all = []
#ref_masks_all = []
ref_IDs_all = []
ref_super_all = []

for image_counter in range(len(image_id_all)):
    print(image_counter)
    image_id = int(image_id_all[image_counter])
    #img = coco.loadImgs(image_id)[0]
    annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
    anns_image = coco.loadAnns(annId_img)
    
    ref_names = []
    #ref_masks = []
    ref_IDs = []
    ref_super = []
    
    for item in range(len(anns_image)):
        
        item_catId = anns_image[item]['category_id']
        item_catinfo = coco.loadCats(item_catId)[0]
        
        item_catName = item_catinfo['name']
        item_supercatName  = item_catinfo['supercategory']
        
        #item_mask = coco.annToMask(anns_image[item]) #
        
        if item_catName in ref_names:
            item_arg_ref = ref_names.index(item_catName)
            #ref_masks[item_arg_ref] = ref_masks[item_arg_ref] + item_mask
            #ref_masks[item_arg_ref] = 1* (ref_masks[item_arg_ref] > 0)
        else:
            ref_names.append(item_catName)
            #ref_masks.append(item_mask)
            ref_IDs.append(item_catId)
            ref_super.append(item_supercatName)
            
        #print('{}: {}'.format(item,item_catName ))
    
    ref_names_all.append(ref_names)
    #ref_masks_all.append(ref_masks)
    ref_IDs_all.append(ref_IDs)
    ref_super_all.append(ref_super)
    
    del anns_image
    del annId_img
    #del ref_masks
    del ref_names
    del ref_super
    
    # for item_ann in ref_names:
    #     print('............................')
    #     print(item_ann)
    #     print('............................')

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, saving the results


scipy.io.savemat(out_path + 'unified_labels.mat', {'ref_names_all':ref_names_all
                                                    ,'ref_IDs_all':ref_IDs_all
                                                    ,'ref_super_all':ref_super_all})


