
"""

This file does followings:
    1) reads coco annotations for set of test data (output of step 0)), and 
    2) unifies masks for shared object categories
    
"""  

###############################################################################

import os
import scipy.io
from pycocotools.coco import COCO
import pylab

###############################################################################

metadata_path  = '/'
path_in =  "../../testdata/0/"
path_out =  "../../testdata/4/"

###############################################################################

def unifying_coco_labels():
    
    # loading test data list (COCO validation set)
    data = scipy.io.loadmat(metadata_path + 'processed_data_list.mat',
                                      variable_names=['list_of_images','image_id_all'])
    image_id_all = data ['image_id_all'][0]  
            
    temp = image_id_all
    image_id_all = []
    for item in temp:
        value = item[0].strip()
        image_id_all.append(value)
    del temp
    
    # coco annotation #    
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)   
    dataDir='/.../MSCOCO/'
    dataType='val2014'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    #anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    #coco_caps=COCO(anncaptionFile)   
    #cats = coco.loadCats(coco.getCatIds())
    
    # unifying annotations #   
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
    
    # saving the results
    scipy.io.savemat(path_out + 'unified_labels.mat', {'ref_names_all':ref_names_all
                                                        ,'ref_IDs_all':ref_IDs_all
                                                        ,'ref_super_all':ref_super_all})
    

if __name__ == '__main__': 
    
    os.makedirs(path_out, exist_ok=1)
    unifying_coco_labels()
