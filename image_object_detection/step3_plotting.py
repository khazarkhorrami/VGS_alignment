
import os
project_path = '/worktmp/hxkhkh/project2/'
###############################################################################
# this file first loads confusion matrix and TP/FP/FN overlap related data 
# and plotts detection + precision and recall results
###############################################################################
in_path_corrected_ind = os.path.join(project_path , 'outputs/step_6/step_2/')
in_path_labels = os.path.join(project_path , 'outputs/step_6/step_4/')

in_path_processed_nouns = os.path.join(project_path , 'outputs/step_6/step_5/')
metadata_path  = os.path.join(project_path , 'outputs/step_1/validation/')

in_path = os.path.join(project_path , 'outputs/step_7/step_2/')

out_path = os.path.join(project_path , 'outputs/step_7/step_3/')

###############################################################################
import numpy
import scipy.io
from matplotlib import pyplot as plt
###############################################################################
                        # 2. Loading coco tool #
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
   
############################################################################### plotting

data = scipy.io.loadmat(in_path + 'all_info_WI_norm_updated2.mat', variable_names = ['all_labels_info','cm_detected'])

all_labels_info = data['all_labels_info']
cm_detected = data['cm_detected']

############################################################################## Plotting overlaps
average_results = numpy.mean((all_labels_info[:,0]/all_labels_info[:,-1]))
average_results = round(average_results,3)

plt.figure(figsize=[24,8])

plt.subplot(3,1,1)
plt.plot( all_labels_info[:,0]/all_labels_info[:,-1] , label = 'Averaged overlap score, overal score: '+str(average_results) , color = 'g')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(3,1,2)
plt.plot(all_labels_info[:,1] / all_labels_info[:,-1] ,label='Averaged mask area', color = 'r')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(3,1,3)
plt.plot(all_labels_info[:,-1]  ,label='Number of instances', color = 'b')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.savefig(out_path + 'test_WI_norm_updated2.pdf', format = 'pdf')

############################################################################## CM measures
# #....................................................................    
cm = cm_detected[0:80,:]

precision_detected = []
for count_object in range(80):
    target_col = cm[:, count_object]
    p = target_col[count_object] / numpy.sum(target_col)
    p = numpy.round(p,2)
    precision_detected.append(p)

recall_detected = []
fn_all =  all_labels_info[:,3] / all_labels_info[:,-1]
for count_object in range(80):
    target_col = cm[:, count_object]
    r = target_col[count_object] / (target_col[count_object] + all_labels_info[:,3][count_object])
    r = numpy.round(r,2)
    recall_detected.append(r)

precision_detection_all = numpy.round ( numpy.mean(precision_detected) ,2)
recall_detection_all = numpy.round ( numpy.mean(recall_detected) ,2)

plt.figure(figsize=[10,5])
plt.title('class-specific precision and recall for overlaps \n')
plt.subplot(2,1,1)
plt.plot(precision_detected, label = 'detection precision, average = ' + str(precision_detection_all))
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(recall_detected, label = 'detection recall, average = ' + str(recall_detection_all))
plt.legend()
plt.grid()


plt.savefig(out_path + 'precision_overlap_norm_updated2.pdf', format = 'pdf')       
