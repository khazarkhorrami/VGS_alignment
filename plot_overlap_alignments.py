import numpy
import scipy.io
from matplotlib import pyplot as plt
import os


path_project = '/worktmp/hxkhkh/project3/'

file_in_AVtensor  = 'SI_CNN2_v2'

file_in = 'alignments' + file_in_AVtensor 
file_out = file_in


path_in = os.path.join(path_project , 'outputs/step_2/')
path_out = os.path.join(path_project , 'outputs/step_3/')

###############################################################################
                        # 2. Loading coco tool #
############################################################################### loading coco package
from pycocotools.coco import COCO
dataDir='/worktmp/hxkhkh/data/coco/MSCOCO/'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)
cats = coco.loadCats(coco.getCatIds())
cats_id = [item['id'] for item in cats]
cats_names = [item['name']for item in cats]  
############################################################################### plotting

data = scipy.io.loadmat(path_in + file_in + '.mat' , variable_names = ['all_sa_scores', 'all_ta_scores' , 'all_meta_info'])

all_sa_scores = data['all_sa_scores']
all_ta_scores = data['all_ta_scores']
all_meta_info = data['all_meta_info']
############################################################################## Plotting overlaps
average_sa = numpy.mean((all_sa_scores[:,0]/all_meta_info[:,-1]))
average_sa = round(average_sa,3)

average_ta = numpy.mean((all_ta_scores[:,0]/all_meta_info[:,-1]))
average_ta = round(average_sa,3)


plt.figure(figsize=[24,8])

plt.subplot(5,1,1)
plt.plot( all_sa_scores[:,0]/all_meta_info[:,-1] , label = 'Averaged sa_score '+str(average_sa) , color = 'g')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,2)
plt.plot( all_ta_scores[:,0]/all_meta_info[:,-1] , label = 'Averaged ta_score '+str(average_ta) , color = 'g')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,3)
plt.plot(all_meta_info[:,0] /all_meta_info[:,-1] ,label='Averaged pixel area', color = 'r')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,4)
plt.plot(all_meta_info[:,0] / all_meta_info[:,-1] ,label='Averaged duration (in frames)', color = 'r')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,5)
plt.plot(all_meta_info[:,-1]  ,label='Number of instances', color = 'k')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.savefig(path_out + file_out + '_overlaps.pdf', format = 'pdf')

