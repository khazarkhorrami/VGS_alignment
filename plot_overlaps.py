import numpy
import scipy.io
from matplotlib import pyplot as plt
import os


path_project = '/worktmp/hxkhkh/project_3/'

file_in_AVtensor  = 'SI_CNN2_v2'

file_in = 'alignments_' + file_in_AVtensor 
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

data = scipy.io.loadmat(path_in + file_in + '.mat' , variable_names = ['all_sa_scores', 'all_ta_scores' ,'allrand_sa_scores', 'allrand_ta_scores', 'all_meta_info'])

all_meta_info = data['all_meta_info']
all_sa_scores = data['all_sa_scores'] 
all_ta_scores = data['all_ta_scores'] 
allrand_sa_scores = data['allrand_sa_scores']
allrand_ta_scores = data['allrand_ta_scores'] 


all_sa_scores = all_sa_scores[:,0]/all_meta_info[:,-1]
all_ta_scores = all_ta_scores[:,0]/all_meta_info[:,-1]
allrand_sa_scores = allrand_sa_scores[:,0]/all_meta_info[:,-1]
allrand_ta_scores = allrand_ta_scores[:,0]/all_meta_info[:,-1]

x = all_meta_info[:,0] /all_meta_info[:,-1] # pixel area
z = all_meta_info[:,1] / all_meta_info[:,-1] # duration

############################################################################## Plotting overlaps
average_sa = numpy.mean((all_sa_scores))
average_sa = round(average_sa,3)

average_ta = numpy.mean((all_ta_scores))
average_ta = round(average_ta,3)

average_saRand = numpy.mean((allrand_sa_scores))
average_saRand = round(average_saRand,3)

average_taRand = numpy.mean((allrand_ta_scores))
average_taRand = round(average_taRand,3)

plt.figure(figsize=[24,8])

plt.subplot(5,1,1)
plt.plot( all_sa_scores , label = 'Average sa_score '+str(average_sa) , color = 'r')
plt.plot( allrand_sa_scores , label = 'baseline '+str(average_saRand) , color = 'k')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,2)
plt.plot( all_ta_scores , label = 'Average ta_score '+str(average_ta) , color = 'b')
plt.plot( allrand_ta_scores , label = 'baseline '+str(average_taRand) , color = 'k')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,3)
plt.plot(x ,label='Average pixel area', color = 'g')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.subplot(5,1,4)
plt.plot(z ,label='Average duration (frames)', color = 'g')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

all_meta_info[0,-1]= 1500
plt.subplot(5,1,5)
plt.plot(all_meta_info[:,-1]  ,label='Number of instances', color = 'k')
plt.xticks(numpy.arange(80), fontsize = 10)
plt.grid()
plt.legend(fontsize = 10)

plt.savefig(path_out + file_out + '_overlaps.pdf', format = 'pdf')

############################################################################## Scatter plot



y_sa = all_sa_scores
y_sa_base = allrand_sa_scores

y_ta = all_ta_scores
y_ta_base = allrand_ta_scores

plt.figure(figsize=[16,8])
plt.subplot(1,2,1)
scatter_plot = plt.scatter( x,y_sa , color='r' , label = 'spatial')
scatter_plot = plt.scatter( x,y_sa_base , color='r', alpha = 0.2, label = 'baseline spatial' )
scatter_plot = plt.scatter( x,y_ta , color='b' , marker = '^', label = 'temporal')
scatter_plot = plt.scatter( x,y_ta_base , color='b',marker = '^', alpha = 0.2, label = 'baseline temporal' )
plt.xlabel('\nAverage object area',fontsize = 12)
plt.ylabel('Alignment scores\n',fontsize = 12)
plt.legend(fontsize = 12)
plt.subplot(1,2,2)
scatter_plot = plt.scatter( z,y_sa , color='r' , label = 'spatial')
scatter_plot = plt.scatter( z,y_sa_base , color='r', alpha = 0.2, label = 'baseline spatial' )
scatter_plot = plt.scatter( z,y_ta , color='b' , marker = '^', label = 'temporal')
scatter_plot = plt.scatter( z,y_ta_base , color='b',marker = '^', alpha = 0.2, label = 'baseline temporal' )
plt.xlabel('\nAverage word duration',fontsize = 12)
plt.ylabel('Alignment scores\n',fontsize = 12)
plt.legend(fontsize = 12)
plt.savefig(path_out + file_out + '_scatter.pdf', format = 'pdf')
