import numpy
import scipy.io
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
import os


path_project = '/worktmp/hxkhkh/project_3/'


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

def scatter_plot(x, z, y_sa,y_sa_base,y_ta,y_ta_base):
    
    
    plt.figure(figsize=[16,8])
    plt.subplot(1,2,1)
    
    plt.scatter( x,y_sa , color='r' , label = 'spatial')
    b, m = polyfit(x, y_sa, 1)
    plt.plot(x, b + m * x, color='r')
    
    plt.scatter( x,y_sa_base , color='r', alpha = 0.2, label = 'baseline spatial' )
    b, m = polyfit(x, y_sa_base, 1)
    plt.plot(x, b + m * x, color='r', alpha = 0.2)
    
    plt.scatter( x,y_ta , color='b' , marker = '^', label = 'temporal')
    b, m = polyfit(x, y_ta, 1)
    plt.plot(x, b + m * x, color='b')
    
    plt.scatter( x,y_ta_base , color='b', alpha = 0.2, marker = '^', label = 'baseline temporal' )
    b, m = polyfit(x,y_ta_base, 1)
    plt.plot(x, b + m * x, color='b', alpha = 0.2)
    
    plt.xlabel('\nAverage object area (per image area)',fontsize = 12)
    plt.ylabel(y_label + '\n',fontsize = 12)
    plt.legend(fontsize = 12)
    plt.grid()
    
    plt.subplot(1,2,2)
    
    plt.scatter( z,y_sa , color='r' , label = 'spatial')
    b, m = polyfit(z,y_sa, 1)
    plt.plot(z, b + m * z, color='r')
    
    plt.scatter( z,y_sa_base , color='r', alpha = 0.2, label = 'baseline spatial' )
    b, m = polyfit(z,y_sa_base, 1)
    plt.plot(z, b + m *z, color='r', alpha = 0.2)
    
    plt.scatter( z,y_ta , color='b' , marker = '^', label = 'temporal')
    b, m = polyfit(z,y_ta, 1)
    plt.plot(z, b + m * z, color='b')
    
    plt.scatter( z,y_ta_base , color='b', alpha = 0.2 , marker = '^', label = 'baseline temporal' )
    b, m = polyfit(z,y_ta_base, 1)
    plt.plot(z, b + m * z, color='b', alpha = 0.2)
    
    plt.xlabel('\nAverage word duration (frames)',fontsize = 12)
    plt.ylabel(y_label + '\n',fontsize = 12)
    plt.legend(fontsize = 12)
    plt.grid()

############################################################################### Reading data file
def reading_input_data(path_in,file_in):
    data = scipy.io.loadmat(path_in + file_in + '.mat' , variable_names = ['all_sa_scores', 'all_ta_scores' , 'all_meta_info',
                                                                           'allrand1_sa_scores', 'allrand1_ta_scores','allrand2_sa_scores', 'allrand2_ta_scores',
                                                                           'cm_detection', 'cm_rand1','cm_rand2',' cm_object_area'])
    
    all_meta_info = data['all_meta_info']
    all_sa_scores = data['all_sa_scores'] 
    all_ta_scores = data['all_ta_scores'] 
    allrand_sa_scores = data['allrand1_sa_scores']
    allrand_ta_scores = data['allrand1_ta_scores'] 
    
    
    all_sa_scores = all_sa_scores[:,0]/all_meta_info[:,-1]
    all_ta_scores = all_ta_scores[:,0]/all_meta_info[:,-1]
    allrand_sa_scores = allrand_sa_scores[:,0]/all_meta_info[:,-1]
    allrand_ta_scores = allrand_ta_scores[:,0]/all_meta_info[:,-1]
    
    x = all_meta_info[:,0] /all_meta_info[:,-1] # pixel area
    z = all_meta_info[:,1] / all_meta_info[:,-1] # duration
    x = x / numpy.repeat(224*224, 80)
    x = numpy.round(x,2)
    
    y_sa = all_sa_scores
    y_sa_base = allrand_sa_scores
        
    y_ta = all_ta_scores
    y_ta_base = allrand_ta_scores
    return x, z, y_sa,y_sa_base,y_ta,y_ta_base
###############################################################################

file_in_AVtensor  = 'SI_CNN2_v2_orig'
file_in = 'alignment_' + file_in_AVtensor 
y_label = 'Alignment score'
path_in = os.path.join(path_project , 'outputs/step_2/time_window_zero/')
path_out = os.path.join(path_project , 'outputs/step_3/')

x, z, y_sa,y_sa_base,y_ta,y_ta_base =  reading_input_data(path_in,file_in)   
scatter_plot(x, z, y_sa,y_sa_base,y_ta,y_ta_base)
    
plt.savefig(path_out + 'scatter_plt_all.png', format = 'png')
