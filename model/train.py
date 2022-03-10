import os
import numpy
import scipy.io
from matplotlib import pyplot as plt

from tensorflow import keras

from utils import prepare_triplet_data , triplet_loss , calculate_recallat10
from model import VGS
    
class train_validate (VGS):
    
    def __init__(self,model_name, model_subname, input_dim, model_dir, feature_dir, feature_name, training_chunks, validation_chunks, training_params, validation_params, action_parameters, use_pretrained):
        VGS.__init__(self, model_name, model_subname, input_dim)
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.feature_name = feature_name
        self.training_chunks = training_chunks
        self.validation_chunks = validation_chunks
        self.training_params = training_params
        self.validation_params = validation_params
        self.action_parameters = action_parameters
        self.use_pretrained = use_pretrained
        
        [Xshape, Yshape] = self.input_dim
        self.length_sequence = Xshape[0]
        super().__init__(model_name, model_subname, input_dim) 
        
        
    def initialize_model_parameters(self):
        
        if self.use_pretrained:
            data = scipy.io.loadmat(self.model_dir + 'valtrainloss.mat', variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
            allepochs_valloss = data['allepochs_valloss'][0]
            allepochs_trainloss = data['allepochs_trainloss'][0]
            all_avRecalls = data['all_avRecalls'][0]
            all_vaRecalls = data['all_vaRecalls'][0]
            
            allepochs_valloss = numpy.ndarray.tolist(allepochs_valloss)
            allepochs_trainloss = numpy.ndarray.tolist(allepochs_trainloss)
            all_avRecalls = numpy.ndarray.tolist(all_avRecalls)
            all_vaRecalls = numpy.ndarray.tolist(all_vaRecalls)
            recall_indicator = numpy.max(allepochs_valloss)
            val_indicator = numpy.min(allepochs_valloss)
        else:
            allepochs_valloss = []
            allepochs_trainloss = []
            all_avRecalls = []
            all_vaRecalls = []
            recall_indicator = 0
            val_indicator = 1000
            
        saving_params = [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ]       
        return saving_params    


    def train_model(self, vgs_model):    
       
        [audio_feature_name,visual_feature_name ] = self.feature_name
        [number_of_captions_per_image, length_sequence] = self.training_params
        
        set_of_input_chunks = self.training_chunks
        for chunk_counter, chunk_name in enumerate(set_of_input_chunks): 
            for i_caption in range(number_of_captions_per_image):
                Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet_data (self.feature_dir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , self.length_sequence)          
                print('.......... train chunk ..........' + str(chunk_counter))
                print('.......... audio caption ........' + str(i_caption))            
                history = vgs_model.fit([Ydata_triplet, Xdata_triplet ], bin_triplet, shuffle=False, epochs=1,batch_size=120)   
                del Xdata_triplet
            del Ydata_triplet      
        final_trainloss = history.history['loss'][0]
        return final_trainloss

    

    def evaluate_model (self, vgs_model,  visual_embedding_model, audio_embedding_model ) :
        
        [audio_feature_name,visual_feature_name ] = self.feature_name
        [ find_recall, save_best_recall]  = self.validation_params  
                
        epoch_cum_val = 0
        epoch_cum_recall_av = 0
        epoch_cum_recall_va = 0
        i_caption = numpy.random.randint(0, 5)
        set_of_input_chunks = self.validation_chunks

        for chunk_counter, chunk_name in enumerate(set_of_input_chunks):
            print('.......... validation chunk ..........' + str(chunk_counter))
            Ydata_triplet, Xdata_triplet, bin_triplet =  prepare_triplet_data (self.feature_dir , visual_feature_name ,  audio_feature_name , chunk_name , i_caption , self.length_sequence)
            val_chunk = vgs_model.evaluate( [Ydata_triplet,Xdata_triplet ],bin_triplet,batch_size=120)    
            epoch_cum_val += val_chunk                  
            #..................................................................... Recall
            if find_recall:
                           
                Ydata = Ydata_triplet[0::3]
                Xdata = Xdata_triplet[0::3]

                number_of_samples = len(Ydata)

                
                visual_embeddings = visual_embedding_model.predict(Ydata)
                visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1) 

                audio_embeddings = audio_embedding_model.predict(Xdata)
                audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)                 
                
                
                ########### calculating Recall@10                    
                poolsize =  1000
                number_of_trials = 100
                recall_av_vec = calculate_recallat10( audio_embeddings_mean, visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize )          
                recall_va_vec = calculate_recallat10( visual_embeddings_mean , audio_embeddings_mean, number_of_trials,  number_of_samples , poolsize ) 
                recall10_av = numpy.mean(recall_av_vec)/(poolsize)
                recall10_va = numpy.mean(recall_va_vec)/(poolsize)
                epoch_cum_recall_av += recall10_av
                epoch_cum_recall_va += recall10_va               
                del Xdata, audio_embeddings
                del Ydata, visual_embeddings            
            del Xdata_triplet,Ydata_triplet
            
        final_recall_av = epoch_cum_recall_av / (chunk_counter + 1 ) 
        final_recall_va = epoch_cum_recall_va / (chunk_counter + 1 ) 
        final_valloss = epoch_cum_val/ len (set_of_input_chunks) 
        
        validation_output = [final_recall_av, final_recall_va , final_valloss]
        return validation_output
    
    def save_model(self, vgs_model, initialized_output , training_output, validation_output):
        
        os.makedirs(self.model_dir, exist_ok=1)
        [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ] = initialized_output
        [final_recall_av, final_recall_va , final_valloss] = validation_output 
        [find_recall, save_best_recall]  = self.validation_params
        [epoch_recall_av, epoch_recall_va , epoch_valloss] = validation_output
               
            
        if save_best_recall:
            epoch_recall = ( epoch_recall_av + epoch_recall_va ) / 2
            if epoch_recall >= recall_indicator: 
                recall_indicator = epoch_recall
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
        else :
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
                      
        allepochs_trainloss.append(training_output)  
        allepochs_valloss.append(epoch_valloss)
        if find_recall: 
            all_avRecalls.append(epoch_recall_av)
            all_vaRecalls.append(epoch_recall_va)
        save_file = self.model_dir + 'valtrainloss.mat'
        scipy.io.savemat(save_file, 
                          {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })  
        
        self.make_plot( [allepochs_trainloss, allepochs_valloss , all_avRecalls, all_vaRecalls ])

        
    def make_plot (self, plot_lists):
        
        plt.figure()
        plot_names = ['training loss','validation loss','speech_to_image recall','image_to_speech recall']
        for plot_counter, plot_value in enumerate(plot_lists):
            plt.subplot(2,2,plot_counter+1)
            plt.plot(plot_value)
            plt.ylabel(plot_names[plot_counter])
            plt.grid()

        plt.savefig(self.model_dir + 'evaluation_plot.pdf', format = 'pdf')
 
        
    def __call__(self):
    
        vgs_model, visual_embedding_model, audio_embedding_model = self.build_model()
        vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
        print(vgs_model.summary())
  
        initialized_output = self.initialize_model_parameters()
        [number_of_epochs , training_mode, evaluating_mode, saving_mode] = self.action_parameters
  
        if self.use_pretrained:
            vgs_model.load_weights(self.model_dir + 'model_weights.h5')

             
        
        for epoch_counter in numpy.arange(number_of_epochs):
            
            print('......... epoch ...........' , str(epoch_counter))
            
            if training_mode:
                training_output = self.train_model(vgs_model)
            else:
                training_output = 0
                
            if evaluating_mode:
                
                validation_output = self.evaluate_model(vgs_model,  visual_embedding_model, audio_embedding_model )
            else: 
                validation_output = [0, 0 , 0 ]
            if saving_mode:
                self.save_model(vgs_model,initialized_output, training_output, validation_output)
                

