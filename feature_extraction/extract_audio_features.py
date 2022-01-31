from read_file_names import split_data, get_wav_files, data_chunker
import pickle
import numpy 
import librosa
from sklearn.utils import shuffle


def get_spokencoco_wavnames ( process_train_data , process_val_data , shuffle_data)   :
        # data and features path
        data_path =  "../../data/SPOKEN-COCO/" 
        feature_path = "../../features/SPOKEN-COCO/"
        
        # Getting file names
        train_imgs,val_imgs,train_caps,val_caps = split_data (data_path)   
          
        if process_train_data:
            input_captions = train_caps
            out_file_name = 'logmel_train_ch'
            #shuffle_data = True
        elif process_val_data:
            input_captions = val_caps
            out_file_name = 'logmel_val_ch'
            #shuffle_data = False
        logmel_path = feature_path + out_file_name
        
        wav_files, wav_files_count = get_wav_files(input_captions)
        
        if shuffle_data:
                   
            inds_shuffled = shuffle(numpy.arange(len(wav_files)))
            wav_files_shuffled = [wav_files[item] for item in inds_shuffled]
            del wav_files
            wav_files = wav_files_shuffled
            numpy.save(feature_path + 'shuffle_indices' , inds_shuffled)
        
        
        return wav_files, wav_files_count, data_path , logmel_path


def save_chunked_logmels (wavnames_chunk, input_path, logmel_savename ,  n_mel_bands , window_len , window_hop , sr):
    features_chunk = []
    for counter_pack , wav_pack in enumerate(wavnames_chunk):
        print(counter_pack)
        features_pack = []      
        for wav_file in  wav_pack:
            wav_path = input_path + wav_file
            wav_logmel = calculate_logmels(wav_path ,  n_mel_bands , window_len , window_hop , sr)
            features_pack.append(wav_logmel)            
        features_chunk.append(features_pack)        
    save_pickle(features_chunk, logmel_savename)

    

def calculate_logmels (wav_file_name , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target):
    
    win_len_sample = int (sr_target * window_len_in_ms)
    win_hop_sample = int (sr_target * window_hop_in_ms)
        
    y, sr = librosa.load(wav_file_name)
    y = librosa.core.resample(y, sr, sr_target) 
      
    mel_feature = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=win_len_sample, hop_length=win_hop_sample, n_mels=number_of_mel_bands,power=2.0)
    
    zeros_mel = mel_feature[mel_feature==0]          
    if numpy.size(zeros_mel)!= 0:
        
        mel_flat = mel_feature.flatten('F')
        mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
    
        if numpy.size(mel_temp)!=0:
            min_mel = numpy.min(numpy.abs(mel_temp))
        else:
            min_mel = 1e-12 
           
        mel_feature[mel_feature==0] = min_mel           
    logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))       
    return logmel_feature


def save_pickle (input_list, filename):
    outfile = open(filename,'wb')
    pickle.dump(input_list ,outfile , protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()
    
    

if __name__ == '__main__':

    processing_train_data = True
    processing_val_data = False 
    data_shuffling = False
    chunk_len=10000
    dataset = 'SPOKEN-COCO'
    
    # Audio features parameters
    number_of_mel_bands = 40     
    window_len_in_ms = 0.025
    window_hop_in_ms = 0.01
    sr_target = 16000
        
    if dataset == 'SPOKEN-COCO':
        # reading wav file names
        all_wav_files, all_wav_files_counts, data_path , output_path = get_spokencoco_wavnames ( processing_train_data , processing_val_data , data_shuffling)          
        # Extracting and saving audio features (one chunk at a time)       
        chuncked_wavs = data_chunker (all_wav_files , chunk_len)    
        for counter_chunk, data_chunk in  enumerate(chuncked_wavs):
            output_name =  output_path + str(counter_chunk) 
            save_chunked_logmels (data_chunk, data_path, output_name ,number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target)

    


