
###############################################################################
# import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.6
# sess = tf.Session(config=config) 

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
###############################################################################

import scipy.io
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('/worktmp/hxkhkh/project_temp/step_9/GoogleNews-vectors-negative300.bin', binary=True)

out_path = '/worktmp/hxkhkh/project_3/output/step_1/2/'
in_path =  '/worktmp/hxkhkh/project_3/output/step_1/3/'

############################################################################### 
data = scipy.io.loadmat(in_path + 'noun_data.mat', variable_names = ['wavfile_nouns','wavfile_nouns_onsets','wavfile_nouns_offsets'])

wavfile_nouns = data ['wavfile_nouns'][0]
wavfile_nouns_onsets = data ['wavfile_nouns_onsets'][0]
wavfile_nouns_offsets = data ['wavfile_nouns_offsets'] [0]


wavfile_nouns_onsets = [item for item in wavfile_nouns_onsets] 
wavfile_nouns_offsets = [item for item in wavfile_nouns_offsets]  

temp = wavfile_nouns
wavfile_nouns = []
for utterance in temp:
    correct_utterance = []
    for phoneme in utterance:
        correct_ph  = phoneme.strip()
        correct_utterance.append(correct_ph)
    wavfile_nouns.append(correct_utterance)   
###############################################################################

###############################################################################
#.......................................... 


ind_accepted = []

for counter_utterance, candidate_utterance  in enumerate(wavfile_nouns):
    
    print(counter_utterance)
    try:
        model.most_similar(candidate_utterance)
        ind_accepted.append(counter_utterance)
    except:
        print("An exception occurred in word:  " + str(candidate_utterance))
        

scipy.io.savemat(out_path + 'corrected_nouns_index.mat', {'ind_accepted':ind_accepted})


