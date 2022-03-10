
###################### initial configuration  #################################


paths = {
  "model_name": "CNNatt",
  "model_subname": "v0",
  "modeldir": "../../model/",
  "featuredir": "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/",#"../../features/SPOKEN-COCO/",
  "visual_feature_name" : "vggb5conv3_",
  "audio_feature_name" : "logmel_"
}


model_settings = {
  "use_pretrained": False,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : True,
  "number_of_epochs" : 100,
}

feature_settings = {
    "n_caps_per_image":5,
    "set_of_train_files":['train_ch0', 'train_ch1', 'train_ch2', 'train_ch3', 'train_ch4', 'train_ch5', 'train_ch6', 'train_ch7'],
    "set_of_validation_files" : ['train_ch8'],
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
