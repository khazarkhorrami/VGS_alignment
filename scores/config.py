
###################### initial configuration  #################################


paths = {
  "file_indices": "../../testdata/3/corrected_nouns_index.mat",
  "file_metadata": "../../testdata/0/processed_data_list.mat",
  "file_nouns": "../../testdata/6/sub_labels.mat",
  "file_labels": "../../testdata/4/unified_labels.mat",
  "file_AVtensor" : "../../model/AVtensors/example_tensor.mat",
  "path_output" : "../../scores"
}


score_settings = {

  "find_GS": False,
  "softmax": True,
  "n_categories" : 80,

  "res_target_h" : 224,    
  "res_target_w" : 224,
  "res_target_t" : 512,

  "res_source_h" : 14,
  "res_source_w" : 14,
  "res_source_t" : 64,
}

