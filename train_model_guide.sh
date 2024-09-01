# A Quick Training Script Guide for the CCIM: Cross-modal Cross-lingual Interactive Image Translation
# Please update the corresponding path and hyper-parameters before running the code in your own environment!
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/CCIM/trainer.py
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
model_path=${path_of_model_saving}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
batch_size=${batch_size}
task_name=interactive_decoding
lmd_value=${value_of_weighted_interactive_decoding}
interactive_type=${interactive_type}    # weighted or hierarchical

# Path of End-to-end Text Image Translation Dataset | lmdb file.
train_path=${path_of_e2e_tit_train_dataset}
valid_path=${path_of_e2e_tit_valid_dataset}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Model Folder.'
rm -rf ${model_path}/${exp_name}/

${python_path} ${code_path} \
--task ${task_name} \
--interactive_lambda ${lmd_value} \
--interactive_type ${interactive_type} \
--imgW ${img_width} --rgb \
--train_data ${train_path} \
--valid_data ${valid_path} \
--saved_model ${model_path} \
--exp_name ${task_}_${src_lang}${tgt_lang}_${granular}-${date_info} \
--src_vocab ${vocab_src} \
--tgt_vocab ${vocab_tgt} \
--batch_size ${batch_size} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \

echo 'Finished Training.'
