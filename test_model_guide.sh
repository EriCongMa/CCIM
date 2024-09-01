# A Quick Testing Script Guide for the CCIM: Cross-modal Cross-lingual Interactive Image Translation
# Please update the corresponding path and hyper-parameters before running the code in your own environment.
echo 'Please update the corresponding path and hyper-parameters before running the code in your own environment!'

code_path=${code_path}/CCIM/evaluate.py
src_max_len=${max_length_of_source_language}
tgt_max_len=${max_length_of_source_language}
let img_width=${src_max_len}*4      # To make the sequence length of image features and text features consistent
batch_size=${batch_size}
task_name=interactive_decoding
model_path=${path_of_saved_path}
exp_name=${name_of_model_setting}   # Finally, the model is saved in ${model_path}/${exp_name}/
model_name=best_bleu    # or best_valid, or best_accuracy, or iter.
saved_iteration=final   # final or specific saved step.
lmd_value=${value_of_weighted_interactive_decoding}
interactive_type=${interactive_type}    # weighted or hierarchical

test_image_path=${path_of_testing_images}
src_decoded_path=${path_of_src_decoded_results}
tgt_decoded_path=${path_of_tgt_decoded_results}

# Path of Vocabulary | txt file.
vocab_src=${path_of_source_language_vocabulary}
vocab_tgt=${path_of_target_language_vocabulary}

echo 'Remove Previous Decoded Results.'
rm ${src_decoded_path}
rm ${tgt_decoded_path}

echo 'Start to Decode ...'
${python_path} ${code_path} \
--task ${task_name} \
--interactive_lambda ${lmd_value} \
--interactive_type ${interactive_type} \
--imgW ${img_width} --rgb \
--image_folder ${test_image_path} \
--src_vocab ${vocab_src} --tgt_vocab ${vocab_tgt} \
--src_batch_max_length ${src_max_len} \
--tgt_batch_max_length ${tgt_max_len} \
--batch_size ${batch_size} \
--saved_model ${model_path}/${exp_name}/${model_name} \
--saved_iter ${saved_iteration} \
--data_format pic \
--src_output ${src_decoded_path} \
--tgt_output ${tgt_decoded_path} 

echo 'Finished Testing.'
