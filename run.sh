# dataprocess
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m data_process_script.raw2parquet_Genshin5_4_CN
# python -m data_process_script.parquet2tokenised \
#     --original_dataset_path "data/processed/yuanshen/Genshin5_4_CN.parquet" \
#     --disk_dataset_path data/processed/yuanshen/ \
#     --SNAC_model_path pretrained_models/hubertsiuzdak/snac_24khz \
#     --tokenizer_name pretrained_models/canopylabs/3b-zh-pretrain-research_release \
#     --task SFT \
#     --num_proc 8 \
#     --target_sampling_rate 24000

# pretrain
cd pretrain
nohup deepspeed --include="localhost:5,6,7" --master_port=29501 train.py > pretrain.log 2>&1 &


# # SFT 
# cd finetune
# deepspeed --include="localhost:4,5,6,7" --master_port=29501 train.py 