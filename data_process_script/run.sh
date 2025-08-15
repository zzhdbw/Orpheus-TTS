
# python 1_raw_to_ds.py \
    # --folder_path "./yuanshen" \
    # --output_parquet "./dataset/temp/yuanshen_dataset.parquet"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python 2_encode_audio.py \
#     --model_path "pretrained_model/hubertsiuzdak/snac_24khz" \
#     --input_parquet "./dataset/temp/yuanshen_dataset.parquet" \
#     --output_parquet "./dataset/temp/yuanshen_snac_dataset.parquet" \
#     --num_cpus 16 \
#     --num_gpus 4 \
#     --num_gpus_per_worker 0.4 \
#     --num_workers 8 \
#     --num_proc 8 \
#     --target_sr 24000

# python 3_encode_text.py \
#     --input_parquet "./dataset/temp/yuanshen_snac_dataset.parquet" \
#     --output_parquet "./dataset/yuanshen_tokenized_dataset.parquet" \
#     --tokenizer_path "pretrained_model/meta-llama" \
#     --num_proc 16 \
#     --task "SFT"
