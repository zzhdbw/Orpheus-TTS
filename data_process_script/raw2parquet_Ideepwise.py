#
import os
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm
import random

folder_path = "data/processed/yuanshen/Genshin5_4_CN.parquet"
output_parquet = "data/processed/yuanshen/Genshin5_4_CN_tokenized.parquet"

os.makedirs(os.path.dirname(output_parquet), exist_ok=True)

file_list = []
for file in os.listdir(folder_path):
    dir_path = os.path.join(folder_path, file)
    if os.path.isdir(dir_path):
        file_list.append(file)
print(len(file_list))

# 读取对应目录下的.lab文件并列出路径
audio_bytes_list = []
audio_paths = []
sampling_rates = []
texts = []
speaker_ids = []

for persion_name in tqdm(file_list):
    persion_path = os.path.join(folder_path, persion_name)
    lab_wav_name_list = []
    for lab_file in os.listdir(persion_path):
        if lab_file.endswith(".txt"):
            lab_wav_name_list.append(
                os.path.join(persion_path, lab_file).replace(".txt", "")
            )

    lab_wav_name_list = random.sample(lab_wav_name_list, 300)

    for lab_wav_name in lab_wav_name_list:
        wav_path = lab_wav_name + ".wav"
        lab_path = lab_wav_name + ".txt"

        # 检查文件是否存在
        if not os.path.exists(wav_path) or not os.path.exists(lab_path):
            print(f"警告: 文件不存在 - {wav_path} 或 {lab_path}")
            continue

        try:
            # 读取音频文件
            with open(wav_path, "rb") as af:
                audio_bytes = af.read()
                _, sr = sf.read(wav_path)

            # 验证音频数据
            if not audio_bytes or len(audio_bytes) == 0:
                print(f"警告: 音频文件为空 - {wav_path}")
                continue

            # 读取文本内容
            with open(lab_path, "r", encoding='utf-8') as f:
                text_content = f.read().strip()

            audio_bytes_list.append(audio_bytes)
            audio_paths.append(wav_path)
            sampling_rates.append(sr)
            texts.append(text_content)  # 存储实际的文本内容
            speaker_ids.append(persion_name)
        except Exception as e:
            print(f"处理文件时出错 {wav_path}: {str(e)}")
            continue
print(len(audio_bytes_list))
try:
    # 直接创建展平的表结构
    table = pa.table(
        {
            "text": pa.array(texts, type=pa.string()),
            "audio_bytes": pa.array(audio_bytes_list, type=pa.large_binary()),
            # "audio_path": pa.array(audio_paths, type=pa.string()),
            "sampling_rate": pa.array(sampling_rates, type=pa.int32()),
            "speaker_id": pa.array(speaker_ids, type=pa.string()),
        }
    )

    pq.write_table(table, output_parquet)
    print(f"成功生成parquet文件: {output_parquet}")
except Exception as e:
    print(f"生成parquet文件时出错: {str(e)}")
