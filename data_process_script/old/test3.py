from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(
    "../pretrained_models/meta-llama/Llama-3.2-3B-Instruct"
)


def tokenize(msg):
    tokenized = tokenizer.apply_chat_template(
        msg,
        tokenize=True,
    )

    return tokenized


import json

path = "/home/zzh/code/TTS/Orpheus-TTS/data/raw/text/swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT/qwen3_235b_2507_distill_110k.jsonl"

all_data = []
with open(path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        data = json.loads(line)
        # print(data['messages'])
        input_ids = tokenize(data['messages'])
        attention_mask = [1] * len(input_ids)
        labels = input_ids
        # example["input_ids"] = input_ids
        # example["labels"] = input_ids
        # example["attention_mask"] = [1] * len(input_ids)

        all_data.append(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        )

# 将data转换成parquet文件
import pandas as pd

df = pd.DataFrame(all_data)
df.to_parquet("../data/processed/text/data-00000-of-00001_pretrain.parquet")
