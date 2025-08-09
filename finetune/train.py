from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import os
import torch
from typing import Any


os.environ["SWANLAB_PROJECT"] = "Orpheus-TTS"
os.environ["SWANLAB_WORKSPACE"] = "zhangzhihao"


config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)


# parquet_data = pq.read_table(dsn)

ds = load_dataset("parquet", data_files={"train": dsn})
ds = ds["train"]
print(ds.column_names)

# 去掉超长样本
# def truncate_sequences(example):
#     max_length = 8192
#     if len(example["input_ids"]) > max_length:
#         example["input_ids"] = None
#         print(f"{example} is too long !")
#     return example

# ds = ds.map(truncate_sequences)
# ds = ds.filter(lambda x: x["input_ids"] is not None)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=1,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="swanlab",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    # ddp_find_unused_parameters=False,
    # DeepSpeed configuration
    deepspeed="ds_z2_config.json",
    dataloader_pin_memory=True,
    # 梯度检查点
    gradient_checkpointing=True,
    use_liger_kernel=True,
)


def custom_data_collator(features: list) -> dict[str, Any]:

    max_length = max(len(feature["input_ids"]) for feature in features)

    batch = {}

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]

        # 计算需要padding的长度
        padding_length = max_length - len(input_ids)

        # Padding input_ids
        padded_input_ids = torch.cat(
            [
                torch.tensor(input_ids),
                torch.full((padding_length,), pad_token),
            ]
        )

        # Padding attention_mask
        padded_attention_mask = torch.cat(
            [torch.tensor(attention_mask), torch.zeros(padding_length)]
        )

        # Padding labels with -100
        padded_labels = torch.cat(
            [torch.tensor(labels), torch.full((padding_length,), -100)]
        )

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
        labels_list.append(padded_labels)

    batch["input_ids"] = torch.stack(input_ids_list)
    batch["attention_mask"] = torch.stack(attention_mask_list)
    batch["labels"] = torch.stack(labels_list)

    return batch


# pad
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=custom_data_collator,
)


trainer.train()
