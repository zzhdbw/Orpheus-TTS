from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
import torchaudio.transforms as T
import pandas as pd
import torch
import io
import soundfile as sf
import random
from transformers import AutoTokenizer
import os
from snac import SNAC
import pyarrow.parquet as pq
import glob

# pretrained model inference
# start_of_human,
# "Something or the other",
# end_of_text
# end_of_human
# start_of_ai
# start_of_speech
# ref_audio
# end_of_ai
# start_of_human
# text
# end_of_text
# end_of_human
# start_of_ai
# start_of_speech

# pretrained model train
# [start_of_human]
# example["text_tokens"]
# [end_of_human]
# [start_of_ai]
# [start_of_speech]
# example["codes_list"]
# [end_of_speech]
# [end_of_ai]
# )

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10


def tokenise_audio(waveform, original_sampling_rate, target_sampling_rate: int = 24000):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    resample_transform = T.Resample(
        orig_freq=original_sampling_rate, new_freq=target_sampling_rate
    )

    waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to(f"cuda")

    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

    return all_codes


def add_codes(example):
    codes_list = None
    try:
        audio_bytes = example.get("audio_bytes")  # Adjusted to match flat column name

        if not isinstance(audio_bytes, (bytes, bytearray)):
            print(f"Unsupported type for audio_bytes: {type(audio_bytes)}")
            return example

        with io.BytesIO(audio_bytes) as audio_buf:
            audio_array, _ = sf.read(audio_buf)

        codes_list = tokenise_audio(
            audio_array,
            example["sampling_rate"],
            args.target_sampling_rate,
        )

    except Exception as e:
        print(f"Skipping row due to error: {e}")

    example["codes_list"] = codes_list
    return example


def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i : i + 7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example


def create_input_ids(example):
    if is_SFT:
        text_ids = tokenizer.encode(
            example["speaker_id"] + ": " + example["text"], add_special_tokens=True
        )
    else:
        text_ids = tokenizer.encode(example["text"], add_special_tokens=True)

    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dataset_path",
        type=str,
        default="data/processed/yuanshen/Genshin5_4_CN.parquet",
    )
    parser.add_argument(
        "--disk_dataset_path", type=str, default="data/processed/yuanshen/"
    )
    parser.add_argument(
        "--SNAC_model_path",
        type=str,
        default="pretrained_models/hubertsiuzdak/snac_24khz",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="pretrained_models/canopylabs/3b-zh-pretrain-research_release",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["SFT", "pretrain"],
    )
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--target_sampling_rate", type=int, default=24000)
    args = parser.parse_args()

    is_SFT = args.task == "SFT"
    print(f"is_SFT: {is_SFT}")

    ###############################################################
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading SNAC model...")
    model = SNAC.from_pretrained(args.SNAC_model_path).to(f"cuda")
    # model compile

    print("Loading parquet data...")
    # "data/processed/yuanshen/Genshin5_4_CN_*.parquet"
    # 读取符合这个格式的全部文件
    parquet_data = pq.read_table(args.original_dataset_path)
    # 只要前100条

    print("Creating dataset...")
    ds = Dataset(parquet_data)
    print(len(ds))
    # dict_keys(['text', 'audio_bytes', 'sampling_rate', 'speaker_id'])

    print(f"读取音频并重采样至{args.target_sampling_rate}Hz，编码成tokenid")
    ds = ds.map(add_codes, remove_columns=["audio_bytes"], num_proc=1)
    ds = ds.filter(lambda x: x["codes_list"] is not None)
    ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

    print("删除冗余列")
    ds = ds.map(remove_duplicate_frames, num_proc=args.num_proc)

    print("tokenize文本")
    ds = ds.map(
        create_input_ids, num_proc=args.num_proc, remove_columns=["text", "codes_list"]
    )

    print("删除冗余列")
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    if is_SFT:
        ds.to_parquet(
            os.path.join(args.disk_dataset_path, "data-00000-of-00001_SFT.parquet")
        )
    else:
        ds.to_parquet(
            os.path.join(args.disk_dataset_path, "data-00000-of-00001_pretrain.parquet")
        )
