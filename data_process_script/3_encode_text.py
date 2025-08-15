from transformers import AutoTokenizer
from datasets import Dataset
import os


def tokenize_text(example, tokenizer, task):

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

    if task == "SFT":
        text_ids = tokenizer.encode(
            example["speaker_id"] + ": " + example["text"], add_special_tokens=True
        )
    elif task == "pretrain":
        text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    else:
        raise ValueError("Task must be either 'SFT' or 'pretrain'")

    text_ids.append(end_of_text)
    example["text_ids"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_ids"]
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

    parse = argparse.ArgumentParser()
    parse.add_argument('--input_parquet', type=str, help='input parquet file path')
    parse.add_argument('--output_parquet', type=str, help='output parquet file path')
    parse.add_argument('--tokenizer_path', type=str, help='output parquet file path')
    parse.add_argument('--num_proc', type=int, help='dataset map num_proc')
    parse.add_argument(
        '--task', type=str, choices=["SFT", "pretrain"], help='SFT or pretrain task'
    )
    opt = parse.parse_args()
    ##################################################
    ds = Dataset.from_parquet(opt.input_parquet)

    tokenizer = AutoTokenizer.from_pretrained(opt.tokenizer_path)
    is_SFT = False  # Set to True if you are using SFT
    ds = ds.map(
        tokenize_text,
        fn_kwargs={"tokenizer": tokenizer, "task": opt.task},
        num_proc=opt.num_proc,
    )
    ds = ds.select_columns(["input_ids", "labels", "attention_mask"])
    print(ds[0])

    ds.to_parquet(opt.output_parquet)
