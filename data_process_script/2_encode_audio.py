import torch
from snac import SNAC
from datasets import Dataset, Audio
import soundfile as sf
import os
from tqdm import tqdm
import ray
import librosa
import time
from light_util import get_time


class SNACWrapper:
    def __init__(self, model_path, target_sr) -> None:
        model = SNAC.from_config(os.path.join(model_path, "config.json"))
        state = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        model.load_state_dict(state)
        self.model = model.cuda().eval()
        # self.model = torch.compile(
        #     self.model
        # )  # There is a risk of decreased effectiveness instead of accelerating
        self.target_sr = target_sr

    @torch.inference_mode()
    def call(self, data):

        audio_array, sr = librosa.load(data["audio_path"], sr=self.target_sr)

        audio_array = (
            torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0).cuda()
        )

        codes = self.model.encode(audio_array)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))

        data["codes_list"] = all_codes

        return data


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


@get_time
def main():
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str, help='snac model path')
    parse.add_argument('--input_parquet', type=str, help='input parquet file path')
    parse.add_argument('--output_parquet', type=str, help='onput parquet file path')
    parse.add_argument('--num_cpus', type=int, help='ray num_cpus')
    parse.add_argument('--num_gpus', type=int, help='ray num_gpus')
    parse.add_argument(
        '--num_gpus_per_worker', type=float, help='ray num_gpus_per_worker'
    )
    parse.add_argument('--num_workers', type=int, help='ray num_workers')
    parse.add_argument('--num_proc', type=int, help='dataset map num_proc')
    parse.add_argument('--target_sr', type=int, help='audio target sample rate')
    opt = parse.parse_args()

    # model_path = "pretrained_model/hubertsiuzdak/snac_24khz"
    # input_parquet = "dataset/temp/yuanshen_dataset.parquet"
    # output_parquet = "dataset/temp/yuanshen_snac_dataset.parquet"

    # num_cpus = 16
    # num_gpus = 4
    # num_gpus_per_worker = 0.4
    # num_workers = 8
    # num_proc = 8
    # target_sr = 24000
    #######################
    ray.init(num_gpus=opt.num_gpus, num_cpus=opt.num_cpus)

    SNACWrapper_remote = ray.remote(num_gpus=opt.num_gpus_per_worker)(SNACWrapper)

    # 加载数据集
    print("Loading dataset...")
    ds = Dataset.from_parquet(opt.input_parquet)
    print(f"Dataset loaded with {len(ds)} samples")

    # 创建 SNAC workers
    print("Creating SNAC workers...")
    workers = []
    for gpu_id in range(opt.num_workers):
        worker = SNACWrapper_remote.remote(opt.model_path, opt.target_sr)
        workers.append(worker)
    print("Create SNAC workers done")

    futures = []
    for index, d in enumerate(ds):
        future = workers[index % opt.num_workers].call.remote(d)
        futures.append(future)

    all_results = []
    for future in tqdm(futures, desc="Processing audio"):
        all_results.append(ray.get(future))
    ray.shutdown()

    result_ds = Dataset.from_list(all_results)
    result_ds = result_ds.remove_columns(["audio_path"])
    result_ds = result_ds.map(remove_duplicate_frames, num_proc=opt.num_proc)

    print(len(all_results))
    print(f"Processing completed! Total processed: {len(all_results)} samples")
    print(result_ds[0])
    result_ds.to_parquet(opt.output_parquet)


if __name__ == "__main__":
    main()
