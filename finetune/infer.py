# model_name = (
#     "/home/zzh/code/TTS/Orpheus-TTS/finetune/checkpoints/BZNSYP/checkpoint-1250"
# )

model_name = "ckpt/Genshin5_4_CN/checkpoint-7865"
tokenizer_path = "../pretrained_models/canopylabs/3b-zh-pretrain-research_release"
from snac import SNAC
import torch
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf

# import IPython.display as ipd
# import librosa
# from ipywebrtc import AudioRecorder, Audio
# from IPython.display import display
# import ipywidgets as widgets

snac_model = SNAC.from_pretrained(
    "/home/zzh/code/TTS/Orpheus-TTS/pretrained_models/hubertsiuzdak/snac_24khz"
)
snac_model = snac_model.to("cpu")


# tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

#### CHANGE THIS ####

prompts = [
    "我也可以模仿人类的声音，比如叹气、咳嗽、打哈欠等。",
    "文鹏是深思考的算法工程师",
]
# ['卡齐娜', '莱依拉', '香菱', '提纳里', '辛焱', '娜维娅', '阿贝多', '茜特菈莉', '魈', '胡桃', '安柏', '枫原万叶', '芭芭拉', '久岐忍', '柯莱', '宵宫', '纳西妲', '托马', '珊瑚宫心海', '凝光', '林尼', '莱欧斯利', '赛诺', '重云', '五郎', '雷电将军', '迪奥娜', '流浪者', '达达利亚', '派蒙', '莫娜', '烟绯', '甘雨', '砂糖', '戴因斯雷布', '芙宁娜', '鹿野院平藏', '希诺宁', '琴', '夜兰', '云堇', '温迪', '珐露珊', '丽莎', '迪希雅', '迪卢克', '优菈', '克洛琳德', '可莉', '玛薇卡', '那维莱特', '八重神子', '妮露', '凯亚', '荒泷一斗', '钟离', '夏洛蒂', '恰斯卡', '诺艾尔', '行秋', '艾尔海森', '刻晴', '卡维', '神里绫人', '北斗', '班尼特', '玛拉妮', '早柚', '神里绫华']
chosen_voice = "雷电将军"  # see github for other voices


prompts = [f"{chosen_voice}: " + p for p in prompts]

all_input_ids = []

for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    all_input_ids.append(input_ids)

start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
end_tokens = torch.tensor(
    [[128009, 128260]], dtype=torch.int64
)  # End of text, End of human

all_modified_input_ids = []
for input_ids in all_input_ids:
    modified_input_ids = torch.cat(
        [start_token, input_ids, end_tokens], dim=1
    )  # SOH SOT Text EOT EOH
    all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max(
    [modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids]
)
for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat(
        [torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1
    )
    attention_mask = torch.cat(
        [
            torch.zeros((1, padding), dtype=torch.int64),
            torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64),
        ],
        dim=1,
    )
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
all_attention_masks = torch.cat(all_attention_masks, dim=0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")
# @title Generate Output
print(
    "*** Model.generate is slow - see vllm implementation on github for realtime streaming and inference"
)
print(
    "*** Increase/decrease inference params for more expressive less stable generations"
)

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
    )

# @title Parse Output as speech
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []

for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)


def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    audio_hat = snac_model.decode(codes)
    return audio_hat


my_samples = []
for code_list in code_lists:
    samples = redistribute_codes(code_list)
    my_samples.append(samples)

# from IPython.display import display, Audio

# if len(prompts) != len(my_samples):
#     raise Exception("Number of prompts and samples do not match")
# else:
#     for i in range(len(my_samples)):
#         print(prompts[i])
#         samples = my_samples[i]
#         display(Audio(samples.detach().squeeze().to("cpu").numpy(), rate=24000))
# # 改写成保存至本地

for i in range(len(my_samples)):
    print(prompts[i])
    samples = my_samples[i]
    sf.write(f"output/{i}.wav", samples.detach().squeeze().to("cpu").numpy(), 24000)
