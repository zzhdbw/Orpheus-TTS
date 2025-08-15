from snac import SNAC
import soundfile as sf
import torchaudio.transforms as T
import torch

SNAC_model_path = "../pretrained_models/hubertsiuzdak/snac_24khz"
model = SNAC.from_pretrained(SNAC_model_path).to(f"cuda")

data_path = "../data/raw/yuanshen/中文 - Chinese/「昂利」/vo_SGWLQ002_2_eric_01.wav"
data_path = (
    "../data/raw/yuanshen/中文 - Chinese/「此地的伟大精神」/vo_NTAQ007_9_wayob_01.wav"
)
#################
waveform, sample_rate = sf.read(data_path)
print(f"data: {waveform.shape},sample_rate: {sample_rate}")

waveform = torch.from_numpy(waveform).unsqueeze(0)
waveform = waveform.to(dtype=torch.float32)
print(f"waveform: {waveform.shape}")

resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
waveform1 = resample_transform(waveform).unsqueeze(0)

######################

waveform, sample_rate = sf.read(data_path)
print(f"data: {waveform.shape},sample_rate: {sample_rate}")

waveform = torch.from_numpy(waveform).unsqueeze(0)
waveform = waveform.to(dtype=torch.float32)
print(f"waveform: {waveform.shape}")

resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
waveform2 = resample_transform(waveform).unsqueeze(0)
######################
print(f"waveform1: {waveform1.shape}")
print(f"waveform2: {waveform2.shape}")
waveform = [waveform1, waveform2]
waveform = torch.cat(waveform, dim=0).to(torch.float32).cuda()
print(f"resample_transform: {waveform.shape}")

codes = model.encode(waveform)
print(type(codes))
print(codes[0].shape)
print(codes[1].shape)
print(codes[2].shape)
