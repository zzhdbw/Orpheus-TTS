# Import any necessary modules
# import snac  # Uncommented as per your initial code
import torch
frames = []
# my_tensors = []
import torch
from snac import SNAC
import numpy as np

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

snac_device = "cuda:1"
model = model.to(snac_device)


def turn_token_into_id(token, index): 
    div_seven_remainder = index % 7
    if token.startswith("<custom_token_") and token.endswith(">"):
        try:
            number_str = token[14:-1]  
            return int(number_str) - 10 - (div_seven_remainder*4096)
        except ValueError:
            pass
  
    

def convert_to_audio(multiframe, count):
  if len(multiframe) < 7:
    return
  
  codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
  codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

  num_frames = len(multiframe) // 7
  frame = multiframe[:num_frames*7]

  for j in range(num_frames):
    i = 7*j
    if codes_0.shape[0] == 0:
      codes_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
    else:
      codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)])

    if codes_1.shape[0] == 0:
      
      codes_1 = torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    else:
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=snac_device, dtype=torch.int32)])
      codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=snac_device, dtype=torch.int32)])
    
    if codes_2.shape[0] == 0:
      codes_2 = torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])
    else:
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=snac_device, dtype=torch.int32)])
      codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=snac_device, dtype=torch.int32)])

  codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
  with torch.inference_mode():
    audio_hat = model.decode(codes)
  
  audio_slice = audio_hat[:, :, 2048:4096]
  detached_audio = audio_slice.detach().cpu()
  audio_np = detached_audio.numpy()
  audio_int16 = (audio_np * 32767).astype(np.int16)
  audio_bytes = audio_int16.tobytes()
  return audio_bytes



def tokens_decoder(token_gen):
    buffer = []
    count = 0
    for token_sim in token_gen:        
        token = turn_token_into_id(token_sim, count)

        if token is None:
           yield token_sim
        else:
            if token >0:
                buffer.append(token)
                count += 1

                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

