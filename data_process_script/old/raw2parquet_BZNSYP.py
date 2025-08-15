import pandas as pd
import soundfile as sf
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import os

# === Configuration ===
metadata_path = "raw_datasets/BZNSYP/TMP/meta.txt"
audio_dir = "raw_datasets/BZNSYP/Wave"
output_parquet = "processed_datasets/BZNSYP/BZNSYP_array.parquet"

# === Load Metadata ===
texts = []
audio_bytes_list = []
audio_paths = []
sampling_rates = []

with open(metadata_path, "r", encoding="utf-8") as f:
    for line in tqdm(
        f.readlines(), desc="Reading metadata"
    ):  # tqdm(f, desc="Reading metadata"):
        line = line.strip()
        if not line or "|" not in line:
            continue

        file_id, text = line.split("|", 1)
        text = text.strip()
        audio_path = os.path.join(audio_dir, f"{file_id}.wav")

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for {file_id}")
            continue

        try:
            # Read audio bytes
            with open(audio_path, "rb") as af:
                audio_bytes = af.read()

            # Get sampling rate (we discard the array here)
            _, sr = sf.read(audio_path)

            texts.append(text)
            audio_bytes_list.append(audio_bytes)
            audio_paths.append(audio_path)
            sampling_rates.append(sr)

        except Exception as e:
            print(f"Error processing {file_id}: {e}")

# === Convert to Arrow Table with Proper Structs ===
print(f"Saving {len(texts)} records to {output_parquet}")

# Build Arrow StructArray for audio field
audio_struct_array = pa.StructArray.from_arrays(
    [
        pa.array(audio_bytes_list, type=pa.large_binary()),
        pa.array(audio_paths, type=pa.string()),
    ],
    fields=[pa.field("bytes", pa.large_binary()), pa.field("path", pa.string())],
)

speaker_ids = ["标准女音"] * len(texts)

# Build Arrow Table
table = pa.table(
    {
        "text": pa.array(texts, type=pa.string()),
        "audio": audio_struct_array,
        "sampling_rate": pa.array(sampling_rates, type=pa.int32()),
        "speaker_id": pa.array(speaker_ids, type=pa.string()),
    }
)

# Write to Parquet
pq.write_table(table, output_parquet)
print("Done.")
