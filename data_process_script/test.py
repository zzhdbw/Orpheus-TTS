import pyarrow.parquet as pq
from datasets import load_dataset

# try:
#     table = pq.read_table("../data/processed/yuanshen/data-00000-of-00001_SFT.parquet")
#     print("文件正常，无损坏。")
# except Exception as e:
#     print(f"文件可能已损坏，错误信息：{e}")

try:
    table = load_dataset("../data/processed/yuanshen/data-00000-of-00001_SFT.parquet")
    print("文件正常，无损坏。")
except Exception as e:
    print(f"文件可能已损坏，错误信息：{e}")
