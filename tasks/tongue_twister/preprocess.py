"""
@Desc:
@Reference:
Our preprocess code refers to :
https://github.com/UCSD-AI4H/COVID-Dialogue/blob/master/src/preprocess.py
@Notes:
"""
import os
import sys
from pathlib import Path
import json
from typing import List, Dict
import numpy as np

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

np.random.seed(22)

DEST_DATA_DIR = "tongue_twister/tt-data"

def preprocess_tongue_twister_dataset():
    src_dir = Path(f"{BASE_DIR}/resources/raw-data")
    tgt_dir = Path(f"{BASE_DIR}/datasets/{DEST_DATA_DIR}")
    tgt_dir.mkdir(exist_ok=True, parents=True)
    data_list = json.load(src_dir.joinpath("twister_dataset.json").open("r", encoding="utf-8"))
    train_data, val_data, test_data = split_data(data_list)
    print(f"train: {len(train_data)}; val: {len(val_data)}; test: {len(test_data)}; ")
    output_dataset(data=train_data, prefix="train", output_dir=tgt_dir)
    output_dataset(data=val_data, prefix="val", output_dir=tgt_dir)
    output_dataset(data=test_data, prefix="test", output_dir=tgt_dir)

def split_src_and_tgt_data_file(data: List[Dict]):
    src_txt_list = []
    tgt_txt_list = []
    for one in data:
        input_text = one["keywords"] if len(one["keywords"]) > 0 else one["tongue-twister"].split()[0]
        src_txt_list.append(f'{input_text}\n')
        tgt_txt_list.append(f'{one["tongue-twister"]}\n')
    assert len(src_txt_list) == len(tgt_txt_list)
    return src_txt_list, tgt_txt_list

def output_dataset(data, prefix, output_dir:Path):
    output_dir.mkdir(exist_ok=True)
    src_txt_list, tgt_txt_list = split_src_and_tgt_data_file(data)
    output_dir.joinpath(f"{prefix}.source.txt").open("w", encoding="utf-8").writelines(src_txt_list)
    output_dir.joinpath(f"{prefix}.target.txt").open("w", encoding="utf-8").writelines(tgt_txt_list)

def split_data(data_frames):
    np.random.shuffle(data_frames)
    # ratio: 8/1/1
    total_id_num = len(data_frames)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    train_data = data_frames[:validate_idx]
    val_data = data_frames[validate_idx:test_idx]
    test_data = data_frames[test_idx:]
    return train_data, val_data, test_data

if __name__ == '__main__':
    preprocess_tongue_twister_dataset()