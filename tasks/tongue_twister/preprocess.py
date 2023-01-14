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


def data_analyze(data):
    vocab = set()
    phonetic_vocab = set()
    tt_text = []
    tt_phone = []
    rake_words = []
    input_phone = []
    bertopic_words = []
    for a_sample in data:
        tt_text.append(a_sample["tt_text"].strip().split())
        vocab.update(a_sample["tt_text"].strip().split())
        tt_phone.append([p for one in a_sample["tt_text_phonetics"].strip().split() for p in one[1:-1].split()])
        input_phone.append([p for one in a_sample["rake_keywords_phonetics"].strip().split() for p in one[1:-1].split()])
        phonetic_vocab.update([p for one in a_sample["tt_text_phonetics"].strip().split() for p in one[1:-1].split()])
        rake_words.append(a_sample["rake_keywords"].strip().split())
        bertopic_words.append(a_sample["bertopic_keywords"].strip().split())

    stats = {"tt_samples:": len(data),
             "tt_vocab:": len(vocab),
             "phobenmes:": len(phonetic_vocab),
             "rake_words:": len(set([word for one in rake_words for word in one])),
             "bertopic_words:": len(set([word for one in bertopic_words for word in one])),
             "avg. rake": np.mean([len(one) for one in rake_words]),
             "avg. tt": np.mean([len(one) for one in tt_text]),
             "avg. input phone": np.mean([len(one) for one in input_phone]),
             "avg. output phone": np.mean([len(one) for one in tt_phone]),
    }
    print(f"data stats: {stats}")

def preprocess_tongue_twister_dataset(tgt_dir: Path, process_func=None, analyze_flag=False):
    src_dir = Path(f"{BASE_DIR}/resources/tongue_twister/raw-data")
    tgt_dir.mkdir(exist_ok=True, parents=True)
    data_list = json.load(src_dir.joinpath("enhanced_twister_dataset.json").open("r", encoding="utf-8"))
    train_data, val_data, test_data = split_data(data_list)
    if analyze_flag:
        print(f"analyze train_data =============")
        data_analyze(train_data)
        print(f"analyze val_data =============")
        data_analyze(val_data)
        print(f"analyze test_data =============")
        data_analyze(test_data)
        print(f"analyze total =============")
        data_analyze(data_list)
    print(f"output_dir:{tgt_dir}: train: {len(train_data)}; val: {len(val_data)}; test: {len(test_data)}; ")
    output_dataset(data=train_data, prefix="train", output_dir=tgt_dir, process_func=process_func)
    output_dataset(data=val_data, prefix="val", output_dir=tgt_dir, process_func=process_func)
    output_dataset(data=test_data, prefix="test", output_dir=tgt_dir, process_func=process_func)

def normalize_text(text: str):
    return text.replace("\n", " ").strip()

def process_raw_data_with_prompt(data: List[Dict]):
    src_txt_list = []
    tgt_txt_list = []
    keyword_list = []
    for one in data:
        input_text = one["rake_keywords"]
        src_txt_list.append(f'Generate tongue twisters about key words: {normalize_text(input_text)}\n')
        tgt_txt_list.append(f'{normalize_text(one["tt_text"])}\n')
    assert len(src_txt_list) == len(tgt_txt_list)
    return src_txt_list, tgt_txt_list

def process_raw_data(data: List[Dict]):
    src_txt_list = []
    tgt_txt_list = []
    for one in data:
        input_text = one["rake_keywords"]
        src_txt_list.append(f'{normalize_text(input_text)}\n')
        tgt_txt_list.append(f'{normalize_text(one["tt_text"])}\n')
    assert len(src_txt_list) == len(tgt_txt_list)
    return src_txt_list, tgt_txt_list

def output_dataset(data, prefix, output_dir:Path, process_func=None):
    output_dir.mkdir(exist_ok=True)
    src_txt_list, tgt_txt_list = process_func(data)
    assert len(src_txt_list) == len(tgt_txt_list)
    output_dir.joinpath(f"{prefix}.source.txt").open("w", encoding="utf-8").writelines(src_txt_list)
    output_dir.joinpath(f"{prefix}.target.txt").open("w", encoding="utf-8").writelines(tgt_txt_list)

def split_data(data_frames):
    np.random.shuffle(data_frames)
    # ratio: 9/0.5/0.5
    total_id_num = len(data_frames)
    validate_idx = int(float(total_id_num) * 9 / 10)
    test_idx = int(float(total_id_num) * 9.5 / 10)

    train_data = data_frames[:validate_idx]
    val_data = data_frames[validate_idx:test_idx]
    test_data = data_frames[test_idx:]
    return train_data, val_data, test_data

if __name__ == '__main__':
    analyze_flag = False
    preprocess_tongue_twister_dataset(tgt_dir = Path(f"{BASE_DIR}/datasets/tongue_twister/tt-data"),
                                      process_func=process_raw_data, analyze_flag=analyze_flag)
    preprocess_tongue_twister_dataset(tgt_dir = Path(f"{BASE_DIR}/datasets/tongue_twister/tt-prompt-data"),
                                      process_func=process_raw_data_with_prompt)