"""
@Desc:
@Reference:
Our preprocess code refers to :
https://github.com/UCSD-AI4H/COVID-Dialogue/blob/master/src/preprocess.py
Bertopic 使用:
https://www.51cto.com/article/679535.html
G2P 使用：
https://blog.csdn.net/u013625492/article/details/110877174
@Notes:
"""
import os
import sys
from pathlib import Path
import json
from typing import List, Dict
import numpy as np
from g2p_en import G2p

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.utils.file_utils import save_json
from bertopic import BERTopic
np.random.seed(22)

def enhance_tt_dataset():
    src_dir = Path(f"{BASE_DIR}/resources/tongue_twister/raw-data")
    tgt_dir = Path(f"{BASE_DIR}/resources/tongue_twister/raw-data")
    tgt_dir.mkdir(exist_ok=True, parents=True)
    origin_data = json.load(src_dir.joinpath("twister_dataset.json").open("r", encoding="utf-8"))
    new_data = []
    count = 0
    g2p = G2p()
    topic_model = BERTopic(verbose=True)
    corpus = load_tongue_twister_dataset()
    topics, probabilities = topic_model.fit_transform(corpus)
    topic_ids = topic_model.get_topic_info()["Topic"].tolist()
    topic_names = topic_model.get_topic_info()["Name"].tolist()
    topic_name_map = dict([(idx, topic.strip().split("_")[1:]) for (idx, topic) in zip(topic_ids, topic_names)])
    topic_name_map[-1] = []
    bad_cases = []
    for one in origin_data:
        topic_keywords = " ".join(topic_name_map[topics[count]])
        temp = {"tt_idx": count,
                "rake_keywords": one["keywords"],
                "rake_keywords_phonetics": " ".join([f"[{' '.join(g2p(word))}]" for word in one["keywords"].strip().split()]),
                "bertopic_keywords": topic_keywords,
                "bertopic_keywords_phonetics": " ".join([f"[{' '.join(g2p(word))}]" for word in topic_keywords.strip().split()]),
                "tt_text": one["tongue-twister"],
                "tt_text_phonetics": one["phonetic-transcription"],
                "source": one["source"],
                }
        count += 1
        # filter ---------------
        if temp["rake_keywords"] == temp["tt_text"]:
            bad_cases.append(temp)
        elif len(temp["rake_keywords"])>=len(temp["tt_text"])*0.8:  # too long
            bad_cases.append(temp)
        elif len(temp["rake_keywords"])==0:  # too short
            bad_cases.append(temp)
        else:
            new_data.append(temp)
    print(f"original data size: {len(origin_data)}; Dataset size: {len(new_data)}; Badcase size: {len(bad_cases)}.")
    save_json(new_data, tgt_dir.joinpath("enhanced_twister_dataset.json"))
    save_json(new_data, tgt_dir.joinpath("bad_cases.json"))

def load_tongue_twister_dataset():
    src_dir = Path(f"{BASE_DIR}/resources/tongue_twister/raw-data")
    origin_data = json.load(src_dir.joinpath("twister_dataset.json").open("r", encoding="utf-8"))
    corpus = []
    for one in origin_data:
        corpus.append(one["tongue-twister"])
    return corpus

def run_bertopic():
    src_dir = Path(f"{BASE_DIR}/resources/tongue_twister/raw-data")
    model = BERTopic(verbose=True)
    corpus = load_tongue_twister_dataset()
    topics, probabilities = model.fit_transform(corpus)

    # 获得的主题
    print(model.get_topic_freq().head(11))
    model.get_topic_info().to_csv(src_dir.joinpath("topics.csv").absolute(), encoding='utf-8')

    fig = model.visualize_heatmap()
    fig.write_html(src_dir.joinpath("topic_heat_map.html"))
    fig = model.visualize_topics()
    fig.write_html(src_dir.joinpath("topic_visual.html"))



if __name__ == '__main__':
    enhance_tt_dataset()
    # run_bertopic()