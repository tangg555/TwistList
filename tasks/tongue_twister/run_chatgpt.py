"""
@Desc:
@Reference:
- logger and WandLogger
Weights and Biases is a third-party logger
https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html
@Notes:

"""

import sys
import json
import numpy as np
from pathlib import Path

from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configuration.tongue_twister.config_args import parse_args_for_config
from src.utils.file_utils import copy_file_or_dir, output_obj_to_file, pickle_save, pickle_load
from src.utils import nlg_eval_utils
from src.utils.tongue_twister import tt_eval_utils
from train import TongueTwisterTrainer
from src.utils.string_utils import rm_extra_spaces
from src.utils.file_utils import pickle_save, pickle_load
from src.models.tongue_twister.chatgpt import chatgpt_generate
import time
from transformers import BartTokenizer

class ChatGPTTester(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.tokenizer: BartTokenizer = BartTokenizer.from_pretrained(self.hparams.model_name_or_path)
        self.data_dir = Path(self.hparams.data_dir)
        self.resources_dir = Path(self.hparams.resources_dir)
        self.output_dir = Path(self.hparams.output_dir)
        self.experiment_name = self.hparams.experiment_name
        self.experiment_output_dir = self.output_dir.joinpath(self.experiment_name)
        self.experiment_output_dir = Path(hparams.output_dir)
        self.generation_dir = self.experiment_output_dir / "gen_result"
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.experiment_output_dir / "cache_dir"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir.joinpath("chatgpt.pkl")
        self.cache = self.load_cache()

        self.src_data_file = self.data_dir.joinpath("test.source.txt")
        self.tgt_data_file = self.data_dir.joinpath("test.target.txt")
        self.src_data = self._read_clean_lines(self.src_data_file)
        self.tgt_data = self._read_clean_lines(self.tgt_data_file)

        self.gen_file = self.generation_dir / f"chatgpt_gen.txt"
        self.eval_file = self.generation_dir / f"chatgpt_eval.txt"

    def load_cache(self):
        if self.cache_path.exists():
            print(f"cache loaded from {self.cache_path}")
            return pickle_load(self.cache_path)
        else:
            return dict()

    def save_cache(self):
        pickle_save(self.cache, self.cache_path)


    def _read_clean_lines(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = rm_extra_spaces(line)
                if len(line) > 0:
                    data.append(line)
        return data

    def eval_golden(self):
        targets = self.tgt_data
        phoneme_metrics = tt_eval_utils.compute_phonemes(predictions=targets)
        print(phoneme_metrics)

    def clean_response(self, preds: list):
        new = []
        for one in preds:
            # new.append(one.replace("\n", "").strip())
            new.append(one.split("\n")[0].strip())
        return new

    def generate(self):
        preds = []
        targets = []
        for src_input, tt_target in tqdm(list(zip(self.src_data, self.tgt_data)), desc=f"ChatGPT generating......"):
            targets.append(tt_target)
            if src_input in self.cache and self.cache[src_input] is not None:
                preds.append(self.cache[src_input]['message'])
            else:
                result = chatgpt_generate(query=src_input)
                self.cache[src_input] = result
                self.save_cache()
                time.sleep(31)

        preds = self.clean_response(preds)
        with open(self.gen_file, "w", encoding="utf-8") as fw_out:
            fw_out.write("\n".join(preds))

    def eval(self):
        targets = self.tgt_data
        with open(self.gen_file, "r", encoding="utf-8") as fr:
            preds = [one.strip() for one in fr]

        tgt_lines_toks, pred_lines_toks = \
            [self.tokenizer.tokenize(l) for l in targets], [self.tokenizer.tokenize(l) for l in preds]

        metrics = {}
        # calculate bleu score
        nlg_eval_utils.calculate_bleu(ref_lines=tgt_lines_toks, gen_lines=pred_lines_toks, metrics=metrics)
        # calculate rouge score
        rouge_metrics = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
        metrics.update(**rouge_metrics)
        phoneme_metrics = tt_eval_utils.compute_phonemes(predictions=preds)
        metrics.update(**phoneme_metrics)
        phoneme_sim_metrics = tt_eval_utils.compute_phonemes_similarity(predictions=preds, references=targets)
        metrics.update(**phoneme_sim_metrics)
        bertscore_metrics = tt_eval_utils.compute_bert_score(predictions=preds, references=targets)
        metrics.update(**bertscore_metrics)
        gen_len = np.mean([len(one.strip().split()) for one in preds])
        metrics["gen_len"] = gen_len
        key = sorted(metrics.keys())
        for k in key:
            print(k, metrics[k])
        print("=" * 10)

        output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)
        print(f"store metrics to {self.eval_file}")
        return metrics

if __name__ == '__main__':
    hparams = parse_args_for_config()
    tester = ChatGPTTester(hparams)
    # tester.generate()
    tester.eval()