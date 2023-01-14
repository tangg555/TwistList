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


def chatgpt_generate_responses(hparams):
    data_dir = Path(hparams.data_dir)
    resource_dir = Path(hparams.resource_dir)
    output_dir = Path(hparams.output_dir)


    preds = self.test_output["preds"]
    targets = self.test_output["tgts"]
    tgt_lines_toks, pred_lines_toks = \
        [self.tokenizer.tokenize(t) for t in targets], [self.tokenizer.tokenize(c) for c in preds]

    metrics = {}
    # calculate bleu score
    nlg_eval_utils.calculate_bleu(ref_lines=tgt_lines_toks, gen_lines=pred_lines_toks, metrics=metrics)
    # calculate rouge score
    rouge_metrics = nlg_eval_utils.calculate_rouge(pred_lines=preds, tgt_lines=targets)
    metrics.update(**rouge_metrics)
    phoneme_metrics = tt_eval_utils.compute_phonemes(predictions=preds, references=targets)
    metrics.update(**phoneme_metrics)
    bertscore_metrics = tt_eval_utils.compute_bert_score(predictions=preds, references=targets)
    metrics.update(**bertscore_metrics)
    gen_len = np.mean(list(map(len, preds)))
    metrics["gen_len"] = gen_len
    metrics["ppl"] = round(np.exp(metrics["loss"]), 2)
    key = sorted(metrics.keys())
    for k in key:
        print(k, metrics[k])
    print("=" * 10)

    print(f"model {self.model.model_name} eval {self.gen_file}")
    output_obj_to_file(json.dumps(metrics, indent=4), self.eval_file)
    return metrics

if __name__ == '__main__':
    hparams = parse_args_for_config()
    tester = chatgpt_generate_responses(hparams)