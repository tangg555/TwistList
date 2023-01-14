"""
@Desc:
@Reference:
Bert Score:
https://huggingface.co/spaces/evaluate-metric/bertscore
PPl:
https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
gpt roberta ppl:
https://blog.csdn.net/qq_36605433/article/details/120540172
@Notes:
"""

import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
import nltk

from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaConfig
from evaluate import load
from g2p_en import G2p
from tqdm import tqdm
from collections import Counter

def compute_phonemes(predictions: List, display=False):
    g2p = G2p()
    def parse_text_to_phonemes(text):
        phoneme_list = []
        for word in text.strip().split():
            phoneme_list.append(g2p(word))
        return phoneme_list

    record = {"init_po_count": [],
              "po_count": []}

    bar = predictions if not display else tqdm(predictions, desc="compute_phonemes")
    for pred in bar:
        pred_phonemes = parse_text_to_phonemes(pred)

        init_po_counter = Counter()
        po_count = Counter()
        word_count =0
        for word_phonemes in pred_phonemes:
            if len(word_phonemes) == 0:
                continue
            word_count += 1
            init_po_counter[word_phonemes[0]] += 1
            for po in word_phonemes:
                po_count[po] += 1
        record["init_po_count"].append(init_po_counter.most_common(1)[0][1]/word_count)
        record["po_count"].append(po_count.most_common(1)[0][1]/word_count)

    metric_dict = {"init_po_count": round(np.mean(record["init_po_count"]), 4),
                    "po_count": round(np.mean(record["po_count"]), 2)
                   }
    return metric_dict

def compute_bert_score(predictions: List, references: List):
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    metric_dict = {"bertscore_precision": round(np.mean(scores["precision"]), 4),
                   "bertscore_recall": round(np.mean(scores["recall"]), 4),
                   "bertscore_f1": round(np.mean(scores["f1"]), 4)
                   }
    return metric_dict

def compute_gpt_ppl(sentences: List[str]):
    config: GPT2Config = GPT2Config.from_pretrained("gpt2")
    # tokenizer
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    inputs = tokenizer(sentences,
                       padding='longest',
                       max_length=config.max_length,
                       add_special_tokens=True,
                       truncation=True,
                       return_tensors="pt")
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    return ppl

def compute_roberta_ppl(sentences: List[str]):
    config: RobertaConfig = RobertaConfig.from_pretrained("roberta-base")
    # tokenizer
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base", config=config)
    with torch.no_grad():
        sentence_ppl = []
        for sentence in tqdm(sentences, desc="compute_roberta_ppl"):
            tokenize_input = tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            sen_len = len(tokenize_input)
            sentence_loss = 0.

            for i, word in enumerate(tokenize_input):
                # add mask to i-th character of the sentence
                tokenize_input[i] = '[MASK]'
                mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

                output = model(mask_input)

                prediction_scores = output[0]
                softmax = torch.nn.Softmax(dim=0)
                ps = softmax(prediction_scores[0, i]).log()
                word_loss = ps[tensor_input[0, i]]
                sentence_loss += word_loss.item()

                tokenize_input[i] = word
            ppl = np.exp(-sentence_loss / sen_len)
            sentence_ppl.append(ppl)
    return np.mean(sentence_ppl)

if __name__ == '__main__':
    inputs = ["sells thick socks", "sells thick socks"]
    references = ["Seth at Sainsbury's sells thick socks.", "You cuss, I cuss, we all cuss, for asparagus!"]
    predictions = ["Seth at Sainsbury's sells thicks.", "You cuss, I asparagus!"]
    # gpt2 ppl ----
    # print(f"references ppl: {compute_gpt_ppl(sentences=references)}")
    # print(f"predictions ppl: {compute_gpt_ppl(sentences=predictions)}")
    # roberta ppl ----
    # print(f"references roberta: {compute_roberta_ppl(sentences=references)}")
    # print(f"predictions roberta: {compute_roberta_ppl(sentences=predictions)}")

    # # bert score ----
    # print(f"bert score: {compute_bert_score(predictions=predictions, references=references)}")

    print(f"compute_phonemes: {compute_phonemes(predictions=references)}")
