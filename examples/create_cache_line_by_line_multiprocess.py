"""

Input:

    input is a single line by line text file

Output:



"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)


from run_language_modeling import MODEL_CLASSES, TextDataset, LineByLineTextDataset, load_and_cache_examples, set_seed

import multiprocessing
import json

def tokenize_line(line: str, block_size: int) -> List[List[int]]:
    examples = []
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
    if len(tokenized_text) <= block_size:
        examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))
    else:
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--data_file", type=str, required=True, help="The data file to process")
    parser.add_argument("--line_by_line", action="store_true", help="Dont mix tokens across lines")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Model checkpoint for weights initialization (or None for scratch)")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    special_tokens = {"additional_special_tokens": ["<|tgt|>", "<|CITE|>"]}
    tokenizer.add_special_tokens(special_tokens)

    args.block_size = tokenizer.max_len_single_sentence

    assert os.path.isfile(args.data_file)
    logger.info("Creating features from dataset file at %s", args.data_file)

    style = 'linebyline' if args.line_by_line else 'standard'
    cached_features_file = os.path.join(os.path.dirname(args.data_file), args.model_type + f'_{style}_' + "_cached_lm_" + str(args.block_size) + "_" + os.path.basename(args.data_file).replace('.txt', '.json'))

    if args.line_by_line:

        with open(args.data_file, encoding="utf-8") as f:
            lines = [line.strip() for line in tqdm(f) if line.strip()]

        # multiprocessing the tokenization
        def _mp_tokenize_line(line: str):
            return tokenize_line(line=line, block_size=args.block_size)

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            tokenized_liness = p.map(_mp_tokenize_line, lines)

        with open(cached_features_file, 'w') as f_out:
            for tokenized_lines in tqdm(tokenized_liness):
                for tokenized_line in tokenized_lines:
                    json.dump(tokenized_line, f_out)
                    f_out.write('\n')

    else:
        with open(args.data_file, encoding="utf-8") as f:
            text = f.read()
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        with open(cached_features_file, 'w') as f_out:
            for i in range(0, len(tokenized_text)-args.block_size+1, args.block_size): # Truncate in block of block_size
                json.dump(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+args.block_size]), f_out)
                f_out.write('\n')
