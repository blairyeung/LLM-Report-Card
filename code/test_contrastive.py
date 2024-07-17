import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
from core.config import *
import json
from tqdm import tqdm
import os
# from eval_everything import 
import jsonlines
import re
import pandas as pd
import random

from core.data import Batch, load_batches
from core.models import CostManager, select_model
from core.utils import ResourceManager
import random
from core.card import GenerativeCard
from eval.eval_partially_contrastive import ContrastiveCardEvaluator, ContrastiveAnswerEvaluator, get_qc_str_one_batch
# from eval.eval_scratch import ContrastiveAnswerEvaluator
from eval.eval_fully_contrastive import ContrastiveFullEvaluator, load_cards


def run_contrastive(params, random_seed=42):
    meta_topic = params.get('meta_topic', 'anthropic-eval')
    few_shot = params.get('few_shot', 0)
    model1 = params.get('model1', 'Mixtral-8x7B-Instruct-v0.1')
    ensemble_shots = params.get('ensemble_shots', 5)
    ensemble_cnt = params.get('ensemble_cnt', 1)
    diff_cnt = params.get('diff_cnt', -1)
    model2 = params.get('model2', 'Meta-Llama-3-70B-Instruct')
    topic = params.get('topic', 'high_school_mathematics')
    generation_method = params.get('generation_method', 'contrastive')
    guesser = params.get('guesser', 'meta-llama/Meta-Llama-3-70B-Instruct')
    card_format = params.get('card_format', 'dict')
    evaluator = params.get('evaluator', 'gpt')
    eval_method = params.get('eval_method', 'card')
    max_workers = params.get('max_workers', 40)

    
    models = [model1, model2]

    random.seed(random_seed)

    batch1 = load_batches(f"datasets/{meta_topic}/", topic, model1, "test", [60], False)[0]
    batch2 = load_batches(f"datasets/{meta_topic}/", topic, model2, "test", [60], False)[0]

    if few_shot > 0:
        train_batch1 = load_batches(f"datasets/{meta_topic}/", topic, model1, "train", [40], False)[0]
        train_batch2 = load_batches(f"datasets/{meta_topic}/", topic, model2, "train", [40], False)[0]

        indices = random.sample(range(len(train_batch1)), few_shot)

        few_shot_1 = [train_batch1[i] for i in indices]
        few_shot_2 = [train_batch2[i] for i in indices]

        few_shot_1 = get_qc_str_one_batch(meta_topic, train_batch1, indices)
        few_shot_2 = get_qc_str_one_batch(meta_topic, train_batch2, indices)

        cards = [few_shot_1, few_shot_2]

    else:
        cards = load_cards(topic, card_format, models, evaluator, method=generation_method)
        card1, card2 = cards[model1], cards[model2]
        cards = (str(card1), str(card2))

    batches = [batch1, batch2]

    if eval_method == 'card':
        eval_method = ContrastiveCardEvaluator(
            meta_topic,
            topic,
            batches,
            models,
            cards,
            guesser,
            ResourceManager("eval_contrastive-answer"),
            generation_method=generation_method,
            few_shot=few_shot,
            k_shots=ensemble_shots,
            max_workers=max_workers
        )

    elif eval_method == 'answer':
        eval_method = ContrastiveAnswerEvaluator(
            meta_topic,
            topic,
            batches,
            models,
            cards,
            guesser,
            ResourceManager("eval_contrastive-answer"),
            generation_method=generation_method,
            few_shot=few_shot,
            k_shots=ensemble_shots,
            ensemble_cnt=ensemble_cnt,
            diff_shots=diff_cnt,
            max_workers=max_workers
        )

    elif eval_method == 'full':
        eval_method = ContrastiveFullEvaluator(
            meta_topic,
            topic,
            batches,
            models,
            cards,
            guesser,
            ResourceManager("eval_contrastive-answer"),
            generation_method=generation_method,
            few_shot=few_shot,
            k_shots=ensemble_shots,
            ensemble_cnt=ensemble_cnt,
            diff_shots=diff_cnt,
            max_workers=max_workers
        )

    else:
        print("Invalid eval_method")
        return 
    
    if 'groq' in guesser:
        eval_method.main(max_workers=5)

    else:
        eval_method.main(max_workers=60)

def run_answer_contrastive(params):
    few_shot = params.get('few_shot', 0)

def run_full_contrastive(params):
    few_shot = params.get('few_shot', 0)

if __name__ == '__main__':
    args = ArgumentParser()
    # args.add_argument('--meta_topic', type=str, default='openend')
    args.add_argument('--meta_topic', type=str, default='mmlu')
    args.add_argument('--few_shot', type=int, default=0)
    args.add_argument('--ensemble_shots', type=int, default=3)
    args.add_argument('--ensemble_cnt', type=int, default=3)

    args.add_argument('--topic', type=str, default='high_school_mathematics')
    # args.add_argument('--topic', type=str, default='Writing efficient code for solving concrete algorthimic problems')
    args.add_argument('--model1', type=str, default='Mistral-7B-Instruct-v0.2')
    # args.add_argument('--model1', type=str, default='Meta-Llama-3-70B-Instruct')
    # args.add_argument('--model2', type=str, default='gpt-3.5-turbo')
    args.add_argument('--model2', type=str, default='gpt-4-turbo')
    # args.add_argument('--model2', type=str, default='Meta-Llama-3-70B-Instruct')
    # args.add_argument('--model2', type=str, default='Llama-2-13b-chat-hf')
    args.add_argument('--generation_method', type=str, default='generative')
    # guesser
    args.add_argument('--guesser', type=str, default='meta-llama/Meta-Llama-3-70B-Instruct')
    args.add_argument('--card_format', type=str, default='dict')
    args.add_argument('--evaluator', type=str, default='gpt')
    # eval_method
    args.add_argument('--eval_method', type=str, default='full')
    # max_workers
    args.add_argument('--max_workers', type=int, default=60)

  
    args = args.parse_args()

    params = args.__dict__

    # params['model2'] = GPT_4_MODEL_NAME

    run_contrastive(params)