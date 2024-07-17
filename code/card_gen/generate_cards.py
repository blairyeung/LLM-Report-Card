import os
from concurrent.futures import wait

import math
from typing import List
from peripherals.oracle import compute_baseline_accuracy
from eval.eval_contrastive_full import *
from card_gen.few_shot import *
import argparse
from card_gen.generative_method import *
from card_gen.contrastive_method import *
import matplotlib.pyplot as plt

import os

# Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Change the current working directory to the parent directory
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

ANTHROPIC_TOPICS = ['corrigible-less-HHH',
                    'myopic-reward',
                    'power-seeking-inclination',
                    'survival-instinct',
                    'self-awareness-general-ai']


def generative_method(meta_topic='mmlu',
                      topic='high_school_physics', 
                      evaluator='gpt',
                      epoch=5, 
                      student_models=['Mistral-7B-Instruct-v0.2'],
                      card_format='dict',
                      max_worker=5, 
                      kwargs=None):

    assert meta_topic in ['mmlu',
                         'anthropic-eval',
                         'gsm8k',
                         'hotpot_qa',
                         'openend',]
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = kwargs[0]


    topics = [topic]
    
    supported_models = [
        'Mistral-7B-Instruct-v0.2',
        'Mixtral-8x7B-Instruct-v0.1',
        GPT_4_MODEL_NAME,
        GPT_3_MODEL_NAME,
        'Meta-Llama-3-8B-Instruct',
        'Meta-Llama-3-70B-Instruct'
        
    ]

    if student_models == ['all']:
        student_models = supported_models

    assert 40 % epoch == 0, "Epoch should be a factor of 40"

    train_batches = [int(40 / epoch)] * epoch

    if len(train_batches) == 1:
        optim_method = 'one-pass'
    
    else:
        optim_method = 'prog-reg'

    hps = []
    for topic in topics:
        for model in student_models:
            exp = f'generative/{topic}/{optim_method}/{card_format}/{evaluator}/{model}'
            # Make dir if not exist
            os.makedirs(exp, exist_ok=True)
 
            hp = {
                # general
                'experiment': exp,
                'name': f'{model}_main',
                'method': 'generative',
                'load_from': None,
                # dataset
                'dataset_folder': f'datasets/{meta_topic}',
                'topic': topic,
                'shuffle': False,
                'seed': 311,
                'batch_nums': train_batches + [60],
                'model': model,
                # training
                'evaluator': evaluator,
                'card_format' : card_format,
                'epoch': None,  # None: use the number of training batches as epoch
                'use_refine': False,
                'initial_criteria': get_initial_criteria(topic),
                'CoT': False,
                # eval
                'async_eval': True,
            }

            hps.append(hp)

    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(GenerativeMethod(hp, CostManager()).main) for hp in hps]
        futures = wait(futures)
        for future in futures.done:
            print(future.result())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, help='Met topic to run', default='openend')
    parser.add_argument('--topic', type=str, help='Topic to run', default='Writing efficient code for solving concrete algorthimic problems')
    parser.add_argument('--model', type=str, help='Models to run', default='all')
    # parser.add_argument('--model2', type=str, help='Models to run', default='Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument('--epoch', type=str, help='No. epochs to run', default='5')
    parser.add_argument('--evaluator', type=str, help='Evaluator to run', default='gpt')
    parser.add_argument('--format', type=str, help='Format of the card', default='dict')

    args = parser.parse_args()

    generative_method(args.meta,
                      args.topic, 
                      evaluator=args.evaluator, 
                      student_models=[args.model], 
                      epoch=int(args.epoch), 
                      card_format=args.format)