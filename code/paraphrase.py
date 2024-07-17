from core.models import *
from core.utils import *
import pandas as pd
import json
import jsonlines
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
import re
from core.config import *
from tqdm import tqdm, trange


def helper(index, query, completion, model, verdict=True):
    
    sys_prompt = """You are a good paraphraser. You paraphrase the text to hide any style of the original and make the author undistinguishable. You preserve the meaning and logical flow of the text. Respond using JSON format."""
    model = select_model(model, sys_prompt)
    prompt = f"""
    Question for the original author: {query}
    The original completion: {completion}
    You may and you should remove any structural information that may reveal the original author.
    Respond using the following JSON format. You must firstly analyze the logical flow, then do the paraphrase. Make sure that your paraphrase is still readable.
    [[FORMATTING]] Follow the format exactly, NO extra spaces or characters before or after this.
    {{
    "logical_flow_analysis": "...",
    "paraphrase": "Your paraphrase here"
    }}
    """
    completion = model(prompt, use_json=True)
    time.sleep(10)
    return index, completion['paraphrase']

def paraphrase_anthropic(topic, models):
    train_ds_path = f'datasets/anthropic-eval/_raw/train/{topic}.jsonl'
    test_ds_path = f'datasets/anthropic-eval/_raw/test/{topic}.jsonl'

    with jsonlines.open(train_ds_path) as reader:
        train_data = [line for line in reader]

    with jsonlines.open(test_ds_path) as reader:
        test_data = [line for line in reader]

    ds_keys = ['question', 
                'answer_matching_behavior', 
                'answer_not_matching_behavior']


    all_data = [train_data, test_data]

    for i in tqdm([0, 1], desc='Processing train/test'):
        prefix = 'train' if i == 0 else 'test'
        data = all_data[i]
        questions = [d['question'] for d in data]

        m = models[0]
        for model in tqdm(models, desc=f'Processing {m}'):
            # check if the file already exists, if yes, skip
            model_name = model.split('/')[-1]
            ds_path = f'datasets/anthropic-eval/{topic}/{model_name}-{prefix}.jsonl'
            if os.path.exists(ds_path):
                continue

            if 'claude' in model.lower():
                max_workers = 5

            elif 'gpt-4' in model:
                max_workers = 20

            else:
                max_workers = 5
            
            print(f'Processing with {max_workers} workers for {model}')

            m = model
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(helper, j, questions[j], model) for j in range(len(questions))]
                results = {}
                for future in tqdm(as_completed(futures), desc=f'Processing {model}'):
                    index, completion, verdict = future.result()
                    results[index] = {'completion': completion, 'verdict': verdict}
        
            for i in range(len(data)):
                data[i]['model_answer'] = results[i]['verdict']
                data[i]['completion'] = results[i]['completion']

            directory_path = f'datasets/anthropic-eval/{topic}'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # save
            # split and use last part of model name
            model_name = model.split('/')[-1]
            ds_path = f'datasets/anthropic-eval/{topic}/{model_name}-{prefix}.jsonl'
            with jsonlines.open(ds_path, 'w') as writer:
                writer.write_all(data)

            time.sleep(20)

def paraphrase_openend(topic, models):
    train_ds_path = f'datasets/openend/_raw/train/{topic}.jsonl'
    test_ds_path = f'datasets/openend/_raw/test/{topic}.jsonl'

    with jsonlines.open(train_ds_path) as reader:
        train_data = [line for line in reader]

    with jsonlines.open(test_ds_path) as reader:
        test_data = [line for line in reader]

    ds_keys = ['question', 
                'answer_matching_behavior', 
                'answer_not_matching_behavior']

    all_data = [train_data, test_data]

    for i in tqdm([0, 1], desc='Processing train/test'):
        prefix = 'train' if i == 0 else 'test'
        data = all_data[i]
        questions = [d['question'] for d in data]
        

        m = models[0]
        for model in tqdm(models, desc=f'Processing {m}'):
            # check if the file already exists, if yes, skip
            model_name = model.split('/')[-1]
            ds_path = f'datasets/openend/{topic}/{model_name}-{prefix}.jsonl'
            if os.path.exists(ds_path):
                continue

            m = model
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(helper, j, questions[j], model, False) for j in range(len(questions))]
                results = {}
                for future in tqdm(as_completed(futures), desc=f'Processing {model}'):
                    index, completion = future.result()
                    results[index] = {'completion': completion}
        
            for i in range(len(data)):
                data[i]['completion'] = results[i]['completion']

            directory_path = f'datasets/openend/{topic}'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # save
            # split and use last part of model name
            model_name = model.split('/')[-1]
            ds_path = f'datasets/openend/{topic}/{model_name}-{prefix}.jsonl'
            with jsonlines.open(ds_path, 'w') as writer:
                writer.write_all(data)

            time.sleep(60)

def format_choice_str(choices):
    map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    choice_str = ''
    for i, c in enumerate(choices):
        choice_str += f'{map[i]}. {c}\n'

    return choice_str

def format_mmlu_query(query):
    question = query['question']
    choices = query['choices']
    choices_str = format_choice_str(choices)
    query_str = f'{question}\n\n{choices_str}'

    return query_str

def paraphrase_mmlu(topic, models):
    train_ds_path = f'datasets/mmlu/_raw/train/{topic}.jsonl'
    test_ds_path = f'datasets/mmlu/_raw/test/{topic}.jsonl'
    
    with jsonlines.open(train_ds_path) as reader:
        train_data = [line for line in reader]

    with jsonlines.open(test_ds_path) as reader:
        test_data = [line for line in reader]

    data = [train_data, test_data]
    prefix = ['train', 'test']

    for model in tqdm(models, desc='Processing models'):
        for j in trange(2, desc='Processing train/test'):
            
            d = data[j]
            pre = prefix[j]
            queries = [format_mmlu_query(entry) for entry in d]
            model_name = model.split('/')[-1]


            ds_path = f'datasets/mmlu/{topic}/{model_name}-{pre}.jsonl'
            
            # read the file
            with jsonlines.open(ds_path) as reader:
                d = [line for line in reader]

            if 'paraphrased' in d[0]:
                continue
            
            completions = [d[i]['completion'] for i in range(len(d))]
            # use gpt-3.5

            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(helper, j, queries[j], completions[j], 'gpt-4-turbo', True) for j in range(len(queries))]
                results = {}
                for future in tqdm(as_completed(futures), desc=f'Processing {model}'):
                    index, completion = future.result()
                    results[index] = {'completion': completion}
    
            for i in range(len(d)):
                d[i]['paraphrased'] = results[i]['completion']

                # pop key 'models'
                if 'models' in d[i]:
                    d[i].pop('models')

            directory_path = f'datasets/mmlu/{topic}'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # save
            ds_path = f'datasets/mmlu/{topic}/{model_name}-{pre}.jsonl'
            with jsonlines.open(ds_path, 'w') as writer:
                writer.write_all(d)


def format_gsm8k_query(query):
    question = query['question']
    query_str = f'{question}'

    return query_str

def paraphrase_gsm(topic='gsm8k', models=ALL_STUDENT_MODELS):
    train_ds_path = f'datasets/gsm8k/_raw/train/{topic}.jsonl'
    test_ds_path = f'datasets/gsm8k/_raw/test/{topic}.jsonl'
    
    with jsonlines.open(train_ds_path) as reader:
        train_data = [line for line in reader]

    with jsonlines.open(test_ds_path) as reader:
        test_data = [line for line in reader]

    data = [train_data, test_data]
    prefix = ['train', 'test']

    for model in tqdm(models, desc='Processing models'):
        for j in trange(2, desc='Processing train/test'):
            
            d = data[j]
            pre = prefix[j]
            queries = [format_gsm8k_query(entry) for entry in d]
            model_name = model.split('/')[-1]

            ds_path = f'datasets/gsm8k/{model_name}-{pre}.jsonl'
            
            # read the file
            with jsonlines.open(ds_path) as reader:
                d = [line for line in reader]
    

            if 'paraphrased' in d[0]:
                continue
            
            completions = [d[i]['completion'] for i in range(len(d))]
            # use gpt-3.5

            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(helper, j, queries[j], completions[j], 'gpt-4-turbo', True) for j in range(len(queries))]
                results = {}
                for future in tqdm(as_completed(futures), desc=f'Processing {model}'):
                    index, completion = future.result()
                    results[index] = {'completion': completion}
    
            for i in range(len(d)):
                d[i]['paraphrased'] = results[i]['completion']

                # pop key 'models'
                if 'models' in d[i]:
                    d[i].pop('models')

            directory_path = f'datasets/gsm8k/{topic}'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # save
            ds_path = f'datasets/gsm8k/{topic}/{model_name}-{pre}.jsonl'
            with jsonlines.open(ds_path, 'w') as writer:
                writer.write_all(d)


if __name__ == '__main__':
    mmlu_topics = [
                #    'high_school_mathematics',
                #    'college_mathematics',
                #    'high_school_physics',
                #    'high_school_chemistry',
                #    'high_school_biology',
                #    'high_school_world_history',
                   'machine_learning']
    
    models = ['gpt-4o',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'mistralai/Mixtral-8x7B-Instruct-v0.1', 
            #  'gpt-4-turbo',
             GPT_3_MODEL_NAME,
            'meta-llama/Meta-Llama-3-70B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct',]
            # 'google/gemma-1.1-7b-it']

    import os

    # Get the current working directory
    current_dir = os.getcwd()
    print("Current Directory:", current_dir)

    # Change the current working directory to the parent directory  
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    # split_data_json(meta='mmlu', topics=mmlu_topics, split_ratio=0.4, max=100)

    for topic in tqdm([''], desc=f'Processing topics'):
        t = topic
        paraphrase_gsm(topic='gsm8k', models=models)
        # paraphrase_mmlu(topic=topic, models=models)
        time.sleep(10)
