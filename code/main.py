import os
from concurrent.futures import wait

import math
from oracle import compute_baseline_accuracy
from eval_fully_contrastive import *
from few_shot import *
import argparse
from generative_method import *
from contrastive_method import *
import matplotlib.pyplot as plt


def contrastive_eval():
    topic = 'high_school_physics'
    validation = load_mmlu_batches(f'datasets/mmlu/', topic, model=GPT_3_MODEL_NAME,partition='test', batch_nums=[60])[0]
    models = ['Llama-2-13b-chat-hf', 'Mistral-7B-Instruct-v0.2']
    card1 = GenerativeCard(filename=r'outputs/generative_dict_high_school_physics/03-05_03-27-44_Llama-2-13b-chat-hf_main/cards/epoch_4_card.json')
    card2 = GenerativeCard(filename=r'outputs/generative_dict_high_school_physics/03-05_03-27-43_Mistral-7B-Instruct-v0.2_main/cards/epoch_4_card.json')

    rm = ResourceManager('generative_college_chemistry', 'eval_contrastive_gpt-3.5-turbo-1106_Mistral-7B-Instruct-v0.2_0')
    tm = CostManager()

    evaluator = ContrastiveEvaluator(topic, validation, models, [card1, card2], rm, tm)
    evaluator.main(1, 50)

    rm.shutdown()


def generative_method(meta_topic='mmlu',
                      topic='high_school_physics', 
                      evaluator='gpt',
                      epoch=5, 
                      student_models=['Mistral-7B-Instruct-v0.2'],
                      card_format='dict', 
                      kwargs=None):

    if kwargs is None:
        kwargs = {}
    else:
        kwargs = kwargs[0]


    topics = [topic]
    
    supported_models = [
        # 'Mistral-7B-Instruct-v0.2',
        # 'Mixtral-8x7B-Instruct-v0.1',
        'Llama-2-13b-chat-hf',
        'Llama-2-70b-chat-hf',
        'gpt-3.5-turbo-1106',
        # 'Meta-Llama-3-70B-Instruct'
    ]

    student_models = supported_models

    for model in student_models:
        if model not in supported_models:
            raise ValueError(f"Model {model} is not supported. Supported models are {supported_models}")

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

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(GenerativeMethod(hp, CostManager()).main) for hp in hps]
        futures = wait(futures)
        for future in futures.done:
            print(future.result())


def contrastive_method(topic,
                       evaluator='gpt', 
                       epoch=5,
                       student_models=[['Mistral-7B-Instruct-v0.2', 'Mixtral-8x7B-Instruct-v0.1']], 
                       card_format='dict', 
                       kwargs=None):

    # convert kwargs to dict of params
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = kwargs[0]


    topics = [topic]
    
    supported_models = [
        'Mistral-7B-Instruct-v0.2',
        'Mixtral-8x7B-Instruct-v0.1',
        'Llama-2-13b-chat-hf',
        'Llama-2-70b-chat-hf',
        'gpt-3.5-turbo-1106',
        'Meta-Llama-3-70B-Instruct'
    ]

    for model in student_models:
        if model[0] not in supported_models or model[1] not in supported_models:
            raise ValueError(f"Model {model} is not supported. Supported models are {supported_models}")

    assert 40 % epoch == 0, "Epoch should be a factor of 40"

    train_batches = [int(40 / epoch)] * epoch

    if len(train_batches) == 1:
        optim_method = 'one-pass'
    
    else:
        optim_method = 'prog-reg'

    hps = []
    for topic in topics:
        for model in student_models:
            exp = f'contrastive/{topic}/{optim_method}/{card_format}/{evaluator}/{model}'
            # Make dir if not exist
            os.makedirs(exp, exist_ok=True)

            hp = {
                # general
                'experiment': exp,
                'name': f'{model}_main',
                'method': 'contrastive',
                'load_from': None,
                # dataset
                'dataset_folder': 'datasets/mmlu',
                'topic': topic,
                'shuffle': False,
                'seed': 311,
                'batch_nums': train_batches + [60],
                'model1': model[0],
                'model2': model[1],
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

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(ContrastiveMethod(hp, CostManager()).main) for hp in hps]
        futures = wait(futures)
        for future in futures.done:
            print(future.result())


def generative_method_anthropic():
    topic = 'self-awareness-general-ai'
    # models = ['Mixtral-8x7B-Instruct-v0.1', 'Mistral-7B-Instruct-v0.2', 'Llama-2-13b-chat-hf']
    models = ['Llama-2-13b-chat-hf']
    cm = CostManager()
    card_format = 'dict'

    hps = []
    for model in models:

        if card_format == 'str':
            exp = f'generative_{card_format}_' + topic
        else:
            exp = f'generative_' + topic
        hp = {
            # general
            'experiment'      : exp,
            'name'            : f'{model}_main',
            'method'          : 'generative',
            'load_from'       : None,
            # dataset
            'dataset_folder'  : 'datasets/anthropic-eval',
            'topic'           : topic,
            'shuffle'         : False,
            'seed'            : 311,
            'batch_nums'      : [8] * 5 + [60],
            'model'           : model,
            # training
            'epoch'           : None,  # None: use the number of training batches as epoch
            'use_refine'      : False,
            'initial_criteria': get_initial_criteria(topic),
            'CoT'             : False,
            # eval
            'async_eval'      : True,
            'card_format'     : card_format,
        }
        hps.append(hp)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(GenerativeMethod(hp, cm).main) for hp in hps]
        futures = wait(futures)
        for future in futures.done:
            print(future.result())


def few_shot_method():
    topic = 'high_school_physics'
    exp = 'few_shot_' + topic
    # models = ['gpt-3.5-turbo-1106', 'Mixtral-8x7B-Instruct-v0.1', 'Mistral-7B-Instruct-v0.2', 'Llama-2-70b-chat-hf']
    # models = ['Mixtral-8x7B-Instruct-v0.1', 'Mistral-7B-Instruct-v0.2', 'Llama-2-70b-chat-hf']
    models = ['Llama-2-13b-chat-hf']
    token_manager = CostManager()

    hps = []
    for model in models:
        hp = {
            'method': 'generative',
            'dataset_folder': '../datasets/mmlu',
            'topic': topic,
            'model': model,
            'batch_nums': [10] * 5 + [50, 0],
            'shuffle': True,
            'seed': 311,
        }
        hps.append(hp)

    for hp in hps:
        FewShotMethod(exp, f"{hp['model']}_main", hp, token_manager).main()


def few_shot_refill(method_folder: str):
    if method_folder.startswith('generative/'):
        method_folder = method_folder[len('generative/'):]
    load_few_shot_instance(method_folder).main()


def generative_method_refill(method_folder: str):
    load_method_instance(method_folder).main()


def eval_arbitrary_card():
    path = 'generative/outputs/generative_machine_learning/01-24_22-55-04_Mistral-7B-Instruct-v0.2_human_readable_card'
    if path.startswith('generative/'):
        path = path[len('generative/'):]
    method = load_method_instance(path)
    card = GenerativeCard(filename=f'{path}/cards/epoch_2_card.json')
    method.card = card
    method.predictive_eval('eval_validation_', method.validation_batch, num_times=5)


def generative_ablation():
    exp_folder = 'generative/outputs/generative_high_school_physics_ic2_no_overview'
    if exp_folder.startswith('generative/'):
        exp_folder = exp_folder[len('generative/'):]
    # list all folder under exp_folder
    method_folders = [os.path.join(exp_folder, f) for f in os.listdir(exp_folder)]
    # filter out non-folder
    method_folders = [f for f in method_folders if os.path.isdir(f)]
    for method_folder in method_folders:
        # load method instance
        method = load_method_instance(method_folder, new_instance=True, copy_info=True)
        method.main()

def preferential_eval(params: dict, oracle_1=False, oracle_2=False):

    topic = params['topic']
    model = params['student_model']

    evaluator_1 = params['a']['evaluator']
    exp_name_1 = params['a']['exp_name']
    iter_1 = params['a']['iter']
    format_1 = params['a']['format']
    epoch_1 = params['a']['epoch']
    
    evaluator_2 = params['b']['evaluator']
    exp_name_2 = params['b']['exp_name']
    iter_2 = params['b']['iter']
    format_2 = params['b']['format']
    epoch_2 = params['b']['epoch']


    # Load dataset for validation
    validation = load_mmlu_batches(f'datasets/mmlu/', topic, model=model, partition='test', batch_nums=[60])[0]


    # Load cards
    # if exp_name is 'latest', get the latest experiment
    if exp_name_1 == 'latest':
        all_folders = os.listdir(f'outputs/generative/{topic}/{iter_1}/{format_1}/{evaluator_1}/{model}/')
        all_folders.sort()
        for folder in all_folders:
            if re.match(r'\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_' + model, folder):
                folder_path = folder
        exp_name_1 = folder_path

    card_1_path = f'outputs/generative/{topic}/{iter_1}/{format_1}/{evaluator_1}/{model}/{exp_name_1}/cards/epoch_{epoch_1}_card.json'

    if oracle_1:
        model_accuracy = validation.get_accuracy()
        # converte to percentage
        model_accuracy = np.round(model_accuracy * 100, 2)
        card1 = f'This student answers the questions correctly {model_accuracy}% of the time.'
        evaluator_1 = 'oracle'
        exp_name_1 = 'oracle'
        iter_1 = 'oracle'
        format_1 = 'oracle'
        epoch_1 = 0

        print(card1)
    else: 
        if format_1 == 'dict':
            card1 = GenerativeCard(filename=card_1_path)
        else:
            # just load json file 
            with open(card_1_path, 'r') as f:
                card1 = json.load(f)


    if exp_name_2 == 'latest':
        all_folders = os.listdir(f'outputs/generative/{topic}/{iter_2}/{format_2}/{evaluator_2}/{model}/')
        all_folders.sort()
        for folder in all_folders:
            if re.match(r'\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_' + model, folder):
                folder_path = folder
        exp_name_2 = folder_path

   
    card_2_path = f'outputs/generative/{topic}/{iter_2}/{format_2}/{evaluator_2}/{model}/{exp_name_2}/cards/epoch_{epoch_2}_card.json'

    if oracle_2:
        model_accuracy = validation.get_accuracy()
        print(model_accuracy)
        model_accuracy = np.round(model_accuracy * 100, 2)
        card2 = f'This student answers the questions correctly {model_accuracy}% of the time.'
        evaluator_2 = 'oracle'
        exp_name_2 = 'oracle'
        iter_2 = 'oracle'
        format_2 = 'oracle'
        epoch_2 = 0

        print(card2)

    else:
        if format_2 == 'dict':
            card2 = GenerativeCard(filename=card_2_path)
        else:
            # just load json file 
            with open(card_2_path, 'r') as f:
                card2 = json.load(f)


    rm = ResourceManager('generative_college_chemistry', 'eval_contrastive_gpt-3.5-turbo-1106_Mistral-7B-Instruct-v0.2_0')

    evaluator = PreferentialEvaluator('mmlu',
                                      topic,
                                      model, 
                                      rm,
                                      evaluator_name='NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO')
    
    player_1 = {
        'evaluator': evaluator_1,
        'card_format': format_1,
        'iterative_method': iter_1,
        'epoch': epoch_1
    }

    player_2 = {
        'evaluator': evaluator_2,
        'card_format': format_2,
        'iterative_method': iter_2,
        'epoch': epoch_2
    }

    evaluator.main(name='eval_preferential',
                    batch=validation, 
                    card1=card1, 
                    card2=card2,
                    player_a=player_1,
                    player_b=player_2)

    rm.shutdown()


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, help='Topic to run', default='mmlu')
    parser.add_argument('--topic', type=str, help='Topic to run', default='high_school_mathematics')
    parser.add_argument('--model', type=str, help='Models to run', default='Mistral-7B-Instruct-v0.2')
    parser.add_argument('--model2', type=str, help='Models to run', default='Mixtral-8x7B-Instruct-v0.1')
    # parser.add_argument('--model2', type=str, help='Models to run', default='Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument('--epoch', type=str, help='No. epochs to run', default='5')
    parser.add_argument('--evaluator', type=str, help='Evaluator to run', default='gpt')
    parser.add_argument('--format', type=str, help='Format of the card', default='dict')

    args = parser.parse_args()

    model_pair = [# ['Mistral-7B-Instruct-v0.2', 'Mixtral-8x7B-Instruct-v0.1'],  # OK vs. Good
                #   ['Mistral-7B-Instruct-v0.2', 'Llama-2-13b-chat-hf'], # OK vs. bad
                #   ['gpt-3.5-turbo-1106', 'Mixtral-8x7B-Instruct-v0.1'], # Good vs. Good
                #   ['Meta-Llama-3-70B-Instruct', 'Llama-2-13b-chat-hf'], # Supergood vs. bad
                  ['Meta-Llama-3-70B-Instruct', 'Mixtral-8x7B-Instruct-v0.1'], # Supergood vs. good
                    ]
    
    # for pair in tqdm(model_pair, desc='Running model pairs'):
        # pair = [pair]
        # compute_baseline_accuracy(args.topic, pair[0][0], pair[0][1])
        # contrastive_method(args.topic, evaluator=args.evaluator, student_models=pair, epoch=int(args.epoch), card_format=args.format)
        # time.sleep(60)

    generative_method(args.meta,
                      args.topic, evaluator=args.evaluator, student_models=[args.model], epoch=int(args.epoch), card_format=args.format)
    # generative_method_anthropic()
    # contrastive_eval()
    # few_shot_method()
    # eval_arbitrary_card() 
    # generative_ablation()


    supported_models = [
        'Mistral-7B-Instruct-v0.2',
        'Mixtral-8x7B-Instruct-v0.1',
        'Llama-2-13b-chat-hf',
        'Llama-2-70b-chat-hf',
        'gpt-3.5-turbo-1106',
    ]

    for m in tqdm(supported_models):
       
        pref_params = {
        'topic': 'high_school_mathematics',
        'student_model': m,
        'a': {
            'evaluator': 'claude', 
            'exp_name': 'latest', 
            'iter': 'prog-reg',
            'format': 'dict', 
            'epoch': 4
            },
        
        'b': {
            'evaluator': 'gpt', 
            'exp_name': 'latest', 
            'iter': 'prog-reg',
            'format': 'dict', 
            'epoch': 4
            }
        }
        # preferential_eval(pref_params,
                        # oracle_1=False,
                        # oracle_2=False)
        
        # time.sleep(20)

