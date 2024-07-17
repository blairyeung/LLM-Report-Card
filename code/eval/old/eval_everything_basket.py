import os
import sys
import re
import pandas as pd

sys.path.append('..')
sys.path.append('.')
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import numpy as np
import concurrent

from eval.eval_fully_contrastive import *
from sklearn.metrics import confusion_matrix
import numpy as np
from card_generation.few_shot import *
from card_generation.generative_method import *
import matplotlib.pyplot as plt
from util.card import GenerativeCard
from util.data import *

# html display
from IPython.display import display, HTML

no_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'} #  0-3: A-D, ALL OTHERS: N

        
def count_words(txt):
    return len(re.findall(r'\w+', txt))

def format_single_entry(entry, choice_incl=False, ground_truth_incl=False):
    q, choice, gt, pred, completion = entry

    choice_str = '\n'.join([f'{no_to_letter[i]}: {c}' for i, c in enumerate(choice)])
    gt_str = no_to_letter[gt] if gt in no_to_letter else 'N'

    entry_str = q

    if choice_incl:
        entry_str += f'\nChoices:\n{choice_str}\n'
    if ground_truth_incl and choice_incl:
        entry_str += f'Ground Truth: {gt_str}\n'

    if not choice_incl and not ground_truth_incl:
        pass

    return entry_str

def format_few_shot_entry(entry):
    q, choice, gt, pred, completion = entry

    choice_str = '\n'.join([f'{no_to_letter[i]}: {c}' for i, c in enumerate(choice)])
    gt_str = no_to_letter[gt] if gt in no_to_letter else 'N'
    student_correctness = gt == pred

    return f'{q}\nChoices:\n{choice_str}\nStudent Completion: {completion}\nGround Truth Answer: {gt_str}\nStudents correctness: {student_correctness}'


def format_full_eval_str(method, choice_incl=False, ground_truth_incl=False, shuffle_seed=311):
    str_all = ''
    cnt = 1
    shuffled_batch = [method.testing_batch[i] for i in range(len(method.testing_batch))]
    if shuffle_seed != -1:
        random.seed(shuffle_seed)
        shuffled_order = list(range(len(shuffled_batch)))
        random.shuffle(shuffled_order)
        temp = [shuffled_batch[i] for i in shuffled_order]
        shuffled_batch = temp

    for entry in shuffled_batch:
        entry_formatted = format_single_entry(entry,
                                              choice_incl=choice_incl, 
                                              ground_truth_incl=ground_truth_incl)
        str_all += f'Question {cnt}. {entry_formatted}\n\n'
        cnt += 1
    
    return str_all, shuffled_order


def format_few_shot_string(method, max_sampling=-1, balance=False):
    str_all = ''
    cnt = 1
    positive_cnt, negative_cnt = 0, 0
    # if blance, half true half false
    for i in range(len(method.training_batches)):
        for entry in method.training_batches[i]:
            if balance:
                if entry[2] == entry[3]:
                    if positive_cnt >= negative_cnt:
                        continue
                    positive_cnt += 1
                else:
                    if negative_cnt >= positive_cnt:
                        continue
                    negative_cnt += 1
            entry_formatted = (format_few_shot_entry(entry))
            str_all += f'Few shot sample question {cnt}. {entry_formatted}\n\n'
            cnt += 1

            if max_sampling != -1 and cnt >= max_sampling:
                break
    
    return str_all


def get_training_accuracy(method):
    correct_cnt = 0
    tot_cnt = 0
    for i in range(len(method.training_batches)):
        for entry in method.training_batches[i]:
            rslt = entry
            if len(rslt) == 5:
                q, choices, gt, pred, completion = rslt
            else:
                q, gt, pred, completion = rslt
            if gt == pred:
                correct_cnt += 1
            tot_cnt += 1
    
    # 2 Decimal, percentage
    decimal = round(correct_cnt / tot_cnt, 2)
    percentage = int(decimal * 100)
    return str(percentage) + '%', correct_cnt / tot_cnt


def get_latest_folder(meta, topic, optim_method, card_format, evaluator, model, generation_method='generative'):
    folder_root = f'outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}'
    all_folders =  os.listdir(f'outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}')
    all_folders.sort()
    all_folders = all_folders[::-1]
    for folder in all_folders:
        if re.match(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_" , folder):
            return folder
    return None


def get_card_str(card_obj, format_type=1):
    if format_type == 1:
        return str(card_obj)
    else:
        card_str = ''
        for key, value in card_obj.criteria:
            stuff_str = f"{value['overview']} {value['thinking_pattern']} {value['strength']} {value['weakness']}"
            card_str += f'{key}: {stuff_str}\n'
        return card_str

def get_predictions(testing_len, guesser, user_prompt, m):
    for _ in tqdm([0]):
        rslt = guesser(user_prompt,
                   use_json=False)

    answers = rslt.split('\n')

    cnt = 0
    for i in range(len(answers)):
        if f'{testing_len}' in answers[i]:
            cnt = i + 1
             
    # start from 1, til 60
    answers = answers[cnt-testing_len:cnt]
    pred_list = []

    for i in range(testing_len):
        if isinstance(answers[i], bool):
            pred_list.append(answers[i])
        else:
            # print(answers[i].lower())
            pred_list.append('f' not in answers[i].lower())

    return pred_list


def get_ensembled_query_predictions(testing_len, guessers, user_prompts, ensemble_info, m):
    # Execute the guesser function concurrently for each user prompt
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(guessers[i], user_prompts[i], temperature=0, use_json=False) for i in range(len(guessers))]
        answers = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]

    # Extract JSON parts from answers using regular expressions
    json_answers = []
    for a in answers:
        json_match = re.search(r'\{.*\}', a, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                json_data = json.loads(json_str)
                json_answers.append(json_data)
            except json.JSONDecodeError:
                raise ValueError(f'Failed to decode JSON from string: {json_str}')
            
            print(json_data)

    # Initialize the lists to hold combined predictions
    correct_predictions = np.zeros(testing_len, dtype=int)
    incorrect_predictions = np.zeros(testing_len, dtype=int)

    # Aggregate predictions from each guesser
        # Aggregate predictions from each guesser
    for idx, guesser_ans in enumerate(json_answers):
        correct_indices = guesser_ans.get('Correct', [])
        incorrect_indices = guesser_ans.get('Incorrect', [])
        
        # Recover the original order using the shuffled order information
        shuffled_order = ensemble_info['order'][idx]
        reordered_correct_indices = [shuffled_order.index(i - 1) for i in correct_indices]
        reordered_incorrect_indices = [shuffled_order.index(i - 1) for i in incorrect_indices]

        for idx in reordered_correct_indices:
            correct_predictions[idx] += 1
        for idx in reordered_incorrect_indices:
            incorrect_predictions[idx] += 1

    # Combine predictions using majority voting
    final_predictions = []
    for i in range(testing_len):
        if correct_predictions[i] > incorrect_predictions[i]:
            final_predictions.append(True)
        else:
            final_predictions.append(False)
    
    return final_predictions

# Example usage:
# result = get_ensembled_query_predictions(testing_len, guessers, user_prompts, ensemble_info, m)
# print(result)




def get_ensembled_predictions(testing_len, guessers, user_prompt, m):
    for _ in tqdm([0]):
        # rslt = [g(user_prompt, use_json=False) for g in guessers]
        # concurrently do these
        rslt = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            r = {executor.submit(g, user_prompt, use_json=False): g for g in guessers}
            for future in concurrent.futures.as_completed(r):
                rslt.append(future.result())

    # print(rslt)

    answers = [r.split('\n') for r in rslt]

    # print(answers)

    ans = []
    for a in answers:
        cnt = 0
        for i in range(len(a)):
            if f'{testing_len}' in a[i]:
                cnt = i + 1
                
        a = a[cnt-testing_len:cnt]
        ans.append(a)

    # print(ans)
    answers = ans

    pred_list = []

    pred_each_guesser = []
    for j in range(len(guessers)):
        guesser_sublist = []
        guesser_ans = answers[j]
        for i in range(testing_len):
            if isinstance(guesser_ans[i], bool):
                guesser_sublist.append(guesser_ans[i])
            else:
                guesser_sublist.append('f' not in guesser_ans[i].lower())

        pred_each_guesser.append(guesser_sublist)

    pred_each_guesser = np.array(pred_each_guesser)
    pred_list = np.sum(pred_each_guesser, axis=0) > len(guessers) / 2

    return pred_list


def pred_eval(eval_params):

    meta = eval_params.get('meta', 'mmlu')
    card_format = eval_params.get('card_format', 'dict') # dict, bullet_point, string
    epoch = eval_params.get('epoch', 4) # epoch number
    cot = eval_params.get('cot', False)  # whether to use CoT
    baseline = eval_params.get('baseline', 'card') # card, few_shot
    evaluator = eval_params.get('evaluator', 'gpt') # guessesr, claude, gpt
    model = eval_params.get('model', 'Mixtral-8x7B-Instruct-v0.1') # student models
    topic = eval_params.get('topic', 'high_school_chemistry') # MMLU topics
    source_topic = eval_params.get('source_topic', 'high_school_chemistry') # MMLU topics
    optim_method = eval_params.get('optim_method', 'prog-reg') # prog-reg, one-pass
    folder = eval_params.get('folder', 'latest') # folder name, whether scan for latest
    prior_info = eval_params.get('prior_info', False) # whether to tell LLM past performance
    eval_method = eval_params.get('eval_method', 'single') # single, all
    guesser_name = eval_params.get('guesser', 'gpt') # gpt, claude
    generation_method = eval_params.get('generation_method', 'generative') # generative, contrastive
    eval_type = eval_params.get('eval_type', 'predictive') # predictive, generative
    ensemble_shots = eval_params.get('ensemble_shots', 5) # number of ensembled shots
    rebalance_testset = eval_params.get('rebalance_testset', -1) # whether to rebalance the testset

    if generation_method == 'contrastive':
        target_model = eval_params.get('target_model', "Mixtral-8x7B-Instruct-v0.1") # gpt, claude
    
    else:
        target_model = model

    if optim_method == 'one-pass':
        assert epoch == 0

    initial_criteria = []
    if card_format == 'dict':
        initial_criteria = get_initial_criteria(topic)

    # part 1 get the folder

    folder_root = f'outputs/{generation_method}/{source_topic}/{optim_method}/{card_format}/{evaluator}/{model}'
    
    if folder == 'latest':
        folder_path = get_latest_folder(meta, source_topic, optim_method, card_format, evaluator, model, generation_method)
    else:
        folder_path = folder


    assert model in folder_path

    cm = CostManager()
    exp = 'web_eval' + topic

    hp = {
                # general
                'experiment'      : exp,
                'name'            : f'{model}_main',
                'method'          : 'generative',
                'load_from'       : None,
                # dataset
                'dataset_folder'  : f'datasets/{meta}',
                'topic'           : topic,
                'shuffle'         : False,
                'seed'            : 311,
                'batch_nums'      : [8] * 5 + [60],
                'model'           : target_model,
                # training
                'epoch'           : None,  # None: use the number of training batches as epoch
                'use_refine'      : False,
                'initial_criteria': initial_criteria,
                'CoT'             : False,
                'card_format'     : card_format,
                'evaluator'       : 'gpt',
                # eval
                'async_eval'      : True,
            }
    
    m =  GenerativeMethod(hp, cm)

    if eval_type == 'train':
        training_batches = m.training_batches
        # add everything up
        train_all = []
        for i in range(len(training_batches)):
            train_all += training_batches[i]

        m.testing_batch = train_all

    if rebalance_testset != -1:
        correct_batch = m.testing_batch.get_correct_subbatch()
        incorrect_batch = m.testing_batch.get_incorrect_subbatch()

        correct_portion = rebalance_testset
        incorrect_portion = 1 - rebalance_testset

        total_count = len(correct_batch) + len(incorrect_batch)
        correct_count = int(total_count * correct_portion)
        incorrect_count = int(total_count * incorrect_portion)

        correct_count = min(correct_count, len(correct_batch))
        incorrect_count = min(incorrect_count, len(incorrect_batch))

        correct_samples = random.sample(correct_batch.raw, correct_count)
        incorrect_samples = random.sample(incorrect_batch.raw, incorrect_count)
        mixed_batch = correct_samples + incorrect_samples

        random.shuffle(mixed_batch)

        m.testing_batch = MMLUBatch(mixed_batch, model=target_model)
        print('Rebalanced testset!')
        print(f'New total count: {len(m.testing_batch)}, new correct count: {len(m.testing_batch.get_correct_subbatch())}')



        

    # load card
    card_path = folder_root + f'/{folder_path}/cards/epoch_{epoch}_card.json'

    with open(card_path) as f:
        card_dict = json.load(f)
    
    # load card
    if card_format == 'dict':
        if generation_method == 'contrastive':
            card_dict = card_dict[target_model]

        card = GenerativeCard(d=card_dict)

    elif card_format == 'bullet_point':
        card = '\n -'.join([card_dict['card'][str(i)] for i in range(len(card_dict['card']))])

    else:
        card = card_dict['card']

    if baseline == 'few_shot':
        few_shot_str = format_few_shot_string(m)
        card = few_shot_str

    card = get_card_str(card, 1)

    if prior_info:
        card += f"\n# LLM Student's Past Performance:\nThe student's accuracy on previous questions under this topic is {get_training_accuracy(m)[0]}.\n"
        card += f"This is the prior prbability of the student answering each of these question correctly. Use the evaluation to infer fine-grained predictions for each individual questions that surpasses this prior probability."
        major_correct = get_training_accuracy(m)[1] > 0.5 
        # card += f"Given that the student's previous questions are mostly {True if major_correct else False}. You should only flip the prediction to the opposite if you are confident that the student will answer {'incorrectly' if major_correct else 'correctly'}.\n"
        print('Training accuracy included! Training acc is', get_training_accuracy(m)) 

    if card_format == 'dict':
        print(f'Criteria cnt: {len(GenerativeCard(d=card_dict))}\nWord count {GenerativeCard(d=card_dict).words()}.')

    if eval_method == 'all':
        if guesser_name == 'gpt':
            evaluator_name = GPT_4_MODEL_NAME
        elif guesser_name == 'claude':
            evaluator_name = CLAUDE_3_MODEL_NAME
        elif guesser_name == 'mixtral':
            evaluator_name =  'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
        elif guesser_name == 'llama':
            evaluator_name = 'meta-llama/Meta-Llama-3-70B-Instruct'

        else:
            raise ValueError('Evaluator not recognized.')
        rm = ResourceManager()

        pred_eval = PredictiveEvaluator(topic=topic, model=model, 
                                        rm=rm, type_='mmlu', evaluator_name=evaluator_name)

        card_dict['Student performance summary'] = f'The LLM student models past accuracy in this topic is {get_training_accuracy(m)[0]}.'
        info_dict = pred_eval.main(name=f'eval_test_epoch_{epoch}_predictive', 
                                   batch=m.testing_batch,
                                   card=GenerativeCard(d=card_dict),
                                   num_times=1)
        print(info_dict.keys())

        pred_accuracy = info_dict['metrics']['accuracies'][0]
        ground_truths = [entry[2] == entry[3] for entry in m.testing_batch]
        oracle = sum(ground_truths) / len(ground_truths)

        print(f'Predictive accuracy using model {evaluator_name}:', pred_accuracy)
        print(f'Oracle accuracy:', max(oracle, 1 - oracle))

        
        ground_truths = [entry[2] == entry[3] for entry in m.testing_batch]
        
        oracle = sum(ground_truths) / len(ground_truths)

        oracle_acc = max(oracle, 1 - oracle)


        df_path = f'ExpRslts/Predictive_{meta}_All.csv'
        # try to read, if dne, create a new one
        try:
            df = pd.read_csv(df_path)
        except:
            df = pd.DataFrame(columns=['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Predictive_Accuracy', 'Oracle_Accuracy', 'Word_Count', 'Guesser'])

        # use concat    

        entry = [model, topic, card_format, epoch, pred_accuracy, oracle_acc, count_words(str(card)), evaluator_name]
        columns = ['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Predictive_Accuracy', 'Oracle_Accuracy', 'Word_Count', 'Guesser']


        df = pd.concat([df, pd.DataFrame([entry], columns=columns)], ignore_index=True)

        # when saving, no indexing column

        df.to_csv(df_path, index=False)

        return
        
    else:
        pass

    testing_len = len(m.testing_batch)

    def get_user_prompt(card, ensembled_shots=1):

        sys_prompt = """
        You are an expert in {topic}. Given the evaluation of a student and a few questions, your task is to individually determine if the student can correctly answer questions.
        """

        # with open('prompts/eval/predictive/user.txt', 'r') as f:
        #     user_prompt = f.read() 

        user_prompt = """
        Predict if the Large Language Model (LLM) student will answer The Question correctly based on the Student Evaluation. You will do this by reasoning step-by-step:

        ## Your Task:

        1. Analyze how the student may answer the questions based on the criteria.
        2. Based on your analysis, predict if the student will answer The Question correctly (true) or not (false).
        3. Be critical and objective. Don't make any assumptions about the student. Your prediction should be solely grounded on the Student Evaluation below.
        4. Your prediction for a question shuold be independent of the other questions.
        
        ## The Questions

        {qa}

        ## Student Evaluation

        {card}
        """

        if cot:
            user_prompt += f"""
            The model may either answer correctly or incorrectly. You should predict if the student will answer the question correctly or not.
            Give one-sentence analysis one by one, one for each question, using the format:
            Question index: concise one-sentence reasoning
            <1-{testing_len}>: <concise one-sentence reasoning>
            """

        confidence = False
        if confidence:
            user_prompt += f"""
        The model may either answer correctly or incorrectly. You should predict if the student will answer the question correctly or not.
        Make predictions one by one, using the format, here no reasoning required:
        Confidence is a number between 1 and 5, where 1 is the lowest and 5 is the highest.
        Verdict is either T or F.
        Question index: Confidence, Verdict (no reasoning needed)
        <1-{testing_len}>: <1-5>, <T/F>
        """

        else:
            formatting_str = """
            You should write an aggregated analysis, and then predict the correctness case by case.
            Your task is not to establish expectations for the LLM student but to predict the student's performance based on the evaluation. 
            So you should be smart and objective. The student can answer a question correct or wrong.

            [[Formatting]] Follow the formatting, make predictions one by one, using the following format, no reasoning required. You should use json format for your predictions.
            ## Aggregated analysis: <student's capability, strength, weakness, and the difficulty of the questions>
            ...
        
            ## Predictions: use valid json
            {{
            "Correct": [<a python list of question indices that you believe the student will answer correctly>]
            "Incorrect": [<a python list of question indices that you believe the student will answer incorrectly, must be the complement of the Correct list>]
            }}
            ...
            """

            user_prompt += formatting_str


        if baseline == 'few_shot':
            user_prompt += f"Note that questions under student evaluation are examples for you to check, the {testing_len} questions are what you are going to predict!"


        if ensembled_shots == 1:
            str_all, _ = format_full_eval_str(m, True, True, shuffle_seed=42)

            # add a few-shot example
            # card += f'\n\nFew-shot examples of the students past performance:\n{format_few_shot_string(m, max_sampling=6, balance=True)}'

            user_prompt = user_prompt.format(qa=str_all, card=str(card))

            sys_prompt = sys_prompt.format(topic=topic)

            formatted_str = f"""{sys_prompt.format(topic=topic)}\n\n{user_prompt}"""

            return sys_prompt, user_prompt, str_all
        
        else:
            #
            str_enesmbled, order_ensembled = [], []
            print(ensembled_shots)
            for j in range(ensembled_shots):
                str_all, shuffled_order = format_full_eval_str(m, True, True, shuffle_seed=j)
                str_enesmbled.append(str_all)
                order_ensembled.append(shuffled_order)
            
            user_prompts = []
            for j in range(ensembled_shots):
                # print(type(str_enesmbled[j]))
                qa_str = str_enesmbled[j]
                user_prompt_formatted = user_prompt.format(qa=qa_str, card=str(card))
                user_prompts.append(user_prompt_formatted)

            sys_prompt = sys_prompt.format(topic=topic)
            
            return sys_prompt, user_prompts, {'str_all': str_enesmbled, 'order': order_ensembled}
    

    sys_prompt, user_prompts, ensemble_info = get_user_prompt(card, ensembled_shots=ensemble_shots)
    # sys_prompt, user_prompt, _ = get_user_prompt(card, ensembled_shots=1)

    if guesser_name == 'ensemble':
        guessers = [GPTModel(system_prompt=sys_prompt,
                    model_name=GPT_4_MODEL_NAME),
                    ClaudeModel(system_prompt=sys_prompt,
                    model_name=CLAUDE_3_OPUS),
                    GeminiModel(system_prompt=sys_prompt,
                                model_name=GEMINI_MODEL_NAME),
                    Llama3Model(system_prompt=sys_prompt,
                                model_name='meta-llama/Meta-Llama-3-70B-Instruct'),
                    ClaudeModel(system_prompt=sys_prompt,
                                model_name=CLAUDE_3_SONNET),]


    if guesser_name == 'gpt':        
        guesser = GPTModel(system_prompt=sys_prompt,
                 model_name=GPT_4_MODEL_NAME)
        
    elif guesser_name == 'claude':
        guesser = ClaudeModel(system_prompt=sys_prompt,
                    model_name=CLAUDE_3_OPUS)
        
    elif guesser_name == 'llama':
        guesser = Llama3Model(system_prompt=sys_prompt,
                    model_name='meta-llama/Meta-Llama-3-70B-Instruct')
        
    elif guesser_name == 'gemini':
        guesser = GeminiModel(system_prompt=sys_prompt,
                    model_name=GEMINI_MODEL_NAME)

    if guesser_name == 'ensemble':
        pred_list = get_ensembled_predictions(testing_len, guessers, user_prompts, ensemble_info, m)


    else:
        gusser_name_map = {'gpt': GPT_4_MODEL_NAME,
                           'claude': CLAUDE_3_OPUS,
                           'gemini': GEMINI_MODEL_NAME,
                           'llama': 'meta-llama/Meta-Llama-3-70B-Instruct'}
    
        guessers = [select_model(gusser_name_map[guesser_name], sys_prompt, cm) for _ in range(ensemble_shots)]

    pred_list = get_ensembled_query_predictions(testing_len, guessers, user_prompts, ensemble_info, m)

    print('-' * 50)
    print(pred_list)
    print('-' * 50)

    ground_truths = [entry[2] == entry[3] for entry in m.testing_batch]

    print(f'Model accuracy is {sum(ground_truths) / len(ground_truths)}')

    pred_accuracy = sum([pred_list[i] == ground_truths[i] for i in range(testing_len)]) / testing_len

    oracle = sum(ground_truths) / len(ground_truths)

    oracle_acc = max(oracle, 1 - oracle)

    print(f'Predictive accuracy using model {guesser_name}:', pred_accuracy)

    # SAVE
    df_path = f'ExpRslts/Predictive_{meta}_Single.csv'
    df_history_path  = f'ExpRslts/Predictive_{meta}_History_Single.csv'
    # try to read, if dne, create a new one
    try:
        df = pd.read_csv(df_path)
        df_history = pd.read_csv(df_history_path)
    except:
        df = pd.DataFrame(columns=['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Predictive_Accuracy', 'Oracle_Accuracy', 'Word_Count', 'Guesser'])
        df_history = pd.DataFrame(columns=['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Oracle_Accuracy', 'Word_Count', 'Guesser', 'Conversation_History'])
    # use concat    

    entry = [model, topic, card_format, epoch, pred_accuracy, oracle_acc, count_words(str(card)), guesser_name]
    columns = ['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Predictive_Accuracy', 'Oracle_Accuracy', 'Word_Count', 'Guesser']

    
    df_history.drop(df_history.columns[0], axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame([entry], columns=columns)], ignore_index=True)

    df.to_csv(df_path, index=False)

    # append to another df
    df_history_path = f'ExpRslts/Predictive_{meta}_History_Single.csv'
    # try to read, if dne, create a new one
    try:
        df_history = pd.read_csv(df_history_path)
    except:
        df_history = pd.DataFrame(columns=['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Oracle_Accuracy', 'Word_Count', 'Guesser', 'Conversation_History'])
    
    # use concat
    entry = [model, topic, card_format, epoch, oracle_acc, count_words(str(card)), guesser_name, None]

    columns = ['Student_Model', 'Topic', 'Card_Format', 'Epoch', 'Oracle_Accuracy', 'Word_Count', 'Guesser', 'Conversation_History']
    df_history = pd.concat([df_history, pd.DataFrame([entry], columns=columns)], ignore_index=True)

    # save to csv
    # drop indexing column
    df_history.drop(df_history.columns[0], axis=1, inplace=True)
    df_history.to_csv(df_history_path)

    # display the confusion matrix

    # cm = confusion_matrix(ground_truths, pred_list)
    # plt.matshow(cm)
    # plt.colorbar()
    # plt.xlabel('Predicted')
    # plt.ylabel('True')

    # # blue cmap
    # plt.set_cmap('Blues')

    # # text label
    # for i in range(2):
    #     for j in range(2):
    #         plt.text(j, i, cm[i, j], ha='center', va='center', color='blue')
    # plt.show()

    return
