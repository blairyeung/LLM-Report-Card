import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
import json
from tqdm import tqdm
import os
# from eval_everything import 
import jsonlines
import re
import time
import pandas as pd
import random

from core.card import GenerativeCard
from core.data import Batch
from core.models import CostManager, select_model
from core.utils import ResourceManager
from peripherals.oracle import compute_baseline_accuracy



def get_ground_truth_str(meta, batch, index):
    if meta == 'mmlu':
        choices = batch.get_choices(index)
        gt = batch.get_true_answer(index)
        # convert 0 to A, 1 to B, etc.
        map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

        if gt in range(len(choices)):
            return f'{map[gt]} .{choices[gt]}'
        
        return 'Incorrect, the student chose None of the above'
    
    elif meta == 'anthropic-eval':
        return batch.get_true_answer(index)
       

def get_qc_str(meta, batch1, batch2, indices, student_names=['A', 'B']):
    s = ''
    counter = 1
    for i in indices:
        student_a_correctness = 'correctly' if batch1.get_true_answer(i) == batch1.get_model_answer(i) else 'incorrectly'
        student_b_correctness = 'correctly' if batch2.get_true_answer(i) == batch2.get_model_answer(i) else 'incorrectly'

        s += f'## Question {counter}. {batch1.get_question(i)}\n'
        s += f'Correct Answer: {get_ground_truth_str(meta, batch1, i)}\n'
        s += f'### Student {student_names[0]} Answer:'
        s += f'{batch1.get_model_reasoning(i)}\nStudent {student_names[0]} answered the question {student_a_correctness}.\n\n'
        s += f'### Student {student_names[1]} Answer:'
        s += f'{batch2.get_model_reasoning(i)}\nStudent {student_names[1]} answered the question {student_b_correctness}.\n\n'

        counter += 1

    return s

def get_qc_str_aggregated(meta, batch1, batch2, indices, student_names=['A', 'B']):
    s = ''
    counter = 1
    for i in indices:
        s += f'## Question {counter}. {batch1.get_question(i)}\n'
        # s += f'Correct Answer: {get_ground_truth_str(meta, batch1, i)}\n'
        counter += 1

    s += f'## Answer {student_names[0]}:\n'
    counter = 1
    for i in indices:
        student_a_correctness = 'Correct' if batch1.get_true_answer(i) == batch1.get_model_answer(i) else 'Incorrect'
        s += f'### Question {counter}:\n'
        s += f'{batch1.get_model_reasoning(i)}\n\n'
        counter += 1
        s += f'{batch1.get_model_reasoning(i)}\nAnswer A Correctness: {student_a_correctness}\n\n'

    s += f'## Answer {student_names[1]}:\n'
    counter = 1
    for i in indices:
        student_b_correctness = 'Correct' if batch2.get_true_answer(i) == batch2.get_model_answer(i) else 'Incorrect'
        s += f'### Question {counter}:\n'
        s += f'{batch2.get_model_reasoning(i)}\n\n'
        counter += 1
        s += f'{batch2.get_model_reasoning(i)}\nAnswer B Correctness: {student_b_correctness}\n\n'

        counter += 1

    return s

def get_qc_str_one_batch(meta, batch, indices, student_names=['1', '2']):
    s = ''
    counter = 1
    for i in indices:
        s += f'## Question {counter}. {batch.get_question(i)}\n'
        s += f'Correct Answer: {get_ground_truth_str(meta, batch, i)}\n'
        s += f'### Student Completion\n'
        s += f'{batch.get_model_reasoning(i)}\n\n'
        counter += 1

    return s

def get_latest_folder(topic, optim_method, card_format, evaluator, model, generation_method='generative'):
    folder_root = f'outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}'
    all_folders =  os.listdir(f'outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}')
    all_folders.sort()
    all_folders = all_folders[::-1]
    for folder in all_folders:
        if re.match(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_" , folder):
            return f'{folder_root}/{folder}' 
    return None

def load_cards(topic,
               card_format,
               models,
               evaluator,
               method='contrastive'):
    
    if method == 'contrastive':
        try:
            model_str = f"['{models[0]}', '{models[1]}']"
            path = get_latest_folder(topic, 'prog-reg', card_format, evaluator, model_str, 'contrastive')
            with open(path+'/cards/epoch_4_card.json', 'r') as f:
                cards = json.load(f)
            card_1_dict = cards[model1]
            card_2_dict = cards[model2]

            # card_1 = GenerativeCard(d=card_1_dict)
            # card_2 = GenerativeCard(d=card_2_dict)
            
            card_1 = card_1_dict
            card_2 = card_2_dict

            return {model1: str(card_1), model2: str(card_2)}
        except:
            try: 
                model_str = f"['{models[1]}', '{models[0]}']"
                path = get_latest_folder(topic, 'prog-reg', card_format, evaluator, model_str, 'contrastive')
                with open(path+'/cards/epoch_4_card.json', 'r') as f:
                    cards = json.load(f)
                card_1_dict = cards[model1]
                card_2_dict = cards[model2]
                # convert to GenerativeCard

                # card_1 = GenerativeCard(d=card_1_dict)
                # card_2 = GenerativeCard(d=card_2_dict)
                card_1 = card_1_dict
                card_2 = card_2_dict
                return {model1: str(card_1), model2: str(card_2)}
            except:
                print('No cards found')
                raise Exception

    else:
        cards = {}
        for model in models:
            path = get_latest_folder(topic, 'prog-reg', card_format, evaluator, model, 'generative')
            with open(path+'/cards/epoch_4_card.json', 'r') as f:
                cards[model] = str(GenerativeCard(d= json.load(f)))

        return cards
    

def sample_few_shots(meta, batch1, batch2, rep, k, different_k):
    if different_k < 0 or meta == 'openend':
        # just normal sample
        rslts = []
        for r in range(rep):
            random_portion = random.sample(range(len(batch1)), k - 1)
            # make sure r not in random_portion
            while r in random_portion:
                random_portion = random.sample(range(len(batch1)), k - 1)

            indices = random_portion + [r]
            random.shuffle(indices)
            rslts.append(indices)

        return np.array(rslts)

    # diff requires EXACTLY diff different answers  
    batch_1_ans = [b[3] for b in batch1]
    batch_2_ans = [b[3] for b in batch2]
    same_ans = [i for i in range(len(batch_1_ans)) if batch_1_ans[i] == batch_2_ans[i]]
    diff_ans = [i for i in range(len(batch_1_ans)) if batch_1_ans[i] != batch_2_ans[i]]

    few_shots = []

    same_k = k - different_k

    # not possibel if k > n, different_k > n

    if k > len(batch1) or different_k > len(batch1):
        raise ValueError('k or different_k cannot be greater than the number of questions in the batch')

    for r in range(rep):
        same_ind = random.sample(same_ans, same_k)
        diff_ind = random.sample(diff_ans, different_k)

        # shuffle 
        indices = same_ind + diff_ind

        # make sure indices are different, otherwise resample

        repetition = True
        while repetition:
            repetition = False
            for i in range(len(few_shots)):
                if set(few_shots[i]) == set(indices):
                    repetition = True
            
            if repetition:
                same_ind = random.sample(same_ans, same_k)
                diff_ind = random.sample(diff_ans, different_k)
                indices = same_ind + diff_ind

        random.shuffle(indices)
        few_shots.append(indices)

    return few_shots

class ContrastiveFullEvaluator:
    """
    Contrastive Evaluator:
    Given a card, guess which one of the two answers is authored by the model described by the card.
    """

    topic: str
    batches: Tuple[Batch, Batch]
    models: Tuple[str, str]
    cards: Tuple[str, str]
    evaluator_model: str
    rm: ResourceManager
    cm: Optional[CostManager]

    @staticmethod
    def word_count(card: str) -> int:
        """
        Count the number of words in a card.
        :param card: Card description.
        """
        return len(card.split())

    def __init__(
        self,
        meta: str,
        topic: str,
        batches: Tuple[Batch, Batch],
        models: Tuple[str, str],
        cards: Tuple[str, str],
        evaluator_model: str,
        rm: ResourceManager,
        cm: Optional[CostManager] = None,
        generation_method='contrastive',
        few_shot=False,
        k_shots=5,
        ensemble_cnt=1,
        diff_shots=-1,
        max_workers=40
    ):
        self.meta = meta
        self.topic = topic
        self.batches = batches
        self.models = models
        self.cards = cards
        self.evaluator_model = evaluator_model
        self.rm = rm
        self.cm = cm
        self.generation_method = generation_method
        self.few_shot = few_shot
        self.shots = k_shots
        self.ensemble_cnt = ensemble_cnt
        self.diff_shots = diff_shots
        self.max_workers = max_workers

    def ensemble_helper(self, index, card1, card2, qa_indices, target_model, shuffled_1, shuffled_2):
        system_prompt = self.rm.get_prompt("eval/contrastive/system").format(
            topic=self.topic
        )

        ensembled_shots = len(qa_indices)

        if len(qa_indices.shape) == 2:
           pass

        else:
            qa_indices = qa_indices.reshape(-1, 1)

        # no ensembled evaluator
        evaluators = [select_model(self.evaluator_model, system_prompt, self.cm) for _ in range(ensembled_shots)]

        qc_strs = [get_qc_str(self.meta, self.batches[shuffled_1], self.batches[1 - shuffled_1], qa_indices[i], student_names=['1', '2']) for i in range(ensembled_shots)]

        user_prompt = [self.rm.get_prompt("eval/contrastive/user").format(
            card1=self.cards[shuffled_2], card2=self.cards[1-shuffled_2], qc=qc
        ) for qc in qc_strs]

        r = [evaluators[i](user_prompt[i], temperature=0, use_json=False, max_new_tokens=100) for i in range(ensembled_shots)]

        try:
            # remove all ' or "
            r = [re.sub(r"['\"]", "", i) for i in r]
            # remove ( or )
            r = [re.sub(r"[\(\)]", "", i) for i in r]

            pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
            pattern2 = re.compile(r"\*\*prediction:\*\*\s*([ab])", re.IGNORECASE)
            match = [pattern.search(j.lower()) for j in r]

            decisions = []
            
            for i in range(len(match)):
                if match[i] == None:
                    match[i] = pattern2.search(r[i].lower())

                exp_1 = 'b'
                exp_2 = 'a'

                if (match[i].group()[-1]) == exp_1:
                    decisions.append(1)
                elif (match[i].group()[-1]) == exp_2:
                    decisions.append(0)
                else:
                    decisions.append(-1)
                    print('Error: ', r[i])
    
            # majority vote 
            decision = max(set(decisions), key = decisions.count)

        except Exception:
            decision = -1
            print(f"Error: {r}")

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "model_name": self.models[target_model],
            "shuffled_1": shuffled_1,
            "shuffled_2": shuffled_2,
            "conversation": [evaluators[i].messages for i in range(ensembled_shots)],
        }

        return decision, info_dict, index, target_model

    def helper(
        self, index: int, card1: str, card2: str, qc: str, target_model: int
    ) -> Tuple[int, Dict, int, int]:
        """
        Helper function to evaluate a single contrastive answer.
        :param index: Index of the question and answers.
        :param card: Card description.
        :param q: Question.
        :param a1: Answer 1.
        :param a2: Answer 2.
        :param target_model: The target model. Should produce a1 or a2.
        """
        system_prompt = self.rm.get_prompt("eval/contrastive/system").format(
            topic=self.topic
        )

        evaluator = select_model(self.evaluator_model, system_prompt, self.cm)
        user_prompt = self.rm.get_prompt("eval/contrastive/user").format(
            card1=card1, card2=card2, qc=qc
        )
        r = evaluator(user_prompt, temperature=0.2, use_json=False)
        # TODO: add decision extractor
        try:
            # pattern is prediction: (a|b)
            # pattern = re.compile(r"\([ab]\)(?!.*\([ab]\))")
            # remove ' or "
            r = re.sub(r"['\"]", "", r)
            pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
            match = pattern.search(r.lower())

            if match:
                decision = match.group()
            else:
                decision = ""
                        
            if 'a' in decision:
                decision = 0
            elif 'b' in decision:
                decision = 1
            else:
                decision = -1

        except Exception:
            decision = -1
            print(f"Error: {r}")

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "conversation": evaluator.messages,
        }
        return decision, info_dict, index, target_model
        
    def logit_helper(self, index, card1, card2, qa_indices, target_model, shuffled_1, shuffled_2):
        # target model is the model that produced the first answer, 
        # this is equivalent to A-1, B-2
        system_prompt = self.rm.get_prompt("eval/contrastive/system").format(topic=self.topic)
        
        ensembled_shots = len(qa_indices)
        if qa_indices.ndim == 1:
            qa_indices = qa_indices.reshape(-1, 1)  # convert to 2D array
        
        # qa_indices = qa_indices.T
        
        evaluators = [select_model(self.evaluator_model, system_prompt, self.cm) for _ in range(ensembled_shots)]
        qc_strs = []

        for j in range(ensembled_shots):
            qc_str = get_qc_str(self.meta, self.batches[shuffled_1], self.batches[1 - shuffled_1], qa_indices[j], student_names=['1', '2'])
            qc_strs.append(qc_str)

        user_prompt = [self.rm.get_prompt("eval/contrastive/user").format(
            card1=self.cards[shuffled_2],
            card2=self.cards[1-shuffled_2],
            qc=qc
        ) for qc in qc_strs]

        r = [evaluators[j].get_logits(user_prompt[j], post_fix='{prediction: Choice ') for j in range(len(qc_strs))]
        
        a_logits = [j[' A'] for j in r]
        b_logits = [j[' B'] for j in r]
        
        a_probs = [np.exp(o) / (np.exp(o) + np.exp(j)) for o, j in zip(a_logits, b_logits)]
        b_probs = [np.exp(j) / (np.exp(o) + np.exp(j)) for o, j in zip(a_logits, b_logits)]
        
        avg_a_prob = np.mean(a_probs)
        avg_b_prob = np.mean(b_probs)
        
        decision = 0 if avg_a_prob > avg_b_prob else 1
        
        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "model_name": self.models[target_model],
            "shuffled_1": shuffled_1,
            "shuffled_2": shuffled_2,
            "conversation": [evaluators[o].messages for o in range(len(qc_strs))],
            # "conversation": None
        }
        
        return decision, info_dict, index, target_model

    def main(self, num_times: int = 1, max_workers: int = -1) -> List[float]:
        """
        The main function to run the contrastive-answer evaluation.
        :param num_times: Number of times to run the evaluation.
        :param max_workers: Maximum number of parallel workers to use.
        """
        if max_workers == -1:
            max_workers = self.max_workers
            
        info_dict = {
            "type": "eval",
            "method": "contrastive-answer",
            "topic": self.topic,
            "models": self.models,
            "iterations": {},
        }
        metrics = []
        random.seed(42)

        for i in range(num_times):
            sub_info_dict = {}
            info_dict["iterations"][str(i)] = sub_info_dict
            correct = 0
            total = 0
            m = self.shots * self.ensemble_cnt
            rep = 60
            random_indices = sample_few_shots(self.meta, self.batches[0], self.batches[1], rep, m, self.diff_shots)
            random_indices = np.array(random_indices).reshape(rep, self.ensemble_cnt, self.shots)

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                for index in range(rep):
                    shuffled_1 = random.randint(0, 1)
                    shuffled_2 = random.randint(0, 1)
    
                    flipped = shuffled_1 ^ shuffled_2

                    if 'llama' in self.evaluator_model:
                        f = self.logit_helper

                    else:
                        f = self.ensemble_helper

                    futures.append(
                        executor.submit(
                            f,
                            index,
                            self.cards[0],
                            self.cards[1],
                            random_indices[index],
                            flipped,
                            shuffled_1,
                            shuffled_2
                        )
                    )


                for future in tqdm(as_completed(futures), desc="Evaluating..."):
                    if future.exception() is not None:
                        print(future.exception(), file=sys.stderr)
                        continue
                    decision, d, index, target_model = future.result()
                    sub_info_dict[str(index)] = d
                    if decision == target_model:
                        correct += 1
                    total += 1
            metrics.append(correct / total)
        mean = np.mean(metrics)
        sd = np.std(metrics)
        info_dict["metrics"] = {
            "accuracies": metrics,
            "mean_accuracy": mean,
            "sd_accuracy": sd,
        }
        name = f"eval_contrastive-answer_{'_'.join(self.models)}"
        print(
            f"{name}\n"
            f"accuracy: {mean} ± {sd} accuracies: {metrics}\n"
        )
        self.rm.dump_dict(name, info_dict)
        self.save_to_csv(metrics)
        return metrics

    def save_to_csv(self, metrics: List[float]):
        filename = 'ExpRslts/Contrastive/contrastive_ccaa_mmlu.csv'

        try:
            df = pd.read_csv(filename)
        except:
            df = pd.DataFrame(columns=['Student_Model_1', 
                                       'Student_Model_2',
                                       'Topic',
                                        'Card_Format', 
                                        'Epoch',
                                        'Contrastive_Accuracy', 
                                        'Oracle_Accuracy', 
                                        'Word_Count', 
                                        'Guesser',
                                        'Generation_Method',
                                        'Timestamp',
                                        'Card_type'])
            
        # current time
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        time_str = current_time.replace(' ', '_').replace(':', '-')

        card_type = 'few_shot' if self.few_shot else 'card'
            
        df = pd.concat([df, pd.DataFrame({'Student_Model_1': [self.models[0]],
                                        'Student_Model_2': [self.models[1]],
                                        'Topic': [self.topic],
                                        'Card_Format': ['dict'],
                                        'Epoch': [4],
                                        'Contrastive_Accuracy': [np.mean(metrics)],
                                        'Oracle_Accuracy': [compute_baseline_accuracy(self.topic, self.batches, self.models)[1]],
                                        'Word_Count': [str(self.word_count(self.cards[0])) + '/' + str(self.word_count(self.cards[1]))],
                                        'Guesser': [self.evaluator_model],
                                        'Generation_Method': [self.generation_method],
                                        'Timestamp': [time_str],
                                        'Card_type': [card_type]})], ignore_index=True)
        
        df.to_csv(filename, index=False)
