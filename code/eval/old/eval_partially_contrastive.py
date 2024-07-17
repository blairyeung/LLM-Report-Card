import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.card import GenerativeCard
from core.data import Batch, load_mmlu_batches
from core.models import CostManager, select_model
from core.utils import ResourceManager
from peripherals.oracle import compute_baseline_accuracy

STUDENT_NAMES = ["Bob", "Claire"]


def get_ground_truth_str(meta, batch, index):
    if meta == "mmlu":
        choices = batch.get_choices(index)
        gt = batch.get_true_answer(index)
        # convert 0 to A, 1 to B, etc.
        map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        if gt in range(len(choices)):
            return f"{map[gt]} .{choices[gt]}"
        else:
            return "Incorrect, the student chose None of the above"
    elif meta == "anthropic-eval":
        return batch.get_true_answer(index)
    else:
        raise ValueError(f"Unknown meta: {meta}")


def get_qc_str(meta, batch1, batch2, indices, student_names=None):
    if student_names is None:
        student_names = STUDENT_NAMES
    s = ""
    counter = 1
    for i in indices:
        student_a_correctness = (
            "correctly"
            if batch1.get_true_answer(i) == batch1.get_model_answer(i)
            else "incorrectly"
        )
        student_b_correctness = (
            "correctly"
            if batch2.get_true_answer(i) == batch2.get_model_answer(i)
            else "incorrectly"
        )

        s += f"## Question {counter}. {batch1.get_question(i)}\n"
        s += f"Correct Answer: {get_ground_truth_str(meta, batch1, i)}\n"
        s += f"### Student {student_names[0]} Answer:"
        s += f"{batch1.get_model_reasoning(i)}\nStudent {student_names[0]} answered the question {student_a_correctness}.\n\n"
        s += f"### Student {student_names[1]} Answer:"
        s += f"{batch2.get_model_reasoning(i)}\nStudent {student_names[1]} answered the question {student_b_correctness}.\n\n"

        counter += 1

    return s


def get_qc_str_aggregated(meta, batch1, batch2, indices, student_names=None):
    if student_names is None:
        student_names = STUDENT_NAMES
    s = ""
    counter = 1
    for i in indices:
        s += f"## Question {counter}. {batch1.get_question(i)}\n"
        # s += f'Correct Answer: {get_ground_truth_str(meta, batch1, i)}\n'
        counter += 1

    s += f"## Answer {student_names[0]}:\n"
    counter = 1
    for i in indices:
        student_a_correctness = (
            "correct"
            if batch1.get_true_answer(i) == batch1.get_model_answer(i)
            else "incorrect"
        )
        # s += f'### Anser of question {counter}:\n'
        s += f"{batch1.get_model_reasoning(i)}\n\n"
        counter += 1
        # s += f'{batch1.get_model_reasoning(i)}\n**This answer is**: {student_a_correctness}\n\n'

    s += f"## Answer {student_names[1]}:\n"
    counter = 1
    for i in indices:
        student_b_correctness = (
            "Correct"
            if batch2.get_true_answer(i) == batch2.get_model_answer(i)
            else "Incorrect"
        )
        # s += f'### Answer of question {counter}:\n'
        s += f"{batch2.get_model_reasoning(i)}\n\n"
        counter += 1
        # s += f'{batch2.get_model_reasoning(i)}\n**This answer is**: {student_b_correctness}\n\n'

        counter += 1

    return s


def get_qc_str_one_batch(meta, batch, indices, student_names=None):
    if student_names is None:
        student_names = STUDENT_NAMES
    s = ""
    counter = 1
    for i in indices:
        s += f"## Question {counter}. {batch.get_question(i)}\n"
        s += f"Correct Answer: {get_ground_truth_str(meta, batch, i)}\n"
        s += f"### Student Completion\n"
        s += f"{batch.get_model_reasoning(i)}\n\n"
        counter += 1

    return s


def get_latest_folder(
        topic, optim_method, card_format, evaluator, model,
        generation_method="generative"
):
    folder_root = f"outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}"
    all_folders = os.listdir(
        f"outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}"
    )
    all_folders.sort()
    all_folders = all_folders[::-1]
    for folder in all_folders:
        if re.match(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", folder):
            return f"{folder_root}/{folder}"
    return None


def load_cards(topic, card_format, models, evaluator, method="contrastive"):
    if method == "contrastive":
        try:
            model_str = f"['{models[0]}', '{models[1]}']"
            path = get_latest_folder(
                topic, "prog-reg", card_format, evaluator, model_str, "contrastive"
            )
            with open(path + "/cards/epoch_4_card.json", "r") as f:
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
                path = get_latest_folder(
                    topic, "prog-reg", card_format, evaluator, model_str, "contrastive"
                )
                with open(path + "/cards/epoch_4_card.json", "r") as f:
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
                print("No cards found")
                raise Exception

    else:
        cards = {}
        for model in models:
            path = get_latest_folder(
                topic, "prog-reg", card_format, evaluator, model, "generative"
            )
            with open(path + "/cards/epoch_4_card.json", "r") as f:
                cards[model] = str(GenerativeCard(d=json.load(f)))

        return cards


def sample_few_shots(meta, batch1, batch2, rep, k, different_k):
    if different_k < 0 or meta == "openend":
        # just normal sample
        rslts = []
        for r in range(rep):
            random_portion = random.sample(range(len(batch1)), k - 1)
            # make sure r not in random_portion
            while r in random_portion:
                random_portion = random.sample(range(len(batch1)), k - 1)

            indices = random_portion + [r]
            random.shuffle(indices)
            rslts.append(np.array(indices))

        return rslts

    # diff requires EXACTLY diff different answers
    batch_1_ans = [b[3] for b in batch1]
    batch_2_ans = [b[3] for b in batch2]
    same_ans = [i for i in range(len(batch_1_ans)) if batch_1_ans[i] == batch_2_ans[i]]
    diff_ans = [i for i in range(len(batch_1_ans)) if batch_1_ans[i] != batch_2_ans[i]]

    few_shots = []

    same_k = k - different_k

    # not possibel if k > n, different_k > n

    if k > len(batch1) or different_k > len(batch1):
        raise ValueError(
            "k or different_k cannot be greater than the number of questions in the batch"
        )

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


class ContrastiveCardEvaluator:
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
            topic: str,
            batches: Tuple[Batch, Batch],
            models: Tuple[str, str],
            cards: Tuple[str, str],
            evaluator_model: str,
            rm: ResourceManager,
            cm: Optional[CostManager] = None,
            generation_method="contrastive",
            few_shot=False,
            k_shots=5,
    ):
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
        system_prompt = self.rm.get_prompt("eval/contrastive-card/system").format(
            topic=self.topic
        )
        evaluator = select_model(self.evaluator_model, system_prompt, self.cm)
        user_prompt = self.rm.get_prompt("eval/contrastive-card/user").format(
            card1=card1,
            card2=card2,
            qc=qc,
            a_name=STUDENT_NAMES[0],
            b_name=STUDENT_NAMES[1],
        )
        r = evaluator(user_prompt, temperature=0, use_json=False)
        # TODO: add decision extractor
        try:
            # prediction: (1|2)
            pattern = re.compile(r"verdict: \d")
            pattern2 = re.compile(r"verdict: card \d")
            # either match is fine
            match = pattern.search(r.lower())
            if match is None:
                match = pattern2.search(r.lower())

            decision = int(match.group()[-1]) - 1

        except Exception:
            decision = -1
            print(f"Error: {r}")

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "conversation": evaluator.messages,
        }
        return decision, info_dict, index, target_model

    def main(self, num_times: int = 1, max_workers: int = 30) -> List[float]:
        """
        The main function to run the contrastive-answer evaluation.
        :param num_times: Number of times to run the evaluation.
        :param max_workers: Maximum number of parallel workers to use.
        """
        print(f"Evaluating with guesser {self.evaluator_model}")
        info_dict = {
            "type": "eval",
            "method": "contrastive-answer",
            "topic": self.topic,
            "models": self.models,
            "iterations": {},
        }
        metrics = []
        for i in range(num_times):
            sub_info_dict = {}
            info_dict["iterations"][str(i)] = sub_info_dict
            correct = 0
            total = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index in range(len(self.batches[0])):
                    n = len(self.batches[0])

                    indices = random.sample(range(n), self.shots - 1) + [index]
                    qc_str = get_qc_str_one_batch(self.batches[0], indices)

                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[0],
                            self.cards[1],
                            qc_str,
                            0,
                        )
                    )

                    qc_str = get_qc_str_one_batch(self.batches[1], indices)

                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[0],
                            self.cards[1],
                            qc_str,
                            1,
                        )
                    )
                # let desc be accuracy so far
                for future in tqdm(as_completed(futures), desc=f"Evaluating..."):
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
        print(f"{name}\n" f"accuracy: {mean} ± {sd} accuracies: {metrics}\n")
        self.rm.dump_dict(name, info_dict)
        self.save_to_csv(metrics)
        return metrics

    def save_to_csv(self, metrics: List[float]):
        filename = "ExpRslts/Contrastive/contrastive_cca.csv"

        try:
            df = pd.read_csv(filename)
        except:
            df = pd.DataFrame(
                columns=[
                    "Student_Model_1",
                    "Student_Model_2" "Topic",
                    "Card_Format",
                    "Epoch",
                    "Contrastive_Accuracy",
                    "Oracle_Accuracy",
                    "Word_Count",
                    "Guesser",
                    "Generation_Method",
                    "Timestamp",
                    "Card_type",
                ]
            )

        # current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time_str = current_time.replace(" ", "_").replace(":", "-")
        card_type = "few_shot" if self.few_shot else "card"

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Student_Model_1": [self.models[0]],
                        "Student_Model_2": [self.models[1]],
                        "Topic": [self.topic],
                        "Card_Format": ["dict"],
                        "Epoch": [4],
                        "Contrastive_Accuracy": [np.mean(metrics)],
                        "Oracle_Accuracy": [
                            compute_baseline_accuracy(
                                self.topic, self.batches, self.models
                            )[1]
                        ],
                        "Word_Count": [
                            str(self.word_count(self.cards[0]))
                            + "/"
                            + str(self.word_count(self.cards[1]))
                        ],
                        "Guesser": [self.evaluator_model],
                        "Generation_Method": [self.generation_method],
                        "Timestamp": [time_str],
                        "Card_type": [card_type],
                    }
                ),
            ],
            ignore_index=True,
        )

        df.to_csv(filename, index=False)


class ContrastiveAnswerEvaluator:
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
            evaluator_model,
            rm: ResourceManager,
            cm: Optional[CostManager] = None,
            generation_method="contrastive",
            few_shot=False,
            k_shots=5,
            ensemble_cnt=1,
            diff_shots=-1,
            max_workers=40,
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

    def ensembled_helper(self, index, card, qa_indices, target_model, shuffled):
        system_prompt = self.rm.get_prompt("eval/contrastive-answer/system").format(
            topic=self.topic
        )

        ensembled_shots = len(qa_indices)

        if len(qa_indices.shape) == 2:
            pass

        else:
            qa_indices = qa_indices.reshape(-1, 1)

        # no ensembled evaluator
        evaluators = [
            select_model(self.evaluator_model, system_prompt, self.cm)
            for i in range(ensembled_shots)
        ]

        qc_strs = [
            get_qc_str(
                self.meta,
                self.batches[shuffled],
                self.batches[1 - shuffled],
                qa_indices[i],
            )
            for i in range(ensembled_shots)
        ]

        user_prompt = [
            self.rm.get_prompt("eval/contrastive-answer/user").format(card=card, qc=qc)
            for qc in qc_strs
        ]

        r = [
            evaluators[i](
                user_prompt[i], temperature=0, use_json=False, max_new_tokens=768
            )
            for i in range(ensembled_shots)
        ]

        try:
            # remove all ' or "
            r = [re.sub(r"['\"]", "", i) for i in r]
            pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
            pattern2 = re.compile(r"\*\*prediction:\*\*\s*([ab])", re.IGNORECASE)

            # either match is fine
            match = [pattern.search(j.lower()) for j in r]

            decisions = []

            for i in range(len(match)):
                if match[i] == None:
                    match[i] = pattern2.search(r[i].lower())

                exp_1 = "a" if shuffled == 1 else "b"
                exp_2 = "a" if shuffled == 0 else "b"

                if (match[i].group()[-1]) == exp_1:
                    decisions.append(1)
                elif (match[i].group()[-1]) == exp_2:
                    decisions.append(0)
                else:
                    decisions.append(-1)
                    print("Error: ", r[i])

            # majority vote
            decision = max(set(decisions), key=decisions.count)

        except Exception:
            decision = -1
            print(f"Error: {r}")

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "model_name": self.models[target_model],
            "shuffled": shuffled,
            "conversation": [evaluators[i].messages for i in range(ensembled_shots)],
        }

        return decision, info_dict, index, target_model

    def helper(
            self,
            index: int,
            card: str,
            qa_completion: str,
            target_model: int,
            shuffled: int,
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

        system_prompt = self.rm.get_prompt("eval/contrastive-answer/system").format(
            topic=self.topic
        )

        if type(self.evaluator_model) == list:
            evaluator = [
                select_model(evaluator, system_prompt, self.cm)
                for evaluator in self.evaluator_model
            ]
        else:
            evaluator = select_model(self.evaluator_model, system_prompt, self.cm)

        user_prompt = self.rm.get_prompt("eval/contrastive-answer/user").format(
            card=card, qc=qa_completion
        )

        if type(self.evaluator_model) == list:

            r = [
                evaluator[i](
                    user_prompt, temperature=0, use_json=False, max_new_tokens=64
                )
                for i in range(len(evaluator))
            ]
            try:
                # verdict: a|b, not 1|2
                pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
                pattern2 = re.compile(r"\*\*prediction:\*\*\s*([ab])", re.IGNORECASE)
                match = [pattern.search(j.lower()) for j in r]

                decisions = []

                for i in range(len(match)):
                    if match[i] == None:
                        match[i] = pattern2.search(r[i].lower())

                    if (match[i].group()[-1]) == "b":
                        decisions.append(1)
                    elif (match[i].group()[-1]) == "a":
                        decisions.append(0)
                    else:
                        decisions.append(-1)

                # majority vote
                decision = max(set(decisions), key=decisions.count)

            except Exception:
                decision = -1
                print(f"Error: {r}")

            # align decision with shuffled

            info_dict = {
                "decision": decision,
                "target_model": target_model,
                "conversation": [e.messages for e in evaluator],
            }

            return decision, info_dict, index, target_model

        else:
            r = evaluator(user_prompt, temperature=0, use_json=False, max_new_tokens=64)

            try:
                pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
                pattern2 = re.compile(r"\*\*prediction:\*\*\s*([ab])", re.IGNORECASE)

                # either match is fine
                match = pattern.search(r.lower())
                if match is None:
                    match = pattern2.search(r.lower())

                b_exp = "a" if shuffled == 1 else "b"
                a_exp = "a" if shuffled == 0 else "b"

                if (match.group()[-1]) == b_exp:
                    decision = 1

                elif (match.group()[-1]) == a_exp:
                    decision = 0

                else:
                    # print error
                    print("Error: ", r)
                    decision = -1

            except Exception:
                decision = -1
                print(f"Error: {r}")

            info_dict = {
                "decision": decision,
                "target_model": target_model,
                "conversation": evaluator.messages,
            }

            # desctroy evaluator
            evaluator.clear_conversations()

            return decision, info_dict, index, target_model

    def logit_helper(self, index, card, qa_completion_indices, target_model, shuffled):
        system_prompt = self.rm.get_prompt("eval/contrastive-answer/system").format(
            topic=self.topic
        )

        if type(self.evaluator_model) == list:
            evaluator = [
                select_model(evaluator, system_prompt, self.cm)
                for evaluator in self.evaluator_model
            ]
        else:
            evaluator = select_model(self.evaluator_model, system_prompt, self.cm)

        if len(qa_completion_indices.shape) == 2:
            pass
        else:
            qa_completion_indices = qa_completion_indices.reshape(-1, 1)

        ensemble_shots = len(qa_completion)

        qa_completion = [
            get_qc_str(self.meta, self.batches[shuffled], self.batches[1 - shuffled], q)
            for q in qa_completion_indices
        ]

        user_prompts = [
            self.rm.get_prompt("eval/contrastive-answer/user").format(
                card=card, qc=q, a_name=STUDENT_NAMES[0], b_name=STUDENT_NAMES[1]
            )
            for q in qa_completion
        ]

        ensembled = type(self.evaluator_model) == list

        if ensembled:

            r = [
                evaluator[i].get_logit(user_prompts[i], post_fix="{prediction: ")
                for i in range(len(evaluator))
            ]

            # softmax
            a_probs = [
                np.exp(r[i][STUDENT_NAMES[0]])
                / (np.exp(r[i][STUDENT_NAMES[0]]) + np.exp(r[i][STUDENT_NAMES[1]]))
                for i in range(len(r))
            ]
            b_probs = [
                np.exp(r[i][STUDENT_NAMES[1]])
                / (np.exp(r[i][STUDENT_NAMES[0]]) + np.exp(r[i][STUDENT_NAMES[1]]))
                for i in range(len(r))
            ]

            a_prob = np.mean(a_probs)
            b_prob = np.mean(b_probs)

            if a_prob > b_prob:
                decision = 0

            else:
                decision = 1

            # align decision with shuffled

            info_dict = {
                "decision": decision,
                "target_model": target_model,
                "conversation": [e.messages for e in evaluator],
            }

            return decision, info_dict, index, target_model

        else:
            r = evaluator.get_logits(user_prompts[0], post_fix=" {prediction: ")
            # print(r)
            a_logit = r[" " + STUDENT_NAMES[0]]
            b_logit = r[" " + STUDENT_NAMES[1]]

            # softmax
            a_prob = np.exp(a_logit) / (np.exp(a_logit) + np.exp(b_logit))
            b_prob = np.exp(b_logit) / (np.exp(a_logit) + np.exp(b_logit))

            if a_prob > b_prob:
                decision = 0

            else:
                decision = 1

            if shuffled:
                decision = 1 - decision

            # print(f'Target model: {target_model} Decision: {decision} A prob: {a_prob} B prob: {b_prob}')

            info_dict = {
                "decision": decision,
                "target_model": target_model,
                "conversation": evaluator.messages,
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
        for i in range(num_times):
            sub_info_dict = {}
            info_dict["iterations"][str(i)] = sub_info_dict
            correct = 0
            total = 0

            batch1 = self.batches[0]
            batch2 = self.batches[1]

            samples = sample_few_shots(
                self.meta,
                batch1,
                batch2,
                rep=(len(self.batches[0])),
                k=self.shots * self.ensemble_cnt,
                different_k=self.diff_shots,
            )

            if isinstance(self.evaluator_model, str):
                self.evaluator_model = [
                    self.evaluator_model for _ in range(self.ensemble_cnt)
                ]

            elif len(self.evaluator_model) == 1:
                self.evaluator_model = [
                    self.evaluator_model[0] for _ in range(self.ensemble_cnt)
                ]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index in range(len(self.batches[0])):

                    qa_completion_indices = samples[index]

                    shuffle_1, shuffle_2 = random.randint(0, 1), random.randint(0, 1)

                    batches = [batch1, batch2]
                    ensembled_helper = True
                    if ensembled_helper:
                        qa_completion_indices = qa_completion_indices.reshape(
                            self.shots, self.ensemble_cnt
                        )
                        futures.append(
                            executor.submit(
                                self.logit_helper,
                                index,
                                self.cards[0],
                                qa_completion_indices,
                                0,
                                shuffle_1,
                            )
                        )

                        futures.append(
                            executor.submit(
                                self.logit_helper,
                                index,
                                self.cards[1],
                                qa_completion_indices,
                                1,
                                shuffle_2,
                            )
                        )
                    # else:

                    #     qa_completion_1 = get_qc_str(self.meta, batches[shuffle_1], batches[1 - shuffle_1], qa_completion_indices)
                    #     qa_completion_2 = get_qc_str(self.meta, batches[shuffle_2], batches[1 - shuffle_2], qa_completion_indices)

                    #     futures.append(
                    #         executor.submit(
                    #             self.logit_helper,
                    #             index,
                    #             self.cards[0],
                    #             qa_completion_1,
                    #             0,
                    #             shuffle_1
                    #         )
                    #     )

                    #     futures.append(
                    #         executor.submit(
                    #             self.logit_helper,
                    #             index,
                    #             self.cards[1],
                    #             qa_completion_2,
                    #             1,
                    #             shuffle_2
                    #         )
                    #     )

                # why is acc_est always 0?
                card1_correct = 0
                card2_correct = 0

                card_1_total = 0
                card_2_total = 0
                for future in tqdm(as_completed(futures), desc=f"Evaluating..."):
                    if future.exception() is not None:
                        print(future.exception(), file=sys.stderr)
                        continue

                    decision, d, index, target_model = future.result()
                    sub_info_dict[str(index)] = d

                    if decision == target_model:
                        correct += 1

                        if target_model == 0:
                            card1_correct += 1

                        else:
                            card2_correct += 1

                    if target_model == 0:
                        card_1_total += 1
                    else:
                        card_2_total += 1
                    total += 1

            model1_acc = card1_correct / card_1_total
            model2_acc = card2_correct / card_2_total
            metrics.append([correct / total, model1_acc, model2_acc])
        mean = np.mean(metrics)
        sd = np.std(metrics)
        info_dict["metrics"] = {
            "accuracies": metrics,
            "mean_accuracy": mean,
            "model1_accuracy": model1_acc,
            "model2_accuracy": model2_acc,
        }
        name = f"eval_contrastive-answer_{'_'.join(self.models)}"
        print(f"{name}\n" f"accuracy: {mean} ± {sd} accuracies: {metrics}\n")

        print(f"Model 1 accuracy: {model1_acc}")
        print(f"Model 2 accuracy: {model2_acc}")
        self.rm.dump_dict(name, info_dict)
        self.save_to_csv(metrics)
        return metrics

    def save_to_csv(self, metrics: List[float]):
        filename = "ExpRslts/Contrastive/contrastive_mmlu.csv"

        try:
            df = pd.read_csv(filename)
        except:

            df = pd.DataFrame(
                columns=[
                    "Student_Model_1",
                    "Student_Model_2",
                    "Topic",
                    "Card_Format",
                    "Epoch",
                    "Contrastive_Accuracy",
                    "Card1_Accuracy",
                    "Card2_Accuracy",
                    "Oracle_Accuracy",
                    "Word_Count",
                    "Guesser",
                    "Generation_Method",
                    "Timestamp",
                    "Card_type",
                ]
            )

        # current time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time_str = current_time.replace(" ", "_").replace(":", "-")
        card_type = "few_shot" if self.few_shot else "card"

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "Student_Model_1": [self.models[0]],
                        "Student_Model_2": [self.models[1]],
                        "Topic": [self.topic],
                        "Card_Format": ["dict"],
                        "Epoch": [4],
                        "Contrastive_Accuracy": [np.mean(metrics)],
                        "Card1_Accuracy": [metrics[0][1]],
                        "Card2_Accuracy": [metrics[0][2]],
                        "Oracle_Accuracy": [0.5],
                        "Word_Count": [
                            str(self.word_count(self.cards[0]))
                            + "/"
                            + str(self.word_count(self.cards[1]))
                        ],
                        "Guesser": [self.evaluator_model],
                        "Generation_Method": [self.generation_method],
                        "Timestamp": [time_str],
                        "Card_type": card_type,
                    }
                ),
            ],
            ignore_index=True,
        )

        df.to_csv(filename, index=False)


def get_latest_folder(
        topic, optim_method, card_format, evaluator, model,
        generation_method="generative"
):
    folder_root = f"outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}"
    all_folders = os.listdir(
        f"outputs/{generation_method}/{topic}/{optim_method}/{card_format}/{evaluator}/{model}"
    )
    all_folders.sort()
    all_folders = all_folders[::-1]
    for folder in all_folders:
        if re.match(r"\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", folder):
            return f"{folder_root}/{folder}"
    return None


if __name__ == "__main__":
    generation = "generative"
    model1 = "Mistral-7B-Instruct-v0.2"
    model2 = "Mixtral-8x7B-Instruct-v0.1"
    models = (model1, model2)

    topic = "high_school_mathematics"

    batch1 = load_mmlu_batches("../../datasets/mmlu/", topic, model1, "test", [60], False)[0]
    batch2 = load_mmlu_batches("../../datasets/mmlu/", topic, model2, "test", [60], False)[0]
    batches = (batch1, batch2)

    evaluator = "meta-llama/Meta-Llama-3-70B-Instruct"

    card1 = "This model is very good."
    card2 = "This model is really bad."
    cards = (str(card1), str(card2))

    eval_method = ContrastiveAnswerEvaluator(
        "mmlu",
        topic,
        batches,
        models,
        cards,
        evaluator,
        ResourceManager("eval_contrastive-answer-scott"),
        generation_method=generation,
    )

    eval_method.main(max_workers=1)
