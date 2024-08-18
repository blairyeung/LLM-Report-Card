import itertools
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm

from core.card import GenerativeCard
from core.models import CostManager, select_model, ROLE_USER, ROLE_ASSISTANT
from core.utils import ResourceManager, sample_few_shots, get_choice_str, append_to_csv
from dataset.data import Batch, load_batches

STUDENT_NAMES = ["Yarix", "Vionelle"]


class Contrastive2R2AEvaluator:
    """
    Full Contrastive Evaluator:
    Given two cards and two answers, guess which model writes which answer.
    """

    dataset: str
    topic: str
    batches: Tuple[Batch, Batch]
    student_models: Tuple[str, str]
    cards: Tuple[str, str]
    guesser_model_names: List[str]
    rm: ResourceManager
    cm: Optional[CostManager]
    cot: bool
    k_shots: int
    max_workers: int
    csv_path: Optional[str]
    few_shot: bool
    paraphrased: bool

    def __init__(
        self,
        dataset: str,
        topic: str,
        batches: Tuple[Batch, Batch],
        models: Tuple[str, str],
        cards: Tuple[str, str],
        evaluator_model_names: List[str],
        rm: ResourceManager,
        cm: Optional[CostManager] = None,
        cot: bool = False,
        k_shots: int = 3,
        max_workers: int = 60,
        csv_path: Optional[str] = None,
        few_shot: bool = False,
        paraphrase: bool = False,
    ):
        self.dataset = dataset
        self.topic = topic
        self.batches = batches
        self.student_models = models
        self.cards = cards
        self.guesser_model_names = evaluator_model_names
        self.rm = rm
        self.cm = cm if cm is not None else CostManager()
        self.cot = cot
        self.k_shots = k_shots
        self.max_workers = max_workers
        self.csv_path = csv_path
        self.few_shot = few_shot
        self.paraphrased = paraphrase

    def main(
        self,
        num_times: int = 1,
        num_samples: int = -1,
        max_workers: int = -1,
        mode: Literal["simplified", "full", "partial", "logit"] = "partial",
    ) -> List[List[float]]:
        """
        The main function to run the full contrastive evaluation.
        :param num_times: Number of times to run the evaluation.
        :param num_samples: Number of samples to evaluate.
                            If -1, then use len(self.batches[0]).
        :param max_workers: Maximum number of parallel workers to use.
        :param mode: Evaluation mode.
        """
        if max_workers == -1:
            max_workers = self.max_workers
        if num_samples == -1:
            num_samples = len(self.batches[0])
        info_dict = {
            "type": "eval",
            "method": f"contrastive-2R2A-{mode}",
            "topic": self.topic,
            "models": self.student_models,
            "iterations": [],
        }
        metrics, decisions = [], []
        for i in range(num_times):
            sub_info_dict = {}
            info_dict["iterations"].append(sub_info_dict)
            decisions.append([])

            samples = sample_few_shots(num_samples, len(self.batches[0]), self.k_shots)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index in range(num_samples):
                    sample = samples[index]

                    if mode == "full":
                        f = self.helper_full
                    elif mode == "partial":
                        f = self.helper_partial
                        # f = self.helper_partial_logits
                    elif mode == "logit":
                        f = self.helper_partial_logits
                    elif mode == "simplified":
                        f = self.helper_simplified
                    else:
                        raise NotImplementedError

                    futures.append(
                        executor.submit(
                            f,
                            index,
                            sample,
                            bool(random.randint(0, 1)),  # random flip
                        )
                    )

                total = 0
                card1_correct = 0
                card2_correct = 0
                for future in tqdm(
                    as_completed(futures), desc=f"Evaluating...", total=len(futures)
                ):
                    if future.exception() is not None:
                        print(future.exception(), file=sys.stderr)
                        continue
                    correct1, correct2, d, index, flip = future.result()
                    sub_info_dict[str(index)] = d
                    decisions[i].append((flip, d["decision"]))
                    card1_correct += int(correct1)
                    card2_correct += int(correct2)
                    total += 2

            model1_acc = card1_correct / (total / 2)
            model2_acc = card2_correct / (total / 2)
            metrics.append(
                [(card1_correct + card2_correct) / total, model1_acc, model2_acc]
            )

        # calculate mean and std
        metrics = np.asarray(metrics).reshape(-1, 3)
        mean = np.mean(metrics, axis=0)
        sd = np.std(metrics, axis=0)
        counter = Counter()
        for decision_list in decisions:
            counter += Counter(decision_list)
        for key in counter:
            counter[key] /= num_times
        counter_dict = {str(k): counter[k] for k in counter}
        info_dict["metrics"] = {
            "accuracies": metrics[:, 0].tolist(),
            "model1_accuracies": metrics[:, 1].tolist(),
            "model2_accuracies": metrics[:, 2].tolist(),
            "mean_accuracy": mean[0],
            "sd_accuracy": sd[0],
            "mean_model1_accuracy": mean[1],
            "sd_model1_accuracy": sd[1],
            "mean_model2_accuracy": mean[2],
            "sd_model2_accuracy": sd[2],
            "average_decision_count": counter_dict,
        }
        name = f"eval_contrastive-full_{'_'.join(self.student_models)}"
        self.rm.dump_dict(name, info_dict)
        print(f"{name}\nMetrics: {metrics}\n")
        print(f"Accuracy: {mean[0]} ± {sd[0]}")
        print(f"Model 1 accuracy: {mean[1]} ± {sd[1]}")
        print(f"Model 2 accuracy: {mean[2]} ± {sd[2]}")
        print(f"Average Decision Count: {counter}")
        print(f"Cost so far: {self.cm.get_cost()}")

        if self.csv_path is not None:  # save to csv
            row_data = [
                [
                    self.student_models[0],
                    self.student_models[1],
                    m[0],
                    m[1],
                    m[2],
                    self.guesser_model_names[0],
                ]
                for m in metrics
            ]
            append_to_csv(
                self.csv_path,
                row_data,
                [
                    "model_1",
                    "model_2",
                    "accuracy",
                    "model_1_accuracy",
                    "model_2_accuracy",
                    "guesser",
                ],
            )

        hyperparameters = {
            "dataset": self.dataset,
            "topic": self.topic,
            "students": self.student_models,
            "guessers": self.guesser_model_names,
            "cot": self.cot,
            "few_shot": self.few_shot,
            "mode": mode,
            "k_shots": self.k_shots,
            "paraphrased": self.paraphrased,
            "num_times": num_times,
            "num_samples": num_samples,
            "max_workers": max_workers,
        }
        self.rm.dump_dict(f"hyperparameters", hyperparameters)
        return metrics.tolist()

    def helper_partial(
        self,
        index: int,
        sample_indices: List[int],
        flip: bool,
    ) -> Tuple[bool, bool, Dict, int, bool]:
        """
        Helper function to evaluate a single full contrastive.
        This helper ask once and assume the other one is the opposite.
        Note that we randomly ask the first or the second.
        :param index: Index of the question and answers.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        system_prompt, user_prompt = self.get_prompts(sample_indices, flip)
        guessers = [
            select_model(guesser, system_prompt, self.cm)
            for guesser in self.guesser_model_names
        ]

        if self.cot:
            r1 = [
                guesser(user_prompt, temperature=0, use_json=False)
                for guesser in guessers
            ]
        else:
            for guesser in guessers:
                guesser.add_message(ROLE_USER, user_prompt)
                guesser.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
        s0, s1 = STUDENT_NAMES
        # generate a random number to decide which one to ask
        target = random.randint(0, 1)
        if target == 0:
            r2 = [
                guesser(
                    f"Who ({s0} or {s1}) authors all The First Responses? Give a one word answer.",
                    temperature=0,
                    use_json=False,
                    max_new_tokens=10,
                )
                for guesser in guessers
            ]
        else:
            r2 = [
                guesser(
                    f"Who ({s0} or {s1}) authors all The Second Responses? Give a one word answer.",
                    temperature=0,
                    use_json=False,
                    max_new_tokens=10,
                )
                for guesser in guessers
            ]
        decisions = []
        s0, s1 = s0.lower(), s1.lower()
        for d in r2:
            d = d.lower()
            if s0 in d:
                decisions.append(0)
            elif s1 in d:
                decisions.append(1)
            else:
                decisions.append(-1)
        # majority vote
        decision = max(set(decisions), key=decisions.count)
        if flip:
            if decision != -1:
                decision = 1 - decision
        info_dict = {
            "decision": decision,
            "target_decision": target,
            "flip": flip,
            "conversation": [e.messages for e in guessers],
        }
        # we assume the other one is the opposite, so if this is correct, the other is also correct
        correct = decision == target
        return correct, correct, info_dict, index, flip

    def helper_partial_logits(
        self,
        index: int,
        sample_indices: List[int],
        flip: bool,
    ) -> Tuple[bool, bool, Dict, int, bool]:
        """
        Helper function to evaluate a single full contrastive.
        This helper ask once and assume the other one is the opposite.
        Note that we randomly ask the first or the second.
        :param index: Index of the question and answers.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """

        def get_decision(logits: Dict, pairs: List[str]):
            """
            Get the decision from the logits.
            """

            def dict_key_helper(logit_dict, item: str):
                """
                Find max possible key in dict.
                """

                max_val = -100
                max_key = -1
                for key, value in logit_dict.items():
                    modified = key.replace(" ", "").lower()
                    if modified in item and value > max_val:
                        max_val = value
                        max_key = key

                return max_key, max_val

            decision = []
            # print(logits)
            for pair in pairs:
                decision.append(dict_key_helper(logits, pair)[1])

            # softmax decision
            decision = np.exp(decision) / np.sum(np.exp(decision))

            return decision[0] > decision[1], decision

        system_prompt, user_prompt = self.get_prompts(sample_indices, flip)
        guessers = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.guesser_model_names
        ]

        if self.cot:
            r1 = [
                guesser(user_prompt, temperature=0, use_json=False)
                for guesser in guessers
            ]

        else:
            for guesser in guessers:
                guesser.add_message(ROLE_USER, user_prompt)
                guesser.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )

        s0, s1 = STUDENT_NAMES
        # generate a random number to decide which one to ask
        target = random.randint(0, 1)
        if target == 0:
            r2 = [
                guesser.get_logits(
                    f"Who ({s0} or {s1}) authors all The First Responses? Give a one word answer.",
                )
                for guesser in guessers
            ]
        else:
            r2 = [
                guesser.get_logits(
                    f"Who ({s0} or {s1}) authors all The Second Responses? Give a one word answer.",
                )
                for guesser in guessers
            ]
        decisions = []
        s0, s1 = s0.lower(), s1.lower()
        for d in r2:
            decision = 1 - get_decision(d, [s0, s1])[0]
            decisions.append(decision)
        # majority vote
        decision = max(set(decisions), key=decisions.count)
        if flip:
            if decision != -1:
                decision = 1 - decision
        info_dict = {
            "decision": decision,
            "target_decision": target,
            "flip": flip,
            "conversation": [e.messages for e in guessers],
        }
        # we assume the other one is the opposite, so if this is correct, the other is also correct
        correct = decision == target
        return correct, correct, info_dict, index, flip


    def helper_full(
        self,
        index: int,
        sample_indices: List[int],
        flip: bool,
    ) -> Tuple[bool, bool, Dict, int, bool]:
        """
        Helper function to evaluate a single full contrastive.
        This helper ask twice.
        :param index: Index of the question and answers.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        system_prompt, user_prompt = self.get_prompts(sample_indices, flip)
        guessers = [
            select_model(guesser, system_prompt, self.cm)
            for guesser in self.guesser_model_names
        ]

        if self.cot:
            r1 = [
                guesser(user_prompt, temperature=0, use_json=False)
                for guesser in guessers
            ]
        else:
            for guesser in guessers:
                guesser.add_message(ROLE_USER, user_prompt)
                guesser.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
        s0, s1 = STUDENT_NAMES
        r2 = [
            guesser(
                f"Who ({s0} or {s1}) authors all The First Responses? Give a one word answer.",
                temperature=0,
                use_json=False,
            )
            for guesser in guessers
        ]
        for guesser in guessers:  # remove the last 2 conversations
            guesser.messages.pop()
            guesser.messages.pop()
        r3 = [
            evaluator(
                f"Who ({s0} or {s1}) authors all The Second Responses? Give a one word answer.",
                temperature=0,
                use_json=False,
            )
            for evaluator in guessers
        ]
        # print(r2, r3)
        decisions1, decisions2 = [], []
        s0, s1 = s0.lower(), s1.lower()
        for d1, d2 in zip(r2, r3):
            d1, d2 = d1.lower(), d2.lower()
            if s0 in d1 and s1 in d1:
                decisions1.append(-1)
            elif s0 in d1:
                decisions1.append(0)
            elif s1 in d1:
                decisions1.append(1)
            else:
                decisions1.append(-1)

            if s0 in d2 and s1 in d2:
                decisions2.append(-1)
            elif s0 in d2:
                decisions2.append(0)
            elif s1 in d2:
                decisions2.append(1)
            else:
                decisions2.append(-1)
        # majority vote
        decision1 = max(set(decisions1), key=decisions1.count)
        decision2 = max(set(decisions2), key=decisions2.count)
        if flip:
            if decision1 != -1:
                decision1 = 1 - decision1
            if decision2 != -1:
                decision2 = 1 - decision2
        # note that the correct decision is (0, 1), as we already handled flip
        info_dict = {
            "decision": (decision1, decision2),
            "target_decision": (0, 1),
            "flip": flip,
            "conversation": [e.messages for e in guessers],
        }
        return decision1 == 0, decision2 == 1, info_dict, index, flip


    def helper_simplified(
        self,
        index: int,
        sample_indices: List[int],
        flip: bool,
    ) -> Tuple[bool, bool, Dict, int, bool]:
        """
        Helper function to evaluate a single full contrastive.
        This helper ask once with multiple choices.
        :param index: Index of the question and answers.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        system_prompt, user_prompt = self.get_prompts(sample_indices, flip)
        guessers = [
            select_model(guesser, system_prompt, self.cm)
            for guesser in self.guesser_model_names
        ]

        if self.cot:
            r1 = [
                guesser(user_prompt, temperature=0, use_json=False)
                for guesser in guessers
            ]
        else:
            for guesser in guessers:
                guesser.add_message(ROLE_USER, user_prompt)
                guesser.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
        r2 = [
            guesser(
                self.rm.get_prompt("eval/contrastive-2R2A/multiple_choice").format(
                    a_name=STUDENT_NAMES[0], b_name=STUDENT_NAMES[1]
                ),
                temperature=0,
                use_json=False,
                max_new_tokens=1,
            )
            for guesser in guessers
        ]
        # print(r2)
        decisions = []
        for raw in r2:
            if raw == "A":
                decisions.append(0)
            elif raw == "B":
                decisions.append(1)
            else:
                decisions.append(-1)
        # majority vote
        decision = max(set(decisions), key=decisions.count)
        # if not flip, we should choose A
        # note that the correct decision is (0, 1), as we already handled flip
        info_dict = {
            "decision": decision,
            "target_decision": int(flip),
            "flip": flip,
            "conversation": [e.messages for e in guessers],
        }
        # if not flip, then choose A (0), otherwise choose B (1)
        correct = decision == int(flip)
        return correct, correct, info_dict, index, flip

    def get_prompts(self, sample_indices: List[int], flip: bool) -> Tuple[str, str]:
        """
        Get the system and user prompts for a single full contrastive.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        # sanity check: questions should be the same
        assert all(
            self.batches[0].get_question(i) == self.batches[1].get_question(i)
            for i in sample_indices
        ), "Questions should be the same!"

        if self.few_shot:
            system_prompt = self.rm.get_prompt(
                "eval/contrastive-2R2A/system-few_shot"
            ).format(topic=self.topic)
        else:
            system_prompt = self.rm.get_prompt("eval/contrastive-2R2A/system").format(
                topic=self.topic
            )

        # construct user prompt
        questions = [self.batches[0].get_question(i) for i in sample_indices]
        if isinstance(self.batches[0].get_true_answer(0), int):
            true_answers = [
                get_choice_str(self.batches[0].get_true_answer(i))
                for i in sample_indices
            ]
        else:
            true_answers = [self.batches[0].get_true_answer(i) for i in sample_indices]

        if self.paraphrased:
            answers0 = [
                self.batches[0].get_paraphrased_reasoning(i) for i in sample_indices
            ]
            answers1 = [
                self.batches[1].get_paraphrased_reasoning(i) for i in sample_indices
            ]
        else:
            answers0 = [self.batches[0].get_model_reasoning(i) for i in sample_indices]
            answers1 = [self.batches[1].get_model_reasoning(i) for i in sample_indices]

        if flip:
            answers0, answers1 = answers1, answers0

        qa = ""
        for i in range(len(questions)):
            qa += f"### Question {i + 1}\n\n{questions[i]}\n"
            if (gt := true_answers[i]) is not None:
                qa += f"Ground Truth Answer: {gt}\n\n"
            answer0_str = f"#### The First Response\n\n{answers0[i]}"
            answer1_str = f"#### The Second Response\n\n{answers1[i]}"
            qa += f"{answer0_str}\n\n{answer1_str}\n\n"

        card_a, card_b = self.cards
        # for debugging
        # card_a = "This student always give The First Response!"
        # card_b = "This student always give The Second Response!"
        # if flip:
        #     card_a, card_b = card_b, card_a

        if self.few_shot:
            user_prompt = self.rm.get_prompt(
                "eval/contrastive-2R2A/user-few_shot"
            ).format(
                card_a=card_a,
                card_b=card_b,
                qa=qa,
                a_name=STUDENT_NAMES[0],
                b_name=STUDENT_NAMES[1],
            )
        else:
            user_prompt = self.rm.get_prompt("eval/contrastive-2R2A/user").format(
                card_a=card_a,
                card_b=card_b,
                qa=qa,
                a_name=STUDENT_NAMES[0],
                b_name=STUDENT_NAMES[1],
            )
        return system_prompt, user_prompt

    def shutdown(self):
        self.rm.shutdown()

