import itertools
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from core.config import HF_API_TOKENS

from core.card import GenerativeCard
from core.data import Batch, load_batches
from core.models import CostManager, select_model, ROLE_USER, ROLE_ASSISTANT
from core.utils import ResourceManager, sample_few_shots, get_choice_str, append_to_csv

STUDENT_NAMES = ["Yarix", "Vionelle"]


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


def get_decision(logits: Dict, pairs: List[str]):
    """
    Get the decision from the logits.
    """
    decision = []
    # print(logits)
    for pair in pairs:
        decision.append(dict_key_helper(logits, pair)[1])

    # softmax decision
    decision = np.exp(decision) / np.sum(np.exp(decision))

    return decision[0] > decision[1], decision


class ContrastiveFullEvaluator:
    """
    Full Contrastive Evaluator:
    Given two cards and two answers, guess which model writes which answer.
    """

    topic: str
    batches: Tuple[Batch, Batch]
    models: Tuple[str, str]
    cards: Tuple[str, str]
    evaluator_model_names: List[str]
    rm: ResourceManager
    cm: Optional[CostManager]
    cot: bool
    k_shots: int
    max_workers: int
    csv_path: Optional[str]
    few_shot: bool

    def __init__(
        self,
        meta: str,
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
        self.meta = meta
        self.topic = topic
        self.batches = batches
        self.models = models
        self.cards = cards
        self.evaluator_model_names = evaluator_model_names
        self.rm = rm
        self.cm = cm
        self.cot = cot
        self.k_shots = k_shots
        self.max_workers = max_workers
        self.csv_path = csv_path
        self.few_shot = few_shot
        self.paraphrase = paraphrase

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
            "method": "contrastive-answer",
            "topic": self.topic,
            "models": self.models,
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
        name = f"eval_contrastive-full_{'_'.join(self.models)}"
        print(f"{name}\nMetrics: {metrics}\n")
        print(f"Accuracy: {mean[0]} ± {sd[0]}")
        print(f"Model 1 accuracy: {mean[1]} ± {sd[1]}")
        print(f"Model 2 accuracy: {mean[2]} ± {sd[2]}")
        print(f"Average Decision Count: {counter}")
        if self.csv_path is not None:  # save to csv
            row_data = [
                [
                    self.models[0],
                    self.models[1],
                    m[0],
                    m[1],
                    m[2],
                    self.evaluator_model_names[0],
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
        self.rm.dump_dict(name, info_dict)
        hyperparameters = {
            "meta": self.meta,
            "topic": self.topic,
            "models": self.models,
            "guessers": self.evaluator_model_names,
            "cot": self.cot,
            "few_shot": self.few_shot,
            "num_times": num_times,
            "num_samples": num_samples,
            "max_workers": max_workers,
            "mode": mode,
        }
        self.rm.dump_dict(f"hyperparameters", hyperparameters)
        return metrics.tolist()

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
        system_prompt, user_prompt = self.get_prompts(sample_indices, flip)
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]

        if self.cot:
            r1 = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]

        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )

        s0, s1 = STUDENT_NAMES
        # generate a random number to decide which one to ask
        target = random.randint(0, 1)
        if target == 0:
            r2 = [
                evaluator.get_logits(
                    f"Who ({s0} or {s1}) wrote all of the First Responses? Give a one word answer.",
                )
                for evaluator in evaluators
            ]
        else:
            r2 = [
                evaluator.get_logits(
                    f"Who ({s0} or {s1}) wrote all of the Second Responses? Give a one word answer.",
                )
                for evaluator in evaluators
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
            "conversation": [e.messages for e in evaluators],
        }
        # we assume the other one is the opposite, so if this is correct, the other is also correct
        correct = decision == target
        return correct, correct, info_dict, index, flip

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
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]

        for e in evaluators:
            e.api_key_index = index % len(HF_API_TOKENS)

        if self.cot:
            extractor_name = 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
            # extractor_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            extractor = [
                select_model(extractor_name, 'You are a TA, you conclude the most likely student name given another TAs analysis.', self.cm)
                for _ in range(len(evaluators))
            ]

            r1 = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]

            target = 0

            for i in range(len(evaluators)):
                ext = extractor[i]
                ext.add_message(ROLE_USER, r1[i])
                ext.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
                ext.api_key_index = index % len(HF_API_TOKENS)

            extraction_prompt = f"Based on the other TA's analysis above, who ({STUDENT_NAMES[0]} or {STUDENT_NAMES[1]}) wrote all of the FIRST Responses? Give a one word answer."
            
            r2 = [
                extractor[i]( extraction_prompt,    
                        temperature=0,
                        use_json=False,
                        max_new_tokens=10,) for i in range(len(evaluators))
            ]

            decisions = []
            s0, s1 = STUDENT_NAMES
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


        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )

            s0, s1 = STUDENT_NAMES
            # generate a random number to decide which one to ask
            target = random.randint(0, 1)
            if target == 0:
                r2 = [
                    evaluator(
                        f"Who ({s0} or {s1}) authors all The First Responses? Give a one word answer.",
                        temperature=0,
                        use_json=False,
                        max_new_tokens=10,
                    )
                    for evaluator in evaluators
                ]
            else:
                r2 = [
                    evaluator(
                        f"Who ({s0} or {s1}) authors all The Second Responses? Give a one word answer.",
                        temperature=0,
                        use_json=False,
                        max_new_tokens=10,
                    )
                    for evaluator in evaluators
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
            "conversation": [e.messages for e in evaluators],
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
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]

        if self.cot:
            r1 = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]
        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
        s0, s1 = STUDENT_NAMES
        r2 = [
            evaluator(
                f"Who ({s0} or {s1}) authors all The First Responses? Give a one word answer.",
                temperature=0,
                use_json=False,
            )
            for evaluator in evaluators
        ]
        for evaluator in evaluators:  # remove the last 2 conversations
            evaluator.messages.pop()
            evaluator.messages.pop()
        r3 = [
            evaluator(
                f"Who ({s0} or {s1}) authors all The Second Responses? Give a one word answer.",
                temperature=0,
                use_json=False,
            )
            for evaluator in evaluators
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
            "conversation": [e.messages for e in evaluators],
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
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]

        if self.cot:
            r1 = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]
        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which student authors which answer!",
                )
        r2 = [
            evaluator(
                self.rm.get_prompt("eval/contrastive-full/multiple-choice").format(
                    a_name=STUDENT_NAMES[0], b_name=STUDENT_NAMES[1]
                ),
                temperature=0,
                use_json=False,
                max_new_tokens=1,
            )
            for evaluator in evaluators
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
            "conversation": [e.messages for e in evaluators],
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
        )

        if self.few_shot:
            system_prompt = self.rm.get_prompt(
                "eval/contrastive-full/system-few_shot"
            ).format(topic=self.topic)
        else:
            system_prompt = self.rm.get_prompt("eval/contrastive-full/system").format(
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

        if self.paraphrase:
            answers0 = [self.batches[0].get_paraphrased(i) for i in sample_indices]
            answers1 = [self.batches[1].get_paraphrased(i) for i in sample_indices]
        else:
            answers0 = [self.batches[0].get_model_reasoning(i) for i in sample_indices]
            answers1 = [self.batches[1].get_model_reasoning(i) for i in sample_indices]

        if flip:
            answers0, answers1 = answers1, answers0

        qa = ""
        for i in range(len(questions)):
            qa += f"### Question {i + 1}\n\n{questions[i]}\n"
            if (gt := true_answers[i]) is not None:
                if True:
                    qa += f"Ground Truth Answer: {gt}\n\n"
                pass
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
                "eval/contrastive-full/user-few_shot"
            ).format(
                card_a=card_a,
                card_b=card_b,
                qa=qa,
                a_name=STUDENT_NAMES[0],
                b_name=STUDENT_NAMES[1],
            )
        else:
            user_prompt = self.rm.get_prompt("eval/contrastive-full/user").format(
                card_a=card_a,
                card_b=card_b,
                qa=qa,
                a_name=STUDENT_NAMES[0],
                b_name=STUDENT_NAMES[1],
            )
        return system_prompt, user_prompt

    def shutdown(self):
        self.rm.shutdown()


def test(
    models: List[str],
    model_to_dict_card: Dict[str, str],
    meta: str,
    topic: str,
    exp_name: str,
    evaluators: List[str],
    cot: bool,
    mode: Literal["simplified", "full", "partial", "logit"],
    # full matrix or mirror the half matrix
    filling_mode: Literal["full", "mirror"],
    num_samples: int = -1,
    max_workers: int = 60,
    model_pairs: List[Tuple[str, str]] = None,
):
    model_to_dict_card = {
        model: GenerativeCard(path) for model, path in model_to_dict_card.items()
    }
    cm = CostManager()

    data = {}
    avg_accuracy, total = 0.0, 0
    dataset_acc = {model: float("nan") for model in models}

    if model_pairs is None:
        model_pairs = []
        if filling_mode == "full":
            model_pairs = itertools.product(models, models)
            # remove the repeated models
            model_pairs = [
                (model0, model1) for model0, model1 in model_pairs if model0 != model1
            ]
        else:
            for i in range(0, len(models)):
                for j in range(i + 1, len(models)):
                    model_pairs.append((models[i], models[j]))

    for model0, model1 in tqdm(model_pairs):
        batch0 = load_batches(f"datasets/{meta}/", topic, model0, "test", [60], False)[
            0
        ]
        batch1 = load_batches(f"datasets/{meta}/", topic, model1, "test", [60], False)[
            0
        ]
        dataset_acc[model0] = batch0.get_accuracy()
        dataset_acc[model1] = batch1.get_accuracy()
        card0 = str(model_to_dict_card[model0])
        card1 = str(model_to_dict_card[model1])

        eval_method = ContrastiveFullEvaluator(
            meta,
            topic,
            (batch0, batch1),
            (model0, model1),
            (card0, card1),
            evaluators,
            ResourceManager(exp_name=exp_name, name=f"{model0}-{model1}"),
            cm,
            cot=cot,
            k_shots=3,
            max_workers=max_workers,
            csv_path=f"exp_results/{exp_name}.csv",
        )
        metrics = eval_method.main(num_samples=num_samples, mode=mode)[0]
        eval_method.shutdown()
        accuracy = metrics[0]

        data[(model0, model1)] = accuracy
        data[(model0, model0)] = 0.5  # diagonal
        data[(model1, model1)] = 0.5
        avg_accuracy += accuracy
        total += 1
    avg_accuracy /= total

    plot_heatmap(topic, models, data, dataset_acc, avg_accuracy, exp_name)

    print(data)
    print(cm.get_info_dict())
    return data


def plot_heatmap(topic, models, data, dataset_acc, avg_accuracy, exp_name):
    """
    Plot the heatmap of the full contrastive evaluation.
    :param topic: Topic name.
    :param models: List of model names.
    :param data: Dictionary of model pairs to accuracy.
    :param dataset_acc: Dictionary of model to dataset accuracy.
    :param avg_accuracy: Average accuracy.
    :param exp_name: Experiment name.
    """
    # Format model names with their dataset accuracy
    model_names_with_acc = {
        model: f"{model}\n({dataset_acc[model]:.2f})" for model in models
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["index", "value"])
    df[["str1", "str2"]] = pd.DataFrame(
        [
            [model_names_with_acc[model] for model in pair]
            for pair in df["index"].tolist()
        ],
        index=df.index,
    )
    df.drop(columns=["index"], inplace=True)

    # Calculate differences and prepare formatted annotations
    def format_annotations(row):
        model1 = row["str1"].split("\n")[0]
        model2 = row["str2"].split("\n")[0]
        difference = abs(dataset_acc[model1] - dataset_acc[model2])
        return f"{row['value']:.2f}\n({difference:.2f})"

    df["formatted_value"] = df.apply(format_annotations, axis=1)

    # Pivot for color mapping and annotations
    pivot_values = df.pivot(index="str2", columns="str1", values="value")
    pivot_annotations = df.pivot(index="str2", columns="str1", values="formatted_value")

    # Reindex to ensure models are in the correct order
    order = [model_names_with_acc[model] for model in models]
    pivot_values = pivot_values.reindex(index=order, columns=order)
    pivot_annotations = pivot_annotations.reindex(index=order, columns=order)

    # Plot the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pivot_values.astype(float),  # Ensure this is float for colormap calculation
        annot=pivot_annotations,  # Use formatted annotations
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",  # Set format to empty since annotations are strings
        annot_kws={"fontsize": 10, "va": "top", "ha": "center"},
    )
    plt.title(
        f"Topic: {topic}\n"
        f"Heatmap of 2C2A Contrastive Evaluation on Generative Cards\n"
        f"Avg Accuracy: {avg_accuracy:.2f}"
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=10, style="italic")
    plt.yticks(rotation=0, ha="right", fontsize=10, style="italic")
    plt.savefig(f"outputs/{exp_name}/heatmap.png")
    # plt.show()


if __name__ == "__main__":
    # {
    #    "gemma-1.1-7b-it": "",
    #    "gpt-3.5-turbo": "",
    #    "gpt-4-turbo": "",
    #    "gpt-4o": "",
    #    "Meta-Llama-3-8B-Instruct": "",
    #    "Meta-Llama-3-70B-Instruct": "",
    #    "Mistral-7B-Instruct-v0.2": "",
    #    "Mixtral-8x7B-Instruct-v0.1": "",
    # }
    epoch = 4
    math_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gemma-1.1-7b-it/05-12_19-38-06_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-3.5-turbo/05-12_19-38-06_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-4-turbo/05-12_19-38-06_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4o": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-4o/05-18_11-03-15_gpt-4o_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-12_19-38-06_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-12_19-38-06_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-12_19-38-06_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-12_19-33-08_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
    }

    chemistry_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_22-53-43_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_22-53-43_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-4-turbo/05-13_22-53-43_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4o": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-4o/05-18_11-08-10_gpt-4o_main/cards/epoch_4_card.json",
        "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_22-53-43_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_22-53-43_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-13_22-53-43_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_22-53-43_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
    }
    physics_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_12-37-20_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_12-37-20_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-4-turbo/05-13_12-37-20_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4o": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-4o/05-18_11-05-33_gpt-4o_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_12-37-20_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_12-37-20_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-13_12-24-57_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_12-24-57_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
    }
    history_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_22-56-47_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/gpt-3.5-turbo/05-18_21-03-45_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/gpt-4-turbo/05-18_21-03-45_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4o": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/gpt-4o/05-18_18-54-38_gpt-4o_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_22-56-47_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_22-56-47_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-18_21-03-45_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_world_history/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-18_21-03-45_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
    }
    openend_code_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_00-38-08_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_00-38-08_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gpt-4-turbo/05-13_00-38-08_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-8B-Instruct": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_00-38-08_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_00-38-08_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-12_23-39-56_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": "outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-12_23-39-56_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
    }
    openend_dating_model_to_dict_card = {
        "gemma-1.1-7b-it": "",
        "gpt-3.5-turbo": "",
        "gpt-4-turbo": "",
        "Meta-Llama-3-8B-Instruct": "",
        "Meta-Llama-3-70B-Instruct": "",
        "Mistral-7B-Instruct-v0.2": "",
        "Mixtral-8x7B-Instruct-v0.1": "",
    }

    # meta = "mmlu"
    meta = "openend"

    # topic = "high_school_chemistry"
    # topic = "high_school_physics"
    # topic = "high_school_mathematics"

    topic = "Writing efficient code for solving concrete algorthimic problems"

    test(
        models=[
            "gemma-1.1-7b-it",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            # "gpt-4o",
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
            "Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B-Instruct-v0.1",
        ],
        model_to_dict_card=openend_code_model_to_dict_card,
        meta=meta,
        topic=topic,
        exp_name=f"eval/{topic}/contrastive-partial-cot-0",
        evaluators=["meta-llama/Meta-Llama-3-70B-Instruct"],
        # evaluators=["gpt-4o"],
        cot=True,
        mode="partial",
        filling_mode="full",
    )
