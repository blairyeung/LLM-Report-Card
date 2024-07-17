import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from core.card import GenerativeCard
from core.data import Batch, load_batches
from core.models import CostManager, select_model, ROLE_USER, ROLE_ASSISTANT
from core.utils import ResourceManager, get_choice_str, sample_few_shots, append_to_csv

STUDENT_NAMES = ["Yarix", "Vionelle"]


class ContrastiveAnswerEvaluator:
    """
    Contrastive Answer Evaluator:
    Given a card, guess which one of the two answers is authored by the model described by the card.
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

    def main(
        self,
        num_times: int = 1,
        num_samples: int = -1,
        max_workers: int = -1,
        use_logits: bool = True,
        mode: Literal["normal", "logit"] = "normal",
    ) -> List[List[float]]:
        """
        The main function to run the contrastive-answer evaluation.
        :param num_times: Number of times to run the evaluation.
        :param num_samples: Number of samples to evaluate.
                    If -1, then use len(self.batches[0]).
        :param max_workers: Maximum number of parallel workers to use.
        :param use_logits: Whether to use logits for evaluation.
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
        metrics = []
        for i in range(num_times):
            sub_info_dict = {}
            info_dict["iterations"].append(sub_info_dict)

            samples = sample_few_shots(num_samples, len(self.batches[0]), self.k_shots)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index in range(num_samples):
                    sample = samples[index]

                    if mode == "normal":
                        f = self.helper
                    elif mode == "logit":
                        f = self.logit_helper
                    else:
                        raise NotImplementedError

                    futures.append(
                        executor.submit(
                            f,
                            index,
                            self.cards[0],
                            0,
                            sample,
                            bool(random.randint(0, 1)),
                        )
                    )
                    futures.append(
                        executor.submit(
                            f,
                            index,
                            self.cards[1],
                            1,
                            sample,
                            bool(random.randint(0, 1)),
                        )
                    )

                total = 0
                card1_correct = 0
                card2_correct = 0
                card_1_total = 0
                card_2_total = 0
                for future in tqdm(
                    as_completed(futures), desc=f"Evaluating...", total=len(futures)
                ):
                    if future.exception() is not None:
                        print(future.exception(), file=sys.stderr)
                        continue
                    correct, d, index, target_model = future.result()
                    sub_info_dict[str(index)] = d
                    if target_model == 0:
                        card_1_total += 1
                        card1_correct += int(correct)
                    else:
                        card_2_total += 1
                        card2_correct += int(correct)
                    total += 1

            model1_acc = card1_correct / card_1_total
            model2_acc = card2_correct / card_2_total
            metrics.append(
                [(card1_correct + card2_correct) / total, model1_acc, model2_acc]
            )
        metrics = np.asarray(metrics).reshape(-1, 3)
        mean = np.mean(metrics, axis=0)
        sd = np.std(metrics, axis=0)
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
        }
        name = f"eval_contrastive-answer_{'_'.join(self.models)}"
        print(f"{name}\nMetrics: {metrics}\n")
        print(f"Accuracy: {mean[0]} ± {sd[0]}")
        print(f"Model 1 accuracy: {mean[1]} ± {sd[1]}")
        print(f"Model 2 accuracy: {mean[2]} ± {sd[2]}")
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
        return metrics.tolist()

    def logit_helper(
        self,
        index: int,
        card: str,
        target_model: int,
        sample_indices: List[int],
        flip: bool,
    ) -> Tuple[bool, Dict, int, int]:
        """
        Helper function to evaluate a single contrastive answer.
        :param index: Index of the question and answers.
        :param card: Card description.
        :param target_model: The target model. Should produce a1 or a2.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        assert target_model in [0, 1], "target_model should be 0 or 1."
        system_prompt, user_prompt = self.get_prompts(
            card, sample_indices, flip, target_model
        )
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]
        if self.cot:
            r = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]
        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which answer is from the student described by the Evaluation Card.",
                )
        s0, s1 = STUDENT_NAMES
        r2 = [
            evaluator.get_logits(
                prompt=f"Who ({s0} or {s1}) is described by the Evaluation Card? Give an one word answer.",
                post_fix="",
            )
            for evaluator in evaluators
        ]
        s0, s1 = s0.lower(), s1.lower()
        decisions = []
        for raw_decision in r2:
            # bob -> 0, Claire -> 1
            # make all keys lower, if clash, only keep the highest, and remove space
            modified_decisions = {}
            bob_max_logit = -100
            claire_max_logit = -100
            # print(f"{raw_decision=}")
            for key, value in raw_decision.items():
                key = key.lower().replace(" ", "")
                if key.lower() in s0:
                    if value > bob_max_logit:
                        bob_max_logit = value
                        modified_decisions[s0] = value

                if key.lower() in s1:
                    if value > claire_max_logit:
                        claire_max_logit = value
                        modified_decisions[s1] = value

            # print(f'{modified_decisions=}')

            decision = 0 if bob_max_logit > claire_max_logit else 1
            decisions.append(decision)

        # majority vote
        decision = max(set(decisions), key=decisions.count)
        if flip and decision != -1:
            decision = 1 - decision

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "flip": flip,
            "conversation": [e.messages for e in evaluators],
        }
        return decision == target_model, info_dict, index, target_model

    def helper(
        self,
        index: int,
        card: str,
        target_model: int,
        sample_indicis: List[int],
        flip: bool,
    ) -> Tuple[bool, Dict, int, int]:
        """
        Helper function to evaluate a single contrastive answer.
        :param index: Index of the question and answers.
        :param card: Card description.
        :param target_model: The target model. Should produce a1 or a2.
        :param sample_indicis: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        """
        assert target_model in [0, 1]
        system_prompt, user_prompt = self.get_prompts(
            card, sample_indicis, flip, target_model
        )
        evaluators = [
            select_model(evaluator, system_prompt, self.cm)
            for evaluator in self.evaluator_model_names
        ]
        if self.cot:
            r = [
                evaluator(user_prompt, temperature=0, use_json=False)
                for evaluator in evaluators
            ]
        else:
            for evaluator in evaluators:
                evaluator.add_message(ROLE_USER, user_prompt)
                evaluator.add_message(
                    ROLE_ASSISTANT,
                    "I'm confident that I know which answer is from the student described by the Evaluation Card.",
                )
        s0, s1 = STUDENT_NAMES
        r2 = [
            evaluator(
                f"Who ({s0} or {s1}) is described by the Evaluation Card? Give a one word answer.",
                temperature=0,
                use_json=False,
            )
            for evaluator in evaluators
        ]
        s0, s1 = s0.lower(), s1.lower()
        decisions = []
        for raw_decision in r2:
            raw_decision = raw_decision.lower()
            if s0 in raw_decision and s1 in raw_decision:
                decisions.append(-1)
            elif s0 in raw_decision:
                decisions.append(0)
            elif s1 in raw_decision:
                decisions.append(1)
            else:
                decisions.append(-1)

        # majority vote
        decision = max(set(decisions), key=decisions.count)
        if flip and decision != -1:
            decision = 1 - decision

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "flip": flip,
            "conversation": [e.messages for e in evaluators],
        }
        return decision == target_model, info_dict, index, target_model

    def get_prompts(
        self,
        card: str,
        sample_indices: List[int],
        flip: bool,
        target_model: int,
    ) -> Tuple[str, str]:
        """
        Get the system and user prompts for the contrastive answer evaluation.
        :param card: Card description.
        :param sample_indices: List of indices to sample from the batch.
        :param flip: Whether to flip the target model.
        :param target_model: The target model. For debugging use.
        """
        # sanity check: questions should be the same
        assert all(
            self.batches[0].get_question(i) == self.batches[1].get_question(i)
            for i in sample_indices
        ), "The order of questions from two batches should be the same."

        system_prompt = self.rm.get_prompt("eval/contrastive-answer/system").format(
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
        answers0 = [self.batches[0].get_model_reasoning(i) for i in sample_indices]
        answers1 = [self.batches[1].get_model_reasoning(i) for i in sample_indices]

        if flip:
            answers0, answers1 = answers1, answers0

        qa = ""
        for i in range(len(questions)):
            qa += f"### Question {i + 1}\n\n{questions[i]}\n"
            if (gt := true_answers[i]) is not None:
                qa += f"Ground Truth Answer: {gt}\n\n"
            answer0_str = f"### {STUDENT_NAMES[0]}'s Response\n\n{answers0[i]}"
            answer1_str = f"### {STUDENT_NAMES[1]}'s Response\n\n{answers1[i]}"
            qa += f"{answer0_str}\n\n{answer1_str}\n\n"

        # for debugging
        # if flip:
        #     card = f"This evaluation card describes student {STUDENT_NAMES[1 - target_model]}!"
        # else:
        #     card = (
        #         f"This evaluation card describes student {STUDENT_NAMES[target_model]}!"
        #     )

        user_prompt = self.rm.get_prompt("eval/contrastive-answer/user").format(
            card=card, qa=qa, a_name=STUDENT_NAMES[0], b_name=STUDENT_NAMES[1]
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
    mode: Literal["normal", "logit"],
):
    model_to_dict_card = {
        model: GenerativeCard(path) for model, path in model_to_dict_card.items()
    }
    cm = CostManager()

    data = {}
    avg_accuracy, total = 0.0, 0
    dataset_acc = {model: float("nan") for model in models}

    model_pairs = []
    for i in range(0, len(models)):
        for j in range(i + 1, len(models)):
            model_pairs.append((models[i], models[j]))
    for model0, model1 in model_pairs:
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

        eval_method = ContrastiveAnswerEvaluator(
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
            max_workers=80,
            csv_path=f"exp_results/{exp_name}.csv",
        )
        metrics = eval_method.main(mode=mode)[0]
        eval_method.shutdown()
        accuracy = metrics[0]

        data[(model0, model1)] = accuracy
        data[(model0, model0)] = 0.5  # diagonal
        data[(model1, model1)] = 0.5
        data[(model1, model0)] = accuracy  # mirror
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
        f"Heatmap of 1C2A Contrastive Evaluation on Generative Cards\n"
        f"Avg Accuracy: {avg_accuracy:.2f}"
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=10, style="italic")
    plt.yticks(rotation=0, ha="right", fontsize=10, style="italic")
    plt.savefig(f"outputs/{exp_name}/heatmap.png")
    plt.show()


if __name__ == "__main__":
    # {
    #    "gemma-1.1-7b-it": "",
    #    "gpt-3.5-turbo": "",
    #    "gpt-4-turbo": "",
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
    openend_code_model_to_dict_card = {
        "gemma-1.1-7b-it": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_00-38-08_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
        "gpt-3.5-turbo": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_00-38-08_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
        "gpt-4-turbo": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/gpt-4-turbo/05-13_00-38-08_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-8B-Instruct": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_00-38-08_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Meta-Llama-3-70B-Instruct": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_00-38-08_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
        "Mistral-7B-Instruct-v0.2": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-12_23-39-56_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
        "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/Writing efficient code for solving concrete algorthimic problems/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-12_23-39-56_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
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

    meta = "mmlu"
    # meta = "openend"

    # topic = "high_school_chemistry"
    # topic = "high_school_physics"
    topic = "high_school_mathematics"

    # topic = "Writing efficient code for solving concrete algorthimic problems"

    test(
        models=[
            "gemma-1.1-7b-it",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4o",
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
            "Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B-Instruct-v0.1",
        ],
        model_to_dict_card=math_model_to_dict_card,
        meta=meta,
        topic=topic,
        exp_name=f"eval/{topic}/contrastive-answer-no_cot-2",
        evaluators=["meta-llama/Meta-Llama-3-70B-Instruct"],
        # evaluators=["gpt-4o"],
        cot=False,
        mode="normal",
    )
