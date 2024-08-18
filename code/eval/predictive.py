import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Dict, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.card import GenerativeCard
from core.models import CostManager, select_model
from core.utils import ResourceManager
from dataset.data import Batch, load_batches

# DEFAULT_EVALUATOR = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
DEFAULT_EVALUATOR = "gpt-4o-mini"

# DEFAULT_EVALUATOR = "meta-llama/Meta-Llama-3-70B-Instruct"


class PredictiveEvaluator:
    type_: str
    topic: str
    model: str
    rm: ResourceManager
    cm: CostManager
    guesser_name: str

    def __init__(
        self,
        topic: str,
        model: str,
        rm: ResourceManager,
        type_: str,
        evaluator_name: str = DEFAULT_EVALUATOR,
        cm: CostManager = None,
    ):
        self.type_ = type_
        # print(f'PredictiveEvaluator: {type_}')

        assert type_ in [
            "mmlu",
            "anthropic",
            "few_shot",
            "gsm8k",
            "hotpot_qa",
            "openend",
        ]

        self.topic = topic
        self.model = model
        self.rm = rm
        self.cm = cm if cm is not None else CostManager()
        self.guesser_name = evaluator_name

    def helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive/system-{self.type_}"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt, self.cm)
        user_prompt_path = "eval/predictive/user"
        if self.type_ == "few_shot":
            user_prompt_path += "-few_shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), qa=batch.get_eval_predictive_str(index)
        )
        try:
            json_obj = model(
                user_prompt, use_json=True, timeout=30, temperature=1.0, cache=False
            )
            if isinstance(json_obj, Dict):
                if "prediction" in json_obj:
                    actual = bool(json_obj["prediction"])
                    # confidence = int(json_obj.get("confidence", float("nan")))
                    confidence = 0
                else:
                    raise ValueError(f"Response not parsable: {json_obj}")
            elif isinstance(json_obj, str):  # failed to parse a JSON
                raise ValueError(f"Response not parsable: {json_obj}")
            else:
                assert False
            expected = batch.get_true_answer(index) == batch.get_model_answer(index)
            return actual, expected, confidence, model, index
        except Exception as e:
            print(e, file=sys.stderr)
            return None

    def logit_helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive/system_{self.type_}"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt, self.cm)
        user_prompt_path = "eval/predictive/user"
        if self.type_ == "few-shot":
            user_prompt_path += "_few-shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), qa=batch.get_eval_predictive_str(index)
        )
        try:
            if "llama" in model.name.lower():
                rslt = model.get_logits(user_prompt, post_fix='{"prediction": ')

                true_logit = rslt[" true"]
                false_logit = rslt[" false"]

                true_prob = 1 / (1 + np.exp(-true_logit))
                false_prob = 1 / (1 + np.exp(-false_logit))

                actual = true_logit > false_logit
                expected = batch.get_true_answer(index) == batch.get_model_answer(index)

                confidence = max(true_prob, false_prob)

                return actual, expected, confidence, model, index
            else:
                json_obj = model(user_prompt, use_json=True, timeout=30)
                if isinstance(json_obj, Dict):
                    if "prediction" in json_obj:
                        actual = bool(json_obj["prediction"])
                        confidence = int(json_obj.get("confidence", float("nan")))
                    else:
                        raise ValueError(f"Response not parsable: {json_obj}")
                elif isinstance(json_obj, str):  # failed to parse a JSON
                    raise ValueError(f"Response not parsable: {json_obj}")
                else:
                    assert False
                expected = batch.get_true_answer(index) == batch.get_model_answer(index)
                return actual, expected, confidence, model, index
        except Exception as e:
            print(e, file=sys.stderr)
            return None

    def eval(
        self, batch: Batch, card: Union[GenerativeCard, str], max_workers: int = 60
    ):
        info_dict = {"details": {}}
        count = 0
        TP_indices, TN_indices, FP_indices, FN_indices = [], [], [], []
        confidences = [float("nan")] * len(batch)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.helper, batch, i, card) for i in range(len(batch))
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                if future.result() is None:
                    continue
                actual, expected, confidence, model, index = future.result()
                confidences[index] = confidence / 5  # normalize
                count += 1
                if actual == expected:
                    if actual:
                        TP_indices.append(index)
                    else:
                        TN_indices.append(index)
                else:
                    if expected:
                        FN_indices.append(index)
                    else:
                        FP_indices.append(index)

                info_dict["details"][str(index)] = {
                    "actual": actual,
                    "expected": expected,
                    "confidence": confidence,
                    "conversation": model.messages,
                }

        TP, TN, FP, FN = (
            len(TP_indices),
            len(TN_indices),
            len(FP_indices),
            len(FN_indices),
        )
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP) if TN + FP > 0 else float("nan")
        sensitivity = TP / (TP + FN) if TP + FN > 0 else float("nan")
        precision = TP / (TP + FP) if TP + FP > 0 else float("nan")
        mean_confidence = np.nanmean(confidences)
        info_dict["metrics"] = {
            "accuracy": accuracy,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "precision": precision,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "total": TP + TN + FP + FN,
            "confidences": confidences,
            "mean_confidence": mean_confidence,
            "sd_confidence": np.nanstd(confidences),
        }
        return (
            info_dict,
            (accuracy, specificity, sensitivity, precision, mean_confidence),
            (TP_indices, TN_indices, FP_indices, FN_indices),
        )

    def main(
        self,
        name: str,
        batch: Batch,
        card: Union[GenerativeCard, str],
        num_times: int = 3,
    ):
        info_dict = {
            "method": "predictive",
            "topic": self.topic,
            "model": self.model,
            "evaluator": self.guesser_name,
            "iterations": [],
        }
        metrics = []
        indices = []
        for j in range(num_times):
            sub_info, stats, sub_indices = self.eval(batch, card)
            info_dict["iterations"].append(sub_info)
            metrics.append(stats)
            indices.append(sub_indices)

        accuracies = [m[0] for m in metrics]
        specificities = [m[1] for m in metrics]
        sensitivities = [m[2] for m in metrics]
        precisions = [m[3] for m in metrics]
        confidences = [m[4] for m in metrics]
        info_dict["metrics"] = {
            "accuracies": accuracies,
            "specificities": specificities,
            "sensitivities": sensitivities,
            "precisions": precisions,
            "confidences": confidences,
            "mean_accuracy": np.mean(accuracies),
            "mean_specificity": np.mean(specificities),
            "mean_sensitivity": np.mean(sensitivities),
            "mean_precision": np.mean(precisions),
            "mean_confidence": np.mean(confidences),
            "sd_accuracy": np.std(accuracies),
            "sd_specificity": np.std(specificities),
            "sd_sensitivity": np.std(sensitivities),
            "sd_precision": np.std(precisions),
            "sd_confidence": np.std(confidences),
            "classification_indices": indices,
        }
        print(
            f"{name} | Dataset Accuracy: {batch.get_accuracy()}\n"
            f"accuracies: {accuracies}, mean={np.mean(accuracies)}, std={np.std(accuracies)}\n"
            f"specificities: {specificities}, mean={np.mean(specificities)}, std={np.std(specificities)}\n"
            f"sensitivities: {sensitivities}, mean={np.mean(sensitivities)}, std={np.std(sensitivities)}\n"
            f"precisions: {precisions}, mean={np.mean(precisions)}, std={np.std(precisions)}\n"
            f"confidences: {confidences}, mean={np.mean(confidences)}, std={np.std(confidences)}\n"
            f'cost so far: {self.cm.get_cost() if self.cm is not None else "N/A"}\n'
        )
        self.rm.dump_dict(name, info_dict)
        return info_dict

    def shutdown(self):
        self.rm.shutdown()


def plot_epochs(
    topic: str,
    model: str,
    eval_log_filenames: List[str],
    testing_batch: Batch,
    training_batches: List[Batch],
    rm: ResourceManager,
    plot_save_path: str = None,
):
    """
    Plot the predictive evaluation metrics over epochs.
    :param topic: Topic
    :param model: Model
    :param eval_log_filenames: List of filenames for the predictive evaluation logs, relative to the output folder
    :param num_epochs: Number of epochs
    :param testing_batch: Testing batch
    :param training_batches: List of training batches
    :param rm: ResourceManager
    :param plot_save_path: Path to save the plot, relative to the output folder
    """
    num_epochs = len(training_batches)
    batch_accuracy = testing_batch.get_accuracy()
    epoch_accuracies = [batch.get_accuracy() for batch in training_batches]
    metrics = [
        "accuracies",
        "specificities",
        "sensitivities",
        "precisions",
        "confidences",
    ]
    colors = ["blue", "green", "red", "orange", "purple"]

    # Load and process data
    data = {metric: [] for metric in metrics}
    for filename in eval_log_filenames:
        with open(os.path.join(rm.output_folder_path, filename)) as f:
            json_obj = json.load(f)["metrics"]
        for metric in metrics:
            data[metric].append(json_obj[metric])

    # Calculate means and standard deviations
    mean_data = {
        metric: [np.mean(epoch_data) for epoch_data in data[metric]]
        for metric in metrics
    }
    std_data = {
        metric: [np.std(epoch_data) for epoch_data in data[metric]]
        for metric in metrics
    }

    # Plotting
    plt.figure(figsize=(13, 7))
    epochs = list(range(num_epochs))

    for metric, color in zip(metrics, colors):
        plt.plot(epochs, mean_data[metric], label=metric.capitalize(), color=color)
        plt.fill_between(
            epochs,
            np.array(mean_data[metric]) - np.array(std_data[metric]),
            np.array(mean_data[metric]) + np.array(std_data[metric]),
            color=color,
            alpha=0.2,
        )
        for e, value in enumerate(mean_data[metric]):
            plt.text(
                epochs[e], value, f"{value:.2f}", ha="center", va="bottom", color=color
            )

    # Additional plots
    plt.plot(
        epochs,
        epoch_accuracies,
        label="Train Batch Accuracy",
        color="cyan",
        linestyle="dashed",
    )
    oracle = max(batch_accuracy, 1 - batch_accuracy)
    plt.plot(
        epochs,
        [oracle] * num_epochs,
        label="Oracle Accuracy",
        color="gray",
        linestyle="dashed",
    )

    # Formatting
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("Predictive Evaluation Metrics Over Epochs")
    plt.suptitle(
        f"Topic: {topic}  Target Model: {model}  Batch Acc: {batch_accuracy}  Oracle: {oracle:.2f}"
    )
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xticks(range(num_epochs))

    # Save the plot
    if plot_save_path is not None:
        plot_save_path = os.path.join(rm.output_folder_path, plot_save_path)
    else:
        plot_save_path = os.path.join(rm.output_folder_path, "plots/predictive_metrics_over_epochs.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    # plt.show()


if __name__ == "__main__":
    rm = ResourceManager("press-debug", name="run-3")
    topic = "high_school_mathematics"
    model = "Meta-Llama-3-8B-Instruct"
    pe = PredictiveEvaluator(topic, model, rm, "mmlu")
    test_batch = load_batches("datasets/mmlu", topic, model, "test", [60])[0]
    card = GenerativeCard(filename="cards/mmlu/high_school_mathematics/press/bullet_point/gpt-4o-2024-05-13/Meta-Llama-3-8B-Instruct/epoch_4_card.json")
    pe.main("eval/predictive_eval-epoch_4-previous_card", test_batch, card)
