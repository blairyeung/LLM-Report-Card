import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.card import GenerativeCard
from core.data import Batch, get_choice_int, get_choice_str
from core.models import select_model, CostManager
from core.utils import ResourceManager

DEFAULT_GUESSER = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
# DEFAULT_GUESSER = GPT_3_MODEL_NAME
DEFAULT_EXTRACTOR = "mistralai/Mistral-7B-Instruct-v0.2"


class PredictiveEvaluatorMultiRound:
    type_: str
    topic: str
    model: str
    rm: ResourceManager
    tm: CostManager
    guesser_name: str
    answer_extractor_name: str

    def __init__(
        self,
        topic: str,
        model: str,
        rm: ResourceManager,
        type_: str,
        guesser: str = DEFAULT_GUESSER,
        answer_extractor_name: str = DEFAULT_EXTRACTOR,
        cm: CostManager = None,
    ):
        self.type_ = type_
        self.type_ = "mmlu"
        # assert type_ in ['mmlu', 'anthropic', 'few-shot']
        assert self.type_ == "mmlu"  # TODO: implement for others
        self.topic = topic
        self.model = model
        self.rm = rm
        self.tm = cm
        self.guesser_name = guesser
        self.answer_extractor_name = answer_extractor_name

    def helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt("eval/multi-round/guesser_system").format(
            topic=self.topic
        )
        guesser = select_model(self.guesser_name, system_prompt, self.tm)
        # step 1: answer the question
        user_prompt = self.rm.get_prompt("eval/multi-round/guesser_question").format(
            q=batch.get_question(index)
        )
        r = guesser(user_prompt, cache=True, timeout=30)
        # extract choice
        extractor = select_model(
            self.answer_extractor_name,
            self.rm.get_prompt("eval/multi-round/extractor_system"),
            self.tm,
        )
        user_prompt = self.rm.get_prompt("eval/multi-round/extractor_user").format(
            qa=f"{batch.get_question(index)}\nStudent's Reasoning: {r}"
        )
        json_obj = extractor(user_prompt, use_json=True, cache=True, timeout=20)
        if isinstance(json_obj, dict):
            answer = json_obj["choice"]
        else:
            raise ValueError(
                f"Extractor: Cannot parse the response to a JSON object. Response: {json_obj}"
            )
        if answer[0] in ["A", "B", "C", "D"] and get_choice_int(
            answer[0]
        ) == batch.get_true_answer(
            index
        ):  # correct
            user_prompt = self.rm.get_prompt("eval/multi-round/guesser_correct").format(
                topic=self.topic
            )
        else:
            user_prompt = self.rm.get_prompt(
                "eval/multi-round/guesser_incorrect"
            ).format(
                answer=get_choice_str(batch.get_true_answer(index)), topic=self.topic
            )
        guesser(user_prompt, cache=True, timeout=30)
        # step 2: predict
        user_prompt = self.rm.get_prompt("eval/multi-round/guesser_card").format(
            topic=self.topic, card=str(card)
        )
        guesser(user_prompt, timeout=60)
        user_prompt = self.rm.get_prompt("eval/multi-round/guesser_predict")
        r = guesser(user_prompt, use_json=True, timeout=30)
        if isinstance(r, dict):
            actual = bool(r["I believe the student can correctly answer the question"])
            confidence = r.get("confidence", float("nan"))
        else:
            raise ValueError(
                f"Guesser: Cannot parse the response to a JSON object. Response: {r}"
            )
        expected = batch.get_true_answer(index) == batch.get_model_answer(index)
        return actual, expected, confidence, guesser, extractor, index

    def eval(self, batch: Batch, card: Union[GenerativeCard, str]):
        info_dict = {"details": {}}
        count = 0
        TP_indices, TN_indices, FP_indices, FN_indices = [], [], [], []
        confidences = [float("nan")] * len(batch)
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(self.helper, batch, i, card) for i in range(len(batch))
            ]
            for future in tqdm(as_completed(futures)):
                if future.exception():
                    print(future.exception(), file=sys.stderr)
                    continue
                assert future.result() is not None
                actual, expected, confidence, guesser, extractor, index = (
                    future.result()
                )
                confidences[index] = confidence / 5  # normalize
                # if confidences[index] < 0.5:  # filter out low confidence
                #     continue
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
                    "conversation": guesser.messages,
                    "extractor": extractor.messages,
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
        indicies = []
        for j in range(num_times):
            sub_info, stats, sub_indicies = self.eval(batch, card)
            info_dict["iterations"].append(sub_info)
            metrics.append(stats)
            indicies.append(sub_indicies)

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
            "classification_indices": indicies,
        }
        print(
            f"{name}\n"
            f"accuracies: {accuracies}, mean={np.mean(accuracies)}, std={np.std(accuracies)}\n"
            f"specificities: {specificities}, mean={np.mean(specificities)}, std={np.std(specificities)}\n"
            f"sensitivities: {sensitivities}, mean={np.mean(sensitivities)}, std={np.std(sensitivities)}\n"
            f"precisions: {precisions}, mean={np.mean(precisions)}, std={np.std(precisions)}\n"
            f"confidences: {confidences}, mean={np.mean(confidences)}, std={np.std(confidences)}\n"
            f'cost so far: {self.tm.get_cost() if self.tm is not None else "N/A"}\n'
        )
        self.rm.dump_dict(name, info_dict)
        return info_dict

    def plot(
        self,
        prefix: str,
        epoch: int,
        batch_accuracy: float,
        epoch_accuracies: List[float],
    ):
        output_folder = self.rm.output_folder_path
        epochs = list(range(epoch))
        accuracies = []
        specificities = []
        sensitivities = []
        precisions = []
        confidences = []
        for e in range(epoch):
            filename = output_folder + f"/{prefix}_epoch_{e}_predictive.json"
            with open(filename) as f:
                json_obj = json.load(f)
            json_obj = json_obj["metrics"]
            accuracies.append(json_obj["accuracies"])
            specificities.append(json_obj["specificities"])
            sensitivities.append(json_obj["sensitivities"])
            precisions.append(json_obj["precisions"])
            confidences.append(json_obj["confidences"])

        # Calculate means and standard deviations for each epoch
        mean_accuracies = [np.mean(epoch_data) for epoch_data in accuracies]
        std_accuracies = [np.std(epoch_data) for epoch_data in accuracies]
        mean_specificities = [np.mean(epoch_data) for epoch_data in specificities]
        std_specificities = [np.std(epoch_data) for epoch_data in specificities]
        mean_sensitivities = [np.mean(epoch_data) for epoch_data in sensitivities]
        std_sensitivities = [np.std(epoch_data) for epoch_data in sensitivities]
        mean_precisions = [np.mean(epoch_data) for epoch_data in precisions]
        std_precisions = [np.std(epoch_data) for epoch_data in precisions]
        mean_confidences = [np.mean(epoch_data) for epoch_data in confidences]
        std_confidences = [np.std(epoch_data) for epoch_data in confidences]

        # Plot each metric
        plt.figure(figsize=(13, 7))

        # Accuracies
        plt.plot(epochs, mean_accuracies, label="Accuracy", color="blue")
        plt.fill_between(
            epochs,
            np.array(mean_accuracies) - np.array(std_accuracies),
            np.array(mean_accuracies) + np.array(std_accuracies),
            color="blue",
            alpha=0.2,
        )
        for e, value in enumerate(mean_accuracies):
            plt.text(
                epochs[e], value, f"{value:.2f}", ha="center", va="bottom", color="blue"
            )

        # Specificities
        plt.plot(epochs, mean_specificities, label="Specificity", color="green")
        plt.fill_between(
            epochs,
            np.array(mean_specificities) - np.array(std_specificities),
            np.array(mean_specificities) + np.array(std_specificities),
            color="green",
            alpha=0.2,
        )
        for e, value in enumerate(mean_specificities):
            plt.text(
                epochs[e],
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color="green",
            )

        # Sensitivities
        plt.plot(epochs, mean_sensitivities, label="Sensitivity", color="red")
        plt.fill_between(
            epochs,
            np.array(mean_sensitivities) - np.array(std_sensitivities),
            np.array(mean_sensitivities) + np.array(std_sensitivities),
            color="red",
            alpha=0.2,
        )
        for e, value in enumerate(mean_sensitivities):
            plt.text(
                epochs[e], value, f"{value:.2f}", ha="center", va="bottom", color="red"
            )

        # Precisions
        plt.plot(epochs, mean_precisions, label="Precision", color="orange")
        plt.fill_between(
            epochs,
            np.array(mean_precisions) - np.array(std_precisions),
            np.array(mean_precisions) + np.array(std_precisions),
            color="orange",
            alpha=0.2,
        )
        for e, value in enumerate(mean_precisions):
            plt.text(
                epochs[e],
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color="orange",
            )

        # Confidences
        plt.plot(epochs, mean_confidences, label="Confidence", color="purple")
        plt.fill_between(
            epochs,
            np.array(mean_confidences) - np.array(std_confidences),
            np.array(mean_confidences) + np.array(std_confidences),
            color="purple",
            alpha=0.2,
        )
        for e, value in enumerate(mean_confidences):
            plt.text(
                epochs[e],
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color="purple",
            )

        # dataset acc
        plt.plot(
            epochs,
            epoch_accuracies,
            label="Train Batch Accuracy",
            color="cyan",
            linestyle="dashed",
        )

        oracle = max(batch_accuracy, 1 - batch_accuracy)
        # plot oracle as a constant dashed line
        plt.plot(
            epochs,
            [oracle] * epoch,
            label="Oracle Accuracy",
            color="gray",
            linestyle="dashed",
        )

        # Adding legend and labels
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.title("Predictive Evaluation Metrics Over Epochs")
        plt.suptitle(
            f"Topic: {self.topic}  Target Model: {self.model}  Batch Acc: {batch_accuracy}  Oracle: {oracle:.2f}"
        )
        plt.legend(loc="best")
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xticks(range(epoch))
        plt.savefig(
            os.path.join(self.rm.output_folder_path, f"{prefix}_predictive_metrics.png")
        )
        # plt.show()
