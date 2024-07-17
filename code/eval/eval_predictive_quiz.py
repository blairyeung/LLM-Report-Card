import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Dict, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.card import GenerativeCard
from core.config import GPT_3_MODEL_NAME, GPT_4_MODEL_NAME, LLAMA_MODEL_NAME
from core.data import Batch
from core.models import CostManager, select_model
from core.utils import ResourceManager, sample_few_shots

DEFAULT_EVALUATOR = LLAMA_MODEL_NAME


def dict_key_helper(logit_dict, item: str):
    """
    Find max possible key in dict.
    """

    max_val = -100
    max_key = -1
    for key, value in logit_dict.items():
        modified = key.replace(" ", "").lower()
        if modified in item.lower() and value > max_val:
            max_val = value
            max_key = key

    return max_key, max_val


def get_decision(logits: Dict, pairs: List[str]):
    """
    Get the decision from the logits.
    """
    decision = []
    print(logits)
    for pair in pairs:
        decision.append(dict_key_helper(logits, pair)[1])

    # softmax decision
    decision = np.exp(decision) / np.sum(np.exp(decision))

    return decision[0] > decision[1], decision


class PredictiveQuizEvaluator:
    type_: str
    topic: str
    model: str
    rm: ResourceManager
    tm: CostManager
    guesser_name: str

    def __init__(
        self,
        topic: str,
        model: str,
        rm: ResourceManager,
        type_: str,
        evaluator_name: str = DEFAULT_EVALUATOR,
        k_shots: int = 4,
        oracle: float = 0,
        tm: CostManager = None,
    ):
        self.type_ = type_

        assert type_ in [
            "mmlu",
            "anthropic",
            "few-shot",
            "gsm8k",
            "hotpot_qa",
            "openend",
        ]

        # 

        self.topic = topic
        self.model = model
        self.rm = rm
        self.tm = tm
        self.k_shots = k_shots
        self.guesser_name = evaluator_name
        self.oracle_accuracy = oracle
    
    def format_qa_str(self, indices, batch):
        cnt = 1
        qa_str = ""
        for index in indices:
            qa_str += f"### Quiz Question {cnt}: {batch.get_question(index)}\n"
            cnt += 1
        return qa_str
    
    def compute_correct_cnt(self, indices, batch):
        cnt = 0
        for index in indices:
            if batch.get_true_answer(index) == batch.get_model_answer(index):
                cnt += 1
        return cnt

    def helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive-exam/system_{self.type_}"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt)
        user_prompt_path = "eval/predictive-exam/user"
        if self.type_ == "few-shot":
            user_prompt_path += "_few-shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), quiz=batch.get_eval_predictive_str(index)
        )
        try:
            if "llama" in model.name.lower():
                rslt = model(user_prompt, timeout=30)
                if isinstance(rslt, str):
                    if "prediction" in rslt:
                        # re to match it
                        match = re.search(r'"prediction": (\w+)', rslt)
                        actual = "true" in match.group(1).lower()
                        confidence = 0

                else:
                    assert False

                # print(False)
                expected = batch.get_true_answer(index) == batch.get_model_answer(index)
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

    def logit_helper(self, batch: Batch, index: int, samples: List, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive-exam/system_{self.type_}"
        ).format(topic=self.topic, cnt=str(self.k_shots))

        qa_str = self.format_qa_str(samples, batch)
        corrct_cnt = self.compute_correct_cnt(samples, batch)
        
        model = select_model(self.guesser_name, system_prompt)
        user_prompt_path = "eval/predictive-exam/user"
        if self.type_ == "few-shot":
            user_prompt_path += "_few-shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), quiz=qa_str, cnt=str(self.k_shots)
        )
        # print(user_prompt)
        try:
            if "llama" in model.name.lower():
                rslt = model.get_logits(user_prompt, post_fix='choice: ')
                # print(rslt)
                actual = get_decision(rslt, ['A', 'B', 'C', 'D', 'E'])[1]
                print(actual)
                actual = np.sum(actual * np.array(list(range(self.k_shots + 1))))
                # do argmax
                # actual = np.argmax(actual)
                oracle = self.oracle_accuracy * self.k_shots
                expected = corrct_cnt

                print(f'actual: {actual}, expected: {expected}, oracle: {oracle}')

                return actual, expected, oracle, model, index
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

    def eval(self, batch: Batch, card: Union[GenerativeCard, str]):
        info_dict = {"details": {}}
        count = 0
        TP_indices, TN_indices, FP_indices, FN_indices = [], [], [], []
        confidences = [float("nan")] * len(batch)
        num_samples = 60
        pred_total_mse = 0
        oracle_total_mse = 0
        pred_total_dist = []
        oracle_total_dist = []
        pred_list = []
        samples = sample_few_shots(num_samples, len(batch), self.k_shots)
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(self.logit_helper, batch, i, samples[i], card)
                # for i in range(1)
                for i in range(num_samples)
            ]
            for future in tqdm(as_completed(futures)):
                if future.result() is None:
                    continue
                actual, expected, oracle, model, index = future.result()
                # confidences[index] = 0  # normalize
                count += 1
                pred_list.append(actual)
                # print(f'actual: {actual}, expected: {expected}, confidence: {confidence}')
                
                pred_mse = (actual - expected) ** 2 / num_samples
                oracle_mse = (oracle - expected) ** 2 / num_samples

                pred_total_mse += pred_mse
                oracle_total_mse += oracle_mse

                pred_total_dist.append([actual, expected])
                oracle_total_dist.append([oracle, expected])
                info_dict["details"][str(index)] = {
                    "actual": actual,
                    "expected": expected,
                    "oracle": oracle,
                    "pred_mse": pred_mse,
                    "oracle_mse": oracle_mse,
                    "conversation": model.messages,
                }

        # reshape the pred distribution 
        # expected to follow Bionomial distribution with n = 4, p = 0.5
        # reshape the oracle distribution into such var and mean
        pred_dist_mean = self.oracle_accuracy * self.k_shots
        pred_dist_var = self.oracle_accuracy * (1 - self.oracle_accuracy)

        org_dist_stdev =  np.sqrt(np.var(pred_list))
        
        # shift it to mean and var
        pred_list = (pred_list - np.mean(pred_list)) / org_dist_stdev * np.sqrt(pred_dist_var) + pred_dist_mean
        # clip 
        pred_list = np.clip(pred_list, 0, 4)

        pred_total_mse = np.mean((np.array(pred_list) - np.array(pred_total_dist)[:, 1]) ** 2) 
        oracle_total_mse = np.mean((np.array(oracle_total_dist)[:, 0] - np.array(oracle_total_dist)[:, 1]) ** 2)
        pred_total_dist = np.array(pred_total_dist)

        pred_total_dist[:, 0] = pred_list

        print(pred_list)
    
    
        info_dict["metrics"] = {
            "Pred MSE": pred_total_mse,
            "Oracle MSE": oracle_total_mse,
            "oracle": self.oracle_accuracy,
            "pred_dist": np.array(pred_total_dist),
            "oracle_dist": np.array(oracle_total_dist),
        }

        # plot 4 subplots 1. distribution of the actual 
        # 2. distribution of the expected 3. diifference between expected and actual 4. difference between expected and oracle
        plt.figure(figsize=(12, 9))
        plt.subplot(2, 2, 1)
        plt.hist(np.array(pred_total_dist)[:, 0], bins=20, alpha=0.5, label='Pred')
        plt.hist(np.array(pred_total_dist)[:, 1], bins=5, alpha=0.5, label='Expected')
        # add a vline indicating average
        plt.vlines(np.mean(np.array(pred_total_dist)[:, 0]), 0, 20, colors='r', linestyles='dashed', label='Pred Average')
        plt.legend(loc='upper right')
        plt.title(f'Distribution of Pred and Expected of model {self.model}')
        plt.subplot(2, 2, 2)
        plt.hist(np.array(oracle_total_dist)[:, 0], bins=20, alpha=0.5, label='Oracle')
        plt.hist(np.array(oracle_total_dist)[:, 1], bins=5, alpha=0.5, label='Expected')
        # add a vline indicating average
        plt.vlines(np.mean(np.array(oracle_total_dist)[:, 0]), 0, 20, colors='r', linestyles='dashed', label='Oracle Average')
        plt.legend(loc='upper right')
        plt.title(f'Distribution of Oracle and Expected of model {self.model}')
        plt.subplot(2, 2, 3)
        plt.hist(np.abs(np.array(pred_total_dist)[:, 0] - np.array(pred_total_dist)[:, 1]), bins=20, alpha=0.5, label='Pred - Expected')
        plt.legend(loc='upper right')
        plt.title('Difference between Pred and Expected')
        plt.subplot(2, 2, 4)
        plt.hist(np.abs(np.array(oracle_total_dist)[:, 0] - np.array(oracle_total_dist)[:, 1]), bins=20, alpha=0.5, label='Oracle - Expected')
        plt.legend(loc='upper right')
        plt.title('Difference between Oracle and Expected')
        plt.tight_layout()
        plt.show()

        return (
            info_dict,
            (pred_total_mse, oracle_total_mse, self.oracle_accuracy, np.array(pred_total_dist), np.array(oracle_total_dist)),
            (TP_indices, TN_indices, FP_indices, FN_indices),
        )

    def eval_sub_sample(self, batch: Batch, card: GenerativeCard):
        info_dict = {"details": {}}
        sub_cards = card.sub_sample(8, 5)  # 25 criteria should cover the longest card
        results = [[] for _ in range(len(batch))]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(self.helper, batch, i, sub_card)
                for i in range(len(batch))
                for sub_card in sub_cards
            ]
            for future in tqdm(as_completed(futures)):
                if future.result() is None:
                    continue
                actual, expected, model, index = future.result()
                results[index].append((actual, expected))
                if str(index) in info_dict["details"]:
                    info_dict["details"][str(index)]["sub_samples"].append(
                        {
                            "actual": actual,
                            "expected": expected,
                            "conversation": model.messages,
                        }
                    )
                else:
                    info_dict["details"][str(index)] = {
                        "sub_samples": [
                            {
                                "actual": actual,
                                "expected": expected,
                                "conversation": model.messages,
                            }
                        ],
                    }

        count = 0
        TP_indices, TN_indices, FP_indices, FN_indices = [], [], [], []
        for index, result in enumerate(results):
            if len(result) == 0:
                continue
            # majority vote
            # n = len(result)
            # if n % 2 == 0:
            #     continue
            # else:
            #     actual = sum(int(a) for a, _ in result) > (n / 2)
            # minority vote
            expected = result[0][1]
            if any(a == expected for a, _ in result):
                actual = expected
            else:
                actual = not expected
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
            info_dict["details"][str(index)]["actual"] = actual
            info_dict["details"][str(index)]["expected"] = expected

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
        }
        return (
            info_dict,
            (accuracy, specificity, sensitivity, precision),
            (TP_indices, TN_indices, FP_indices, FN_indices),
        )

    def main(
        self,
        name: str,
        batch: Batch,
        card: Union[GenerativeCard, str, dict],
        num_times: int = 3,
        sub_sample_threshold: int = None,
    ):
        if sub_sample_threshold is None:
            sub_sample_threshold = sys.maxsize
        else:
            assert isinstance(card, GenerativeCard)
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
            #     if isinstance(card, GenerativeCard) and len(card) > sub_sample_threshold:
            #         # TODO: fix this functionality (add confidence)
            #         sub_info, stats, sub_indicies = self.eval_sub_sample(batch, card)
            #     else:
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

        #  (pred_total_mse, oracle_total_mse, self.oracle_accuracy, np.array(pred_total_dist), np.array(oracle_total_dist)),
        print(
            f"{name}\n"
            f"Pred total mse: {accuracies}, mean={np.mean(accuracies)}, std={np.std(accuracies)}\n"
            f"Oracle total mse: {specificities}, mean={np.mean(specificities)}, std={np.std(specificities)}\n"
            f"Oracle Accuracy: {sensitivities}, mean={np.mean(sensitivities)}, std={np.std(sensitivities)}\n"
            f'cost so far: {self.tm.get_cost() if self.tm is not None else "N/A"}\n'
        )
        self.rm.dump_dict(name, info_dict)
        return info_dict

    def is_gpt_evaluator(self):
        return self.guesser_name in [GPT_3_MODEL_NAME, GPT_4_MODEL_NAME]

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
