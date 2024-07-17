import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Dict, List

import numpy as np
from tqdm import tqdm

from core.card import GenerativeCard
from core.config import GPT_3_MODEL_NAME, GPT_4_MODEL_NAME, LLAMA_MODEL_NAME
from core.data import Batch
from core.models import CostManager, select_model
from core.utils import ResourceManager

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
    # print(logits)
    for pair in pairs:
        decision.append(dict_key_helper(logits, pair)[1])

    # softmax decision
    decision = np.exp(decision) / np.sum(np.exp(decision))

    return decision[0] > decision[1], decision


class PredictiveEvaluator:
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

        # breakpoint()

        self.topic = topic
        self.model = model
        self.rm = rm
        self.tm = tm
        self.guesser_name = evaluator_name

    def helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive/system_{self.type_}"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt)
        if self.cot:
            user_prompt_path = "eval/predictive/user_cot"
        else:
            user_prompt_path = "eval/predictive/user"
        if self.type_ == "few-shot":
            user_prompt_path += "_few-shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), qa=batch.get_eval_predictive_str(index)
        )
        try:
            # if "llama" in model.name.lower()
            if True:
                try:
                    rslt = model(user_prompt, use_json=False, timeout=30)
                    json_obj = json.loads(rslt)
                    if "Prediction" in json_obj:
                        prediction = json_obj["Prediction"]
                        if isinstance(prediction, bool):
                            actual = prediction
                        elif isinstance(prediction, str):
                            actual = prediction.lower() in ["true", '"true"']
                        else:
                            raise ValueError(f"Unexpected prediction value: {prediction}")
                        # confidence = float(json_obj.get("Confidence", 0))
                        confidence = 0
                    else:
                        raise ValueError(f"Prediction not found in JSON: {json_obj}")
                except json.JSONDecodeError:
                    if "prediction" in rslt.lower():
                        # "prediction: true/false" or "prediction": "true"/"false"
                        match = re.search(r'"?prediction"?\s*:\s*("?\w+"?)', rslt, re.IGNORECASE)
                        if match is None:
                            # replace all " and try again
                            rslt = rslt.replace('"', '')
                            match = re.search(r'prediction\s*:\s*(\w+)', rslt, re.IGNORECASE)
                        assert match is not None
                        pred_value = match.group(1).lower().strip('"')
                        actual = pred_value in ["true", "t", "yes", "y"]
                        confidence = 0
                    else:
                        raise ValueError(f"Prediction not found in string: {rslt}")

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
        
    def logit_helper(self, batch: Batch, index: int, card: Union[GenerativeCard, str]):
        system_prompt = self.rm.get_prompt(
            f"eval/predictive/system_{self.type_}"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt)
        user_prompt_path = "eval/predictive/user"
        if self.type_ == "few-shot":
            user_prompt_path += "_few-shot"
        user_prompt = self.rm.get_prompt(user_prompt_path).format(
            card=str(card), qa=batch.get_eval_predictive_str(index)
        )

        try:
            if True:
            # if "llama" in model.name.lower():
                # model.add_message(user_prompt)
                rslt = model.get_logits(user_prompt, post_fix='{{\n"prediction": ')
                actual, probs = get_decision(rslt, ['true', 'false'])
                expected = batch.get_true_answer(index) == batch.get_model_answer(index)

                # print(f'actual: {actual}, expected: {expected}, confidence: {probs}')
                confidence = max(probs)
                # confidence = 0

                return actual, expected, confidence, model, index
            else:
                pass

        except Exception as e:
            print(e, file=sys.stderr)
            return None
    
    def eval(self, batch: Batch, card: Union[GenerativeCard, str]):
        info_dict = {"details": {}}
        count = 0
        TP_indices, TN_indices, FP_indices, FN_indices = [], [], [], []
        confidences = [float("nan")] * len(batch)
        with ThreadPoolExecutor(max_workers=50) as executor:
            if self.cot:
                futures = [
                    executor.submit(self.helper, batch, i, card)
                    for i in range(len(batch))
                ]
            else:
                futures = [
                    executor.submit(self.logit_helper, batch, i, card)
                    for i in range(len(batch))
                ]
            for future in tqdm(as_completed(futures)):
                if future.result() is None:
                    continue
                actual, expected, confidence, model, index = future.result()
                confidences[index] = confidence / 5  # normalize
                count += 1
                # print(f'actual: {actual}, expected: {expected}, confidence: {confidence}')
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
        cot = False
    ):  
        self.cot = cot
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
