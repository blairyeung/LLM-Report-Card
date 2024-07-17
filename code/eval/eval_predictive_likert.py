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


class LikertEvaluator:
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
            f"eval/likert/system"
        ).format(topic=self.topic)
        model = select_model(self.guesser_name, system_prompt)
        user_prompt = self.rm.get_prompt("eval/likert/user").format(
            card=str(card), qa=batch.get_eval_predictive_str(index)
        )
            
        rslt = model(user_prompt, use_json=False, timeout=30)
        # print(f"rslt: {rslt}")
        try:
            # Try to parse the result as JSON
            if isinstance(rslt, str):
                rslt = json.loads(rslt)
                
            if isinstance(rslt, dict):
                if "accuracy" in rslt:
                    accuracy_rating = int(rslt["accuracy"])
                else:
                    raise ValueError(f"Response not parsable: {rslt}")
            else:
                raise ValueError(f"Unexpected type for rslt: {type(rslt)}")
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try regex search
            if isinstance(rslt, str):
                match = re.search(r'"accuracy"\s*:\s*(\d+)', rslt)
                if match is not None:
                    accuracy_rating = int(match.group(1))
                else:
                    raise ValueError(f"Accuracy rating not found in string: {rslt}")
            else:
                raise ValueError(f"Unexpected type for rslt after JSON parsing: {type(rslt)}")

        
        return accuracy_rating, model, index
    
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
                rslt = model.get_logits(user_prompt, post_fix='{{\n"prediction": ')
                actual, probs = get_decision(rslt, ['true', 'false'])
                expected = batch.get_true_answer(index) == batch.get_model_answer(index)

                # print(f'actual: {actual}, expected: {expected}, confidence: {probs}')
                confidence = max(probs)
                return actual, expected, confidence, model, index
            else:
                pass

        except Exception as e:
            print(e, file=sys.stderr)
            return None
        
    def eval(self, batch: Batch, card: Union[GenerativeCard, str]):
        info_dict = {"details": {}}
        ratings = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(self.helper, batch, i, card)
                for i in range(len(batch))
            ]
            for future in tqdm(as_completed(futures)):
                if future.result() is None:
                    continue
                accuracy_rating, model, index = future.result()
                ratings.append(accuracy_rating)
                info_dict["details"][str(index)] = {
                    "accuracy_rating": accuracy_rating,
                    "conversation": model.messages, 
                }

        mean_accuracy_rating = np.mean(ratings)
        info_dict["metrics"] = {
            "accuracies": mean_accuracy_rating,
            "sd_accuracy_rating": np.std(ratings),
        }
        return info_dict, mean_accuracy_rating

    def main(
        self,
        name: str,
        batch: Batch,
        card: Union[GenerativeCard, str, dict],
        num_times: int = 3,
        cot = False
    ):
        self.cot = cot
        info_dict = {
            "method": "likert_rating",
            "topic": self.topic,
            "model": self.model,
            "evaluator": self.guesser_name,
            "iterations": [],
        }
        mean_ratings = []
        for j in range(num_times):
            sub_info, mean_rating = self.eval(batch, card)
            info_dict["iterations"].append(sub_info)
            mean_ratings.append(mean_rating)

        info_dict["metrics"] = {
            "accuracies": mean_ratings,
            "mean accuracies": np.mean(mean_ratings),
            "std accuracies": np.std(mean_ratings),
        }
        print(
            f"{name}\n"
            f"mean_accuracy_ratings: {mean_ratings}, mean={np.mean(mean_ratings)}, std={np.std(mean_ratings)}\n"
            f'cost so far: {self.tm.get_cost() if self.tm is not None else "N/A"}\n'
        )
        self.rm.dump_dict(name, info_dict)
        return info_dict

    def is_gpt_evaluator(self):
        return self.guesser_name in [GPT_3_MODEL_NAME, GPT_4_MODEL_NAME]