import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
import re

from utils.data import Batch
from utils.models import CostManager, select_model
from utils.utils import ResourceManager


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
        self.k_shots = k_shots
        self.max_workers = max_workers
    

    def helper(
        self, index: int, card: str, q: str, a1: str, a2: str, target_model: int
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
        system_prompt = self.rm.get_prompt("eval/contrastive-answer-2/system").format(
            topic=self.topic
        )
        evaluator = select_model(self.evaluator_model, system_prompt, self.cm)
        user_prompt = self.rm.get_prompt("eval/contrastive-answer-2/user").format(
            card=card, q=q, a1=a1, a2=a2
        )
        r = evaluator(user_prompt, temperature=0, use_json=False)
        # TODO: add decision extractor
        try:
            # prediction: (a or b)
            pattern = re.compile(r"prediction:\s*([ab])", re.IGNORECASE)
            pattern2 = re.compile(r"\*\*prediction:\*\*\s*([ab])", re.IGNORECASE)
            # either match is fine
            match = pattern.search(r.lower())
            if match is None:
                match = pattern2.search(r.lower())
        
            if (match.group()[-1]) == 'b':
                decision = 1
            elif (match.group()[-1]) == 'a':
                decision = 0
            else:
                decision = -1
                print(f"Error: {r}")

        except Exception:
            decision = -1
            print(f"Error: {r}")

        info_dict = {
            "decision": decision,
            "target_model": target_model,
            "conversation": evaluator.messages,
        }
        return decision, info_dict, index, target_model

    def main(self, num_times: int = 1, max_workers: int = 40) -> List[float]:
        """
        The main function to run the contrastive-answer-2 evaluation.
        :param num_times: Number of times to run the evaluation.
        :param max_workers: Maximum number of parallel workers to use.
        """
        info_dict = {
            "type": "eval",
            "method": "contrastive-answer-2",
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
                    # print(self.batches)
                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[0],
                            self.batches[0].get_question(index),
                            self.batches[0].get_model_reasoning(index),
                            self.batches[1].get_model_reasoning(index),
                            0,
                        )
                    )
                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[1],
                            self.batches[0].get_question(index),
                            self.batches[0].get_model_reasoning(index),
                            self.batches[1].get_model_reasoning(index),
                            1,
                        )
                    )

                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[0],
                            self.batches[0].get_question(index),
                            self.batches[1].get_model_reasoning(index),
                            self.batches[0].get_model_reasoning(index),
                            1,
                        )
                    )

                    futures.append(
                        executor.submit(
                            self.helper,
                            index,
                            self.cards[1],
                            self.batches[0].get_question(index),
                            self.batches[1].get_model_reasoning(index),
                            self.batches[0].get_model_reasoning(index),
                            0,
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
        name = f"eval_contrastive-answer-2_{'_'.join(self.models)}"
        print(
            f"{name}\n"
            f"accuracy: {mean} ± {sd} accuracies: {metrics}\n"
        )
        self.rm.dump_dict(name, info_dict)
        # self.rm.dump_dict(f"{name}_cost", self.cm.get_info_dict())
        return metrics

        