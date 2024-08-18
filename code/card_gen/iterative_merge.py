import random
from typing import List, Union, Dict, Optional, Tuple, Literal, Self

import numpy as np
from tqdm import trange

from core.card import GenerativeCard
from core.models import CostManager, select_model_from_messages, Model, select_model
from core.utils import (
    ResourceManager,
    get_initial_criteria,
)
from dataset.data import load_batches, Batch


class IterativeMergeAlgorithm:
    # general
    rm: ResourceManager
    cm: CostManager

    # dataset related
    dataset_folder: str
    batch_nums: List[int]
    shuffle: bool
    seed: int
    training_batches: List[Batch]
    testing_batch: Batch

    # training related
    topic: str
    model: str
    evaluator_name: str
    card_format: Literal["dict", "bullet_point", "paragraph"]
    initial_criteria: List[str]
    epochs: int
    card: Union[str, GenerativeCard]  # current card

    def __init__(self, hp: Dict, rm: ResourceManager, cm: CostManager = None):
        """
        An example hp:
        topic = "high_school_mathematics"
        model = "Meta-Llama-3-8B-Instruct"
        hp = {
            "dataset_folder": "datasets/mmlu",
            "topic": topic,
            "model": model,
            "evaluator_name": "gpt-4o",
            "batch_nums": [8] * 5,
            "shuffle": True,
            "seed": 42,
            "initial_criteria": None,  # default to get_initial_criteria(topic)
            "epochs": None,  # default to len(batch_nums)
            "card_format": "bullet_point",
        }
        """
        self.rm = rm
        self.cm = cm if cm is not None else CostManager()
        self.dataset_folder = hp["dataset_folder"]
        self.topic = hp["topic"]
        self.model = hp["model"]
        self.evaluator_name = hp["evaluator_name"]
        self.batch_nums = hp["batch_nums"]
        self.shuffle = hp["shuffle"]
        self.seed = hp["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.initial_criteria = hp["initial_criteria"]
        if self.initial_criteria is None:
            self.initial_criteria = get_initial_criteria(self.topic)
        self.epochs = hp["epochs"]
        self.card_format = hp["card_format"]

        self.training_batches = load_batches(
            self.dataset_folder,
            self.topic,
            self.model,
            "train",
            self.batch_nums,
            shuffle=self.shuffle,
        )
        if self.epochs is None:
            self.epochs = len(self.training_batches)

        # initial card
        if self.card_format == "dict":
            self.card = GenerativeCard(d={c: {} for c in self.initial_criteria})
        elif self.card_format == "bullet_point":
            self.card = GenerativeCard(d={c: "" for c in self.initial_criteria})
        elif self.card_format == "paragraph":
            self.card = ""
        else:
            raise ValueError(f"Invalid card format: {self.card_format}")

        self.rm.dump_dict("iterative_merge-hyperparameters", hp)

    def main(self):
        self.train()
        self.dump_cost_dict()

    def train(self):
        print(
            f"Iterative-Merge method started on topic {self.topic} and model {self.model}"
        )
        for e in trange(self.epochs, desc="Generating Card"):
            p_card, p_model = self.progress(e)
            r_card, r_model = self.regress(e, p_card)

            final_card = r_card
            self.card = final_card
            if self.card_format == "paragraph":
                self.rm.dump_dict(f"cards/epoch_{e}_card", {"card": final_card})
            else:
                self.rm.dump_dict(f"cards/epoch_{e}_card", final_card.to_dict())

            print(f"Epoch {e} Finished. Cost so far: {self.cm.get_cost()}")

    def progress(self, e: int) -> Tuple[Union[GenerativeCard, str], Model]:
        print(f"Epoch {e} Progressing..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        new_card = self.load_card(f"cards/epoch_{e}_progressive_card")
        p_model = self.load_model_from_messages(f"logs/epoch_{e}_progressive")
        if new_card is not None and p_model is not None:
            return new_card, p_model

        # progression
        batch_str = self.training_batches[e].get_train_str()
        system_prompt = self.rm.get_prompt(
            "gen/iterative_merge/progressive/system"
        ).format(topic=self.topic)
        user_prompt = self.rm.get_prompt(
            f"gen/iterative_merge/progressive/user-{self.card_format}"
        ).format(topic=self.topic, batch=batch_str, criteria=self.initial_criteria)
        user_prompt += self.rm.get_prompt(
            f"gen/iterative_merge/progressive/formatting-{self.card_format}"
        )

        p_model = select_model(self.evaluator_name, system_prompt, self.cm)
        json_obj = p_model(user_prompt, use_json=True)

        self.rm.dump_dict(
            f"logs/epoch_{e}_progressive",
            {"step": "progressive", "conversation": p_model.messages},
        )

        if self.card_format == "str":
            new_card = json_obj["assessment"]  # str
            card_dict = {"card": new_card}
        else:
            new_card = GenerativeCard(d=json_obj)
            card_dict = new_card.to_dict()

        self.rm.dump_dict(f"cards/epoch_{e}_progressive_card", card_dict)
        return new_card, p_model

    def regress(
        self, e: int, p_card: Union[GenerativeCard, str]
    ) -> Tuple[Union[GenerativeCard, str], Optional[Model]]:
        if e == 0:  # no regression for the first epoch
            return p_card, None
        print(f"Epoch {e} Regressing..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        r_card = self.load_card(f"cards/epoch_{e}_regressive_card")
        r_model = self.load_model_from_messages(f"logs/epoch_{e}_regressive")
        if r_card is not None and r_model is not None:
            return r_card, r_model

        # merge the two cards
        system_prompt = self.rm.get_prompt(
            "gen/iterative_merge/regressive/system"
        ).format(topic=self.topic)
        cards_str = (
            f"### Summary 1\n\n{str(self.card)}\n\n### Summary 2\n\n{str(p_card)}"
        )
        user_prompt = self.rm.get_prompt(
            f"gen/iterative_merge/regressive/user-{self.card_format}"
        ).format(cards=cards_str)
        user_prompt += self.rm.get_prompt(
            f"gen/iterative_merge/regressive/formatting-{self.card_format}"
        )

        r_model = select_model(self.evaluator_name, system_prompt, self.cm)
        json_obj = r_model(user_prompt, use_json=self.card_format != "paragraph")
        self.rm.dump_dict(
            f"logs/epoch_{e}_regressive",
            {"step": "regressive", "conversation": r_model.messages},
        )

        if self.card_format == "paragraph":
            r_card = json_obj
            card_dict = {"card": r_card}
        else:
            r_card = GenerativeCard(d=json_obj)
            card_dict = r_card.to_dict()
        self.rm.dump_dict(f"cards/epoch_{e}_regressive_card", card_dict)
        return r_card, r_model

    def load_card(self, card_filename: str) -> Optional[Union[GenerativeCard, str]]:
        """
        Try to load a card from a file. If the file does not exist, return None.
        """
        if self.rm.file_exists(card_filename):
            card_dict = self.rm.load_dict(card_filename)
            if len(card_dict) == 1 and "card" in card_dict:  # if paragraph
                return card_dict["card"]
            else:
                return GenerativeCard(d=card_dict)
        return None

    def load_model_from_messages(self, info_filename: str) -> Optional[Model]:
        """
        Try to load a model from the messages of a previous conversation. If the file does not exist, return None.
        """
        if self.rm.file_exists(info_filename):
            return select_model_from_messages(
                self.evaluator_name,
                self.rm.load_dict(info_filename)["conversation"],
                self.cm,
            )
        return None

    def shutdown(self):
        self.rm.shutdown()

    def dump_cost_dict(self):
        i = 0
        while self.rm.file_exists(f"costs/cost_{i}"):
            i += 1
        self.rm.dump_dict(f"costs/cost_{i}", self.cm.get_info_dict())

    @classmethod
    def load_instance(cls, rm: ResourceManager, cm: CostManager) -> Self:
        hp = rm.load_dict("pr-hyperparameters")
        return cls(hp, rm, cm)


if __name__ == "__main__":
    rm = ResourceManager("pr-debug", "debug")
    cm = CostManager()
    topic = "high_school_mathematics"
    model = "Meta-Llama-3-8B-Instruct"
    hp = {
        "dataset_folder": "datasets/mmlu",
        "topic": topic,
        "model": model,
        # "evaluator_name": "gpt-4o",
        "evaluator_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "batch_nums": [8] * 5,
        "shuffle": False,
        "seed": 42,
        "initial_criteria": None,  # default to get_initial_criteria(topic)
        "epochs": None,  # default to len(batch_nums)
        "card_format": "bullet_point",
    }
    pr = IterativeMergeAlgorithm(hp, rm, cm)
    pr.main()

    # eval
    from eval.predictive import PredictiveEvaluator, plot_epochs

    pe = PredictiveEvaluator(topic, model, rm, "mmlu", cm=cm)
    testing_batch = load_batches("datasets/mmlu", topic, model, "test", [60])[0]
    for i in range(5):
        pe.main(
            f"eval/predictive_eval-epoch_{i}_card",
            testing_batch,
            GenerativeCard(d=rm.load_dict(f"cards/epoch_{i}_card")),
            num_times=3,
        )

    plot_epochs(
        topic,
        model,
        [f"eval/predictive_eval-epoch_{i}_card.json" for i in range(5)],
        testing_batch,
        pr.training_batches,
        rm,
    )

    pe.shutdown()
    pr.shutdown()
