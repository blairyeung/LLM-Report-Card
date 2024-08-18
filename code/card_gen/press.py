import random
from typing import List, Union, Dict, Optional, Tuple, Literal, Self

import numpy as np
from tqdm import trange

from core.card import GenerativeCard, get_num_words
from core.models import (
    select_model,
    CostManager,
    Model,
    select_model_from_messages,
)
from core.utils import (
    REG_WORD_LIM,
    ResourceManager,
    get_initial_criteria,
)
from dataset.data import load_batches, Batch


class PressAlgorithm:
    # general
    rm: ResourceManager
    cm: CostManager

    # dataset related
    dataset_folder: str
    batch_nums: List[int]
    shuffle: bool
    seed: int
    training_batches: List[Batch]

    # training related
    topic: str
    teacher_model_name: str
    student_model_name: str
    card_format: Literal["dict", "bullet_point", "paragraph"]
    initial_criteria: List[str]
    num_epochs: int
    cards: List[Union[str, GenerativeCard]]  # includes the current and previous cards
    card: Union[str, GenerativeCard]

    def __init__(self, hp: Dict, rm: ResourceManager, cm: CostManager = None):
        """
        An example hp:
        topic = "high_school_mathematics"
        hp = {
            "dataset_folder": "datasets/mmlu",
            "batch_nums": [8] * 5,
            "shuffle": False,
            "seed": 42,
            "topic": topic,
            "teacher_model_name": "gpt-4o-2024-05-13",
            "student_model_name": "Meta-Llama-3-8B-Instruct",
            "card_format": "bullet_point",
            "initial_criteria": get_initial_criteria(topic),
            "num_epochs": 5,
        }
        """
        self.rm = rm
        self.cm = cm if cm is not None else CostManager()

        self.dataset_folder = hp["dataset_folder"]
        self.batch_nums = hp["batch_nums"]
        self.shuffle = hp["shuffle"]
        self.seed = hp["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.topic = hp["topic"]
        self.teacher_model_name = hp["teacher_model_name"]
        self.student_model_name = hp["student_model_name"]
        self.card_format = hp["card_format"]
        self.initial_criteria = hp["initial_criteria"]
        self.num_epochs = hp["num_epochs"]

        self.training_batches = load_batches(
            self.dataset_folder,
            self.topic,
            self.student_model_name,
            "train",
            self.batch_nums,
            shuffle=self.shuffle,
        )
        if self.num_epochs is None:
            self.num_epochs = len(self.training_batches)

        self.cards = []
        self.rm.dump_dict("press-hyperparameters", hp)

    def main(self):
        self.train()
        self.dump_cost_dict()

    def dump_cost_dict(self):
        i = 0
        while self.rm.file_exists(f"costs/cost_{i}"):
            i += 1
        self.rm.dump_dict(f"costs/cost_{i}", self.cm.get_info_dict())

    def train(self):
        print(
            f"Training started on topic {self.topic} and model {self.student_model_name}"
        )
        for e in trange(self.num_epochs, desc="Training"):
            p_card, p_model = self.progress(e)
            self.cards.append(p_card)

            if get_num_words(self.cards) > REG_WORD_LIM or e == self.num_epochs - 1:
                r_card, r_model = self.regress(e, p_card)
                self.cards.clear()
                self.cards.append(r_card)
            else:
                r_card, r_model = p_card, p_model

            final_card = r_card

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
        system_prompt = self.rm.get_prompt("gen/press/progressive/system").format(
            topic=self.topic
        )
        user_prompt = self.rm.get_prompt(
            f"gen/press/progressive/user-{self.card_format}"
        ).format(topic=self.topic, batch=batch_str, criteria=self.initial_criteria)
        user_prompt += self.rm.get_prompt(
            f"gen/press/progressive/formatting-{self.card_format}"
        )

        p_model = select_model(self.teacher_model_name, system_prompt, self.cm)
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
        self, e: int, p_card: GenerativeCard
    ) -> Tuple[Union[GenerativeCard, str], Optional[Model]]:
        if e == 0:  # no regression for the first epoch
            return p_card, None
        print(f"Epoch {e} Regressing..., cost so far: {self.cm.get_cost()}")
        # try to recover from file
        r_card = self.load_card(f"cards/epoch_{e}_regressive_card")
        r_model = self.load_model_from_messages(f"logs/epoch_{e}_regressive")
        if r_card is not None and r_model is not None:
            return r_card, r_model

        system_prompt = self.rm.get_prompt("gen/press/regressive/system").format(
            topic=self.topic
        )
        all_cards_str = ""
        for i, card in enumerate(self.cards):
            all_cards_str += f"### Summary {i + 1}\n\n{str(card)}\n\n"
        user_prompt = self.rm.get_prompt(
            f"gen/press/regressive/user-{self.card_format}"
        ).format(cards=all_cards_str)
        user_prompt += self.rm.get_prompt(
            f"gen/press/regressive/formatting-{self.card_format}"
        )
        r_model = select_model(self.teacher_model_name, system_prompt, self.cm)
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
                self.teacher_model_name,
                self.rm.load_dict(info_filename)["conversation"],
                self.cm,
            )
        return None

    def shutdown(self):
        self.rm.shutdown()

    @classmethod
    def load_instance(cls, rm: ResourceManager, cm: CostManager) -> Self:
        hp = rm.load_dict("pr-hyperparameters")
        return cls(hp, rm, cm)


def run_exp():
    topic = "high_school_mathematics"
    hp = {
        "dataset_folder": "datasets/mmlu",
        "batch_nums": [8] * 5,
        "shuffle": True,
        "seed": 42,
        "topic": topic,
        "teacher_model_name": "gpt-4o-2024-05-13",
        "student_model_name": "Meta-Llama-3-8B-Instruct",
        "card_format": "bullet_point",
        "initial_criteria": get_initial_criteria(topic),
        "num_epochs": 5,
    }
    rm = ResourceManager("press-debug")
    cm = CostManager()
    press = PressAlgorithm(hp, rm, cm)
    press.main()


if __name__ == "__main__":
    run_exp()
