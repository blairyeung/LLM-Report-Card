from __future__ import annotations

import json
import os.path
import re
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable, Self, Union, Optional


class GenerativeCard:
    criteria: List[Tuple[str, Union[str, Dict]]]

    def __init__(self, filename: str = None, d: Dict = None):
        self.criteria = []
        if filename is not None:
            with open(filename) as f:
                d = json.load(f)
        if d is not None:
            for k, v in d.items():
                self.criteria.append((k, v))

    def get_criteria_str(self) -> str:
        r = ""
        for k, _ in self.criteria:
            r += f"- {k}\n"
        return r

    def get_criteria(self) -> List[str]:
        return [k for k, _ in self.criteria]

    def add_criteria(self, criteria: Iterable[str]):
        for c in criteria:
            self.criteria.append((c, ""))

    def __str__(self) -> str:
        r = ""
        for k, v in self.criteria:
            if isinstance(v, str):
                if v.strip() == "":
                    continue
                r += f"- {k}: {v}\n"
            elif isinstance(v, dict):
                r += f"- {k}: {v['overview']}\n"
                if v["thinking_pattern"] + v["strength"] + v["weakness"] == "":
                    continue
                r += f"    - Thinking Patterns: {v['thinking_pattern']}\n"
                r += f"    - Strength: {v['strength']}\n"
                r += f"    - Weakness: {v['weakness']}\n"
            else:
                raise ValueError(f"Unknown type: {type(v)}")

        return r

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.criteria)

    def __getitem__(self, key) -> Union[str, Self]:
        if isinstance(key, list):
            d = self.to_dict()
            return GenerativeCard(d={k: d[k] for k in key if k in d})
        return self.criteria[key]

    def __setitem__(self, key, value):
        self.criteria[key] = value

    def __iter__(self):
        return iter(self.criteria)

    def __contains__(self, item):
        return item in self.criteria

    def __eq__(self, other):
        return self.criteria == other.criteria

    def to_dict(self) -> OrderedDict:
        return OrderedDict(self.criteria)

    def get_num_words(self) -> int:
        self_str = str(self)
        # replace all non-word characters with spaces
        self_str = re.sub(r"\W+", " ", self_str)
        return len(re.findall(r"\w+", self_str))

    def __add__(self, other: Self) -> Self:
        """
        Merge the two cards into one.
        - If the two cards have the same criteria, the values are concatenated.
        - If the two cards have different criteria, the resulting card will have all criteria.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")
        raise NotImplementedError("Not implemented yet")


def load_card(
    dataset: str,
    topic: str,
    algorithm: str,
    card_format: str,
    teacher_model: str,
    student_model: str,
    card_identifier: Union[str, int],
) -> Optional[GenerativeCard]:
    """
    Load a card from the cards/ directory.
    The cards/ directory contains all the final cards.
    """
    card_folder_path = os.path.join(
        "cards", dataset, topic, algorithm, card_format, teacher_model, student_model
    )
    if isinstance(card_identifier, int):  # epoch
        card_folder_path = os.path.join(
            card_folder_path, f"epoch_{card_identifier}_card.json"
        )
    elif isinstance(card_identifier, str):  # filename
        if not card_identifier.endswith(".json"):
            card_identifier += ".json"
        card_folder_path = os.path.join(card_folder_path, card_identifier)
    else:
        raise ValueError(f"Invalid card_identifier: {card_identifier}")

    if os.path.exists(card_folder_path):
        return GenerativeCard(filename=card_folder_path)
    else:   
        return None
    

def get_num_words(cards: Union[GenerativeCard, Iterable[GenerativeCard]]) -> int:
    """
    Get the number of words in the cards.
    """
    if isinstance(cards, GenerativeCard):
        return cards.get_num_words()
    return sum(card.get_num_words() for card in cards)
