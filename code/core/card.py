from __future__ import annotations

import json
import random
import re
from collections import OrderedDict
from typing import List, Tuple, Dict, Iterable, Self, Optional, Union


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

        self.other_cards = []

    def sub_sample(self, m: int, k: int) -> List[Self]:
        """
        Randomly sample k sub-cards from this card, each with m criteria at most.
            - Each sub-card contains unique criteria.
            - All criteria in this card are included at least once in the sub-cards.
        Preconditions: m < len(self) = n
        """
        n = len(self.criteria)
        if k * m < n:
            raise ValueError(
                "k * m must be at least as large as the number of criteria"
            )

        # duplicate criteria if necessary
        total_criteria_needed = k * m
        duplicated_criteria = self.criteria.copy()
        # shuffle the duplicated criteria
        random.shuffle(duplicated_criteria)
        duplicated_criteria = (
            self.criteria * (total_criteria_needed // n)
            + self.criteria[: total_criteria_needed % n]
        )

        # create and distribute criteria to sub-cards
        sub_cards = [GenerativeCard() for _ in range(k)]
        for i, criterion in enumerate(duplicated_criteria):
            sub_cards[i % k].criteria.append(criterion)

        # sanity checks
        assert len(sub_cards) == k
        criteria_check_set = set()
        for sub_card in sub_cards:
            criteria_check_set.update(sub_card.get_criteria())
            assert len(sub_card) == m
            assert len(sub_card.get_criteria()) == len(set(sub_card.get_criteria()))
        assert len(criteria_check_set) == len(self.get_criteria())

        return sub_cards

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

    def get_summarization(self, i: int) -> Optional[str]:
        return self.criteria[i][1].get("summarization", None)

    def set_summarization(self, i: int, s: str):
        self.criteria[i][1]["summarization"] = s

    def __str__(self) -> str:
        r = ""
        for k, v in self.criteria:
            if isinstance(v, str):
                r += f"- {k}: {v}\n"
            elif isinstance(v, dict):
                r += f"- {k}: {v['overview']}\n"
                # r += f"- {k}:\n"
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

    def __getitem__(self, key):
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

    def criteria_count(self) -> int:
        all_criteria = [c[0] for c in self.criteria]
        # print(f'{all_criteria= }')
        for card in self.other_cards:
            criteria = card.criteria
            criteria = [c[0] for c in criteria]
            all_criteria += criteria

        all_criteria = list(set(all_criteria))

        # print(f'{all_criteria= }')
        return len(all_criteria)

    def words(self) -> int:
        self_str = str(self)
        others_str = [str(card) for card in self.other_cards]
        self_str += " ".join(others_str)

        # replace all non-word characters with spaces
        self_str = re.sub(r"\W+", " ", self_str)
        return len(re.findall(r"\w+", self_str))

    def get_all_cards(self) -> List[GenerativeCard]:
        return [self] + self.other_cards

    def get_all_card_str(self) -> str:
        all_cards = self.get_all_cards()
        counter = 1
        r = ""
        for card in all_cards:
            r += f"### Evaluation Card {counter}:\n"
            r += str(card) + "\n\n"
            counter += 1

        return r

    def __add__(self, other: GenerativeCard) -> GenerativeCard:
        if not isinstance(other, GenerativeCard):
            raise TypeError("Unsupported operand type for +")

        self.other_cards.append(other)
        return self
