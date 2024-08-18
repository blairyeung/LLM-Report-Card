from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Self, Union

import jsonlines

from core.utils import *


class Batch(ABC):

    @abstractmethod
    def get_train_str(self, include_model_answer: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_eval_predictive_str(self, index: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_eval_preferential_str(self, index: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_accuracy(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_oracle(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_question(self, index: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_true_answer(self, index: int) -> Any:
        """
        Get the ground truth answer for the question at the given index.
        The true answer could be an integer representing the choice, or a string.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_answer(self, index: int) -> Any:
        """
        Get the model's answer for the question at the given index.
        The model answer could be an integer representing the choice, or a string
        Its format should correspond to the ground truth answer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_reasoning(self, index: int) -> str:
        """
        Get the model's reasoning for the question at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_paraphrased_reasoning(self, index: int) -> str:
        """
        Get the model's reasoning for the question at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: Batch):
        raise NotImplementedError


class OpenEndBatch(Batch):
    model: str
    # (question, completion)
    raw: List[Tuple[str, str]]

    def __init__(self, raw: List[Tuple[str, str]], model: str):
        self.model = model
        self.raw = raw

    def get_train_str(self, include_model_answer: bool = True) -> str:
        s = ""
        for i in range(len(self)):
            s += f"Question:\n{self.get_question(i)}\n"
            s += f"Student's Answer: {self.get_model_reasoning(i)}\n"
            s += f"\n{SMALL_SEPARATOR}\n"
        return s

    def get_eval_predictive_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        return s

    def get_eval_preferential_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        s += f"Student's Answer: {self.get_model_reasoning(index)}\n"
        return s

    def get_question(self, index: int) -> str:
        return self.raw[index][0]

    def get_model_reasoning(self, index: int) -> str:
        return self.raw[index][1]

    def get_true_answer(self, index: int) -> Any:
        return None

    def get_model_answer(self, index: int) -> Any:
        return None

    def get_accuracy(self) -> float:
        return float("nan")

    def get_oracle(self) -> float:
        return float("nan")

    def shuffle(self):
        random.shuffle(self.raw)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, item):
        return self.raw[item]

    def __iter__(self):
        return iter(self.raw)

    def __add__(self, other) -> Self:
        assert isinstance(other, OpenEndBatch)
        return OpenEndBatch(self.raw + other.raw, self.model)


class MMLUBatch(Batch):
    model: str
    # (question, choices, true answer, model answer, model reasoning)
    raw: List[Tuple[str, List[str], int, int, str]]

    def __init__(self, raw: List[Tuple[str, List[str], int, int, str]], model: str):
        self.model = model
        self.raw = raw

    def get_train_str(self, include_model_answer: bool = True) -> str:
        s = ""
        for i in range(len(self)):
            s += f"Question:\n{self.get_question(i)}\n"
            s += f"Ground Truth Answer: {get_choice_str(self.get_true_answer(i))}\n"
            if include_model_answer:
                s += f"Student's Answer: {get_choice_str(self.get_model_answer(i))}\n"
                s += f"Student's Reasoning:\n{self.get_model_reasoning(i)}\n"
            s += f"\n{SMALL_SEPARATOR}\n"
        return s

    def get_question_str(self, include_ground_truth=True) -> str:
        # Question only
        s = ""
        for i in range(len(self)):
            s += f"Question {i + 1}:\n{self.get_question(i)}\n"
            if include_ground_truth:
                s += f"Ground Truth Answer: {get_choice_str(self.get_true_answer(i))}\n"
            place_holder = f"student_answer_{i}"
            s += f"Students' Answer: <{place_holder}>\n"
            s += f"\n{SMALL_SEPARATOR}\n"

        return s

    def get_student_completions(self) -> List[dict]:
        completions = []
        for i in range(len(self)):
            if self.get_model_answer(i) not in range(len(self.get_choices(i))):
                answer_str = "None of the above."
            else:
                answer_str = f"{get_choice_str(self.get_model_answer(i))}. {self.get_choices(i)[self.get_model_answer(i)]}"
            completions.append(
                {"choice": answer_str, "completion": self.get_model_reasoning(i)}
            )

        return completions

    def get_eval_predictive_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        return s

    def get_eval_preferential_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        s += f"Student's Answer: {get_choice_str(self.get_model_answer(index))}\n"
        s += f"Student's Reasoning:\n{self.get_model_reasoning(index)}\n"
        return s

    def get_accuracy(self) -> float:
        correct = 0
        for i in range(len(self)):
            if self.get_true_answer(i) == self.get_model_answer(i):
                correct += 1
        return correct / len(self)

    def get_oracle(self) -> float:
        acc = self.get_accuracy()
        return max(acc, 1 - acc)

    def get_choices(self, index: int) -> List[str]:
        return self.raw[index][1]

    def get_question(self, index: int) -> str:
        s = f"{self.raw[index][0]}\n\n"
        for i, choice in enumerate(self.get_choices(index)):
            s += f"{get_choice_str(i)}. {choice}\n"
        return s

    def get_true_answer(self, index: int) -> int:
        return self.raw[index][2]

    def get_model_answer(self, index: int) -> int:
        return self.raw[index][3]

    def get_model_reasoning(self, index: int) -> str:
        return self.raw[index][4]

    def get_paraphrased_reasoning(self, index) -> str:
        return self.raw[index][5]

    def shuffle(self):
        random.shuffle(self.raw)

    def __len__(self):
        return len(self.raw)

    def __getitem__(
        self, item
    ) -> Union[MMLUBatch, Tuple[str, List[str], int, int, str]]:
        if isinstance(item, slice):
            return MMLUBatch(self.raw[item], self.model)
        elif isinstance(item, int):
            return self.raw[item]
        else:
            raise TypeError(f"Invalid type for item: {type(item)}")

    def __iter__(self):
        return iter(self.raw)

    def __add__(self, other) -> Self:
        assert isinstance(other, MMLUBatch)
        return MMLUBatch(self.raw + other.raw, self.model)


class AnthropicBatch(Batch):
    model: str
    #  (question, true answer, false answer, model answer, model reasoning)
    raw: List[Tuple[str, str, str, str, str]]

    def __init__(self, raw: List[Tuple[str, str, str, str, str]], model: str):
        self.model = model
        self.raw = raw

    def get_train_str(self, include_model_answer: bool = True) -> str:
        s = ""
        for i in range(len(self)):
            s += f"Question:\n{self.get_question(i)}\n\n"
            s += f"Proper Choice: {self.get_true_answer(i)}\n"
            s += f"Improper Choice: {self.get_false_answer(i)}\n"
            if include_model_answer:
                s += f"Student's Reasoning:\n{self.get_model_reasoning(i)}\n"
                s += f"Student's Choice: {self.get_model_answer(i)}\n"
            s += f"\n{SMALL_SEPARATOR}\n"
        return s

    def get_eval_predictive_str(self, index: int) -> str:
        s = f"Question:\n{self.get_question(index)}\n\n"
        s += f"Proper (Ground Truth) Choice: {self.get_true_answer(index)}\n"
        s += f"Improper Choice: {self.get_false_answer(index)}\n"
        return s

    def get_eval_preferential_str(self, index: int) -> str:
        s = f"Question:\n{self.get_question(index)}\n\n"
        s += f"Proper (Ground Truth) Choice: {self.get_true_answer(index)}\n"
        s += f"Improper Choice: {self.get_false_answer(index)}\n"
        s += f"Student's Reasoning:\n{self.get_model_reasoning(index)}\n"
        s += f"Student's Choice: {self.get_model_answer(index)}\n"
        return s

    def get_accuracy(self) -> float:
        correct = 0
        for i in range(len(self)):
            if self.get_model_answer(i) in self.get_true_answer(i):
                correct += 1
        return correct / len(self.raw)

    def get_oracle(self) -> float:
        acc = self.get_accuracy()
        return max(acc, 1 - acc)

    def shuffle(self):
        random.shuffle(self.raw)

    def get_question(self, index: int) -> str:
        return self.raw[index][0][:-9]

    def get_true_answer(self, index: int) -> str:
        return self.raw[index][1]

    def get_false_answer(self, index: int) -> str:
        return self.raw[index][2]

    def get_model_answer(self, index: int) -> str:
        return self.raw[index][3]

    def get_model_reasoning(self, index: int) -> str:
        return self.raw[index][4]

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return AnthropicBatch(self.raw[item], self.model)
        elif isinstance(item, int):
            return AnthropicBatch([self.raw[item]], self.model)
        else:
            raise TypeError(f"Invalid type for item: {type(item)}")

    def __iter__(self):
        return iter(self.raw)

    def __add__(self, other):
        assert isinstance(other, AnthropicBatch)
        return AnthropicBatch(self.raw + other.raw, self.model)


class NoChoiceBatch(Batch):
    model: str
    # (question, solution (true reasoning),true answer, model reasoning, is model correct)
    raw: List[Tuple[str, str, str, bool]]

    def __init__(self, raw: List[Tuple[str, str, str, bool]], model: str):
        self.model = model
        self.raw = raw

    def get_train_str(self, include_model_answer: bool = True) -> str:
        s = ""
        for i in range(len(self)):
            s += f"Question:\n{self.get_question(i)}\n"
            s += f"Ground Truth Answer: {self.get_true_answer(i)}\n"
            if include_model_answer:
                s += f"Student's Answer: {self.get_model_reasoning(i)}\n"
            s += f"\n{SMALL_SEPARATOR}\n"
        return s

    def get_eval_predictive_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        return s

    def get_eval_preferential_str(self, index: int) -> str:
        q = self.get_question(index)
        s = f"Question:\n{q}\n\n"
        s += f"Student's Answer: {self.get_model_reasoning(index)}\n"
        return s

    def get_accuracy(self) -> float:
        correct = 0
        for i in range(len(self)):
            if self.is_model_correct(i):
                correct += 1
        return correct / len(self)

    def get_oracle(self) -> float:
        acc = self.get_accuracy()
        return max(acc, 1 - acc)

    def get_question(self, index: int) -> str:
        return self.raw[index][0]

    def get_true_answer(self, index: int) -> str:
        return self.raw[index][1]

    def get_model_answer(self, index: int) -> Any:
        """
        Since this kind of batch has no choices, so there's not a single answer.
        But you can use self.get_model_reasoning(index) to get the model's reasoning.
        The reasoning is the model's answer.
        """
        return None

    def get_model_reasoning(self, index: int) -> str:
        return self.raw[index][2]

    def is_model_correct(self, index: int) -> bool:
        return self.raw[index][3]

    def shuffle(self):
        random.shuffle(self.raw)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return NoChoiceBatch(self.raw[item], self.model)
        elif isinstance(item, int):
            return NoChoiceBatch([self.raw[item]], self.model)
        else:
            raise TypeError(f"Invalid type for item: {type(item)}")

    def __iter__(self):
        return iter(self.raw)

    def __add__(self, other) -> Self:
        assert isinstance(other, NoChoiceBatch)
        return NoChoiceBatch(self.raw + other.raw, self.model)


def load_batches(
    base_folder: str,
    topic: str,
    model: str,
    partition: str,
    batch_nums: List[int],
    shuffle: bool = False,
) -> List[Batch]:
    if "anthropic" in base_folder:
        return load_anthropic_batches(
            base_folder, topic, model, partition, batch_nums, shuffle
        )
    elif "mmlu" in base_folder:
        return load_mmlu_batches(
            base_folder, topic, model, partition, batch_nums, shuffle
        )
    elif "gsm8k" in base_folder or "hotpot_qa" in base_folder:
        return load_no_choice_batches(
            base_folder, model, partition, batch_nums, shuffle
        )
    elif "openend" in base_folder:
        return load_openend_batches(
            base_folder, topic, model, partition, batch_nums, shuffle
        )
    else:
        print(base_folder)
        raise NotImplementedError


def load_mmlu_batches(
    base_folder: str,
    topic: str,
    model: str,
    partition: str,
    batch_nums: List[int],
    shuffle: bool = False,
) -> List[MMLUBatch]:
    """
    Load MMLU batches from disk.
    :param base_folder: the base folder where the data is stored
    :param topic: the topic of the data
    :param model: the model used to generate the data
    :param partition: the partition of the data ("train" or "test")
    :param batch_nums: the batch numbers to load
    :param shuffle: whether to shuffle the batches after loading
    """
    p = os.path.join(base_folder, topic, f"{model}-{partition}.jsonl")
    batches = []
    with jsonlines.open(p) as reader:
        raw_lines = list(reader)
    # lines = [(line["question"], line["choices"], line["answer"],
    #   line[model]["answer"], line[model]["reasoning"]) for line in raw_lines]
    lines = [
        (
            line["question"],
            line["choices"],
            line["answer"],
            line["model_answer"],
            line["completion"],
            line.get("paraphrased", None),
        )
        for line in raw_lines
    ]
    if shuffle:
        random.shuffle(lines)
    s = 0
    for batch_num in batch_nums:
        e = s + batch_num
        batches.append(MMLUBatch(lines[s:e], model))
        s = e
    return batches


def load_anthropic_batches(
    base_folder: str,
    topic: str,
    model: str,
    partition: str,
    batch_nums: List[int],
    shuffle: bool = False,
) -> List[AnthropicBatch]:
    p = os.path.join(base_folder, topic, f"{model}-{partition}.jsonl")
    batches = []
    with jsonlines.open(p) as reader:
        raw_lines = list(reader)
    lines = [
        (
            line["question"],
            line["answer_matching_behavior"],
            line["answer_not_matching_behavior"],
            line["model_answer"],
            line["completion"],
        )
        for line in raw_lines
    ]
    if shuffle:
        random.shuffle(lines)
    s = 0
    for batch_num in batch_nums:
        e = s + batch_num
        batches.append(AnthropicBatch(lines[s:e], model))
        s = e
    return batches


def load_no_choice_batches(
    base_folder: str,
    model: str,
    partition: str,
    batch_nums: List[int],
    shuffle: bool = False,
) -> List[NoChoiceBatch]:
    p = os.path.join(base_folder, f"{model}-{partition}.jsonl")
    batches = []
    with jsonlines.open(p) as reader:
        raw_lines = list(reader)
    lines = [
        (line["question"], line["answer"], line["completion"], line["correctness"])
        for line in raw_lines
    ]
    if shuffle:
        random.shuffle(lines)
    s = 0
    for batch_num in batch_nums:
        e = s + batch_num
        batches.append(NoChoiceBatch(lines[s:e], model))
        s = e
    return batches


def load_openend_batches(
    base_folder, topic, model, partition, batch_nums, shuffle=False
):
    p = os.path.join(base_folder, topic, f"{model}-{partition}.jsonl")
    batches = []
    with jsonlines.open(p) as reader:
        raw_lines = list(reader)
    lines = [(line["question"], line["completion"]) for line in raw_lines]
    if shuffle:
        random.shuffle(lines)
    s = 0
    for batch_num in batch_nums:
        e = s + batch_num
        batches.append(OpenEndBatch(lines[s:e], model))
        s = e

    return batches


if __name__ == "__main__":
    # try to load mmlu batch
    batch = load_batches(
        "datasets/anthropic_eval",
        "corrigible-less-HHH",
        "gpt-4o",
        "test",
        [60],
        shuffle=False,
    )[0]

    # compute the accuracy
    print(batch.get_accuracy())
