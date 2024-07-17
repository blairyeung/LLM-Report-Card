from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional, List, Any

from .config import *


def append_to_csv(csv_path: str, rows: List[List[Any]], header: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        rows.insert(0, header)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def sample_few_shots_fully_stochastic(
    reps, batch_len: int, k: int, random_seed: int = 42
):
    random.seed(random_seed)
    samples = []
    for _ in range(reps):
        random_index = random.sample(range(batch_len), k)
        samples.append(random_index)

    return samples


def sample_few_shots(num_samples: int, batch_len: int, k: int, random_seed: int = 42):
    """
    Sample k shots from each batch of batch_len, return a list of samples.
    """
    random.seed(random_seed)
    samples = []
    for i in range(num_samples):
        r = i % batch_len
        random_portion = random.sample(range(batch_len), k - 1)
        # make sure r not in random_portion
        while r in random_portion:
            random_portion = random.sample(range(batch_len), k - 1)
        indices = random_portion + [r]
        random.shuffle(indices)
        samples.append(indices)
    return samples


def get_initial_criteria(topic: str):
    with open("datasets/mmlu/0_mmlu_criteria.json") as f:
        initial_criteria = json.load(f)
    if topic in initial_criteria:
        if len(initial_criteria[topic]) == 0:
            raise ValueError(f"No initial criteria for topic: {topic}")
        return initial_criteria[topic]
        # for now
        # return []
    with open("datasets/anthropic-eval/0_criteria.json") as f:
        initial_criteria = json.load(f)
    if topic in initial_criteria:
        if len(initial_criteria[topic]) == 0:
            raise ValueError(f"No initial criteria for topic: {topic}")
        return initial_criteria[topic]
    with open("datasets/gsm8k/0_criteria.json") as f:
        initial_criteria = json.load(f)
    if topic in initial_criteria:
        if len(initial_criteria[topic]) == 0:
            raise ValueError(f"No initial criteria for topic: {topic}")
        return initial_criteria[topic]
    with open("datasets/openend/0_criteria.json") as f:
        initial_criteria = json.load(f)
    if topic in initial_criteria:
        if len(initial_criteria[topic]) == 0:
            raise ValueError(f"No initial criteria for topic: {topic}")
        return initial_criteria[topic]
    print("No initial criteria found for topic:", topic)

    return []


def get_choice_str(choice: int) -> str:
    # 0 -> A, 1 -> B, ...
    return chr(ord("A") + choice)


def get_choice_int(choice: str) -> int:
    # A -> 0, B -> 1, ...
    return ord(choice) - ord("A")


def read_all(filename: str) -> str:
    with open(filename) as f:
        r = f.read()
    return r


def get_folder_name(name: Optional[str]) -> str:
    if name is None:
        return datetime.now().strftime("%m-%d_%H-%M-%S")
    else:
        return datetime.now().strftime("%m-%d_%H-%M-%S") + "_" + name


class ResourceManager:
    output_folder_path: str
    prompt_map: Dict[str, str]
    executor: ThreadPoolExecutor  # for async writing

    def __init__(
        self,
        exp_name: str = "general",
        name: str = None,
        existing_output_path: str = None,
    ):
        self.prompt_map = {}
        self.executor = ThreadPoolExecutor(max_workers=5)

        self.output_folder_path = os.path.join(
            OUTPUTS_FOLDER_NAME, exp_name, get_folder_name(name)
        )
        if existing_output_path is None:
            # init output folder
            while os.path.exists(
                self.output_folder_path
            ):  # wait until the folder name is unique and available
                time.sleep(1)
                self.output_folder_path = os.path.join(
                    OUTPUTS_FOLDER_NAME, exp_name, get_folder_name(name)
                )
            os.makedirs(self.output_folder_path, exist_ok=False)
            # create cards dir
            os.makedirs(os.path.join(self.output_folder_path, "cards"), exist_ok=True)
            # copy prompts: we maintain a copy of prompts for each method
            shutil.copytree(
                PROMPTS_FOLDER_NAME,
                os.path.join(self.output_folder_path, PROMPTS_FOLDER_NAME),
                dirs_exist_ok=True,
            )
            # create README.md
            open(os.path.join(self.output_folder_path, "README.md"), "w").close()
        else:
            self.output_folder_path = existing_output_path

    def get_prompt(self, prompt_path: str) -> str:
        if prompt_path in self.prompt_map:
            return self.prompt_map[prompt_path]
        # not in map, read from file
        if not prompt_path.endswith(".txt"):
            prompt_path = prompt_path + ".txt"
        return read_all(
            os.path.join(self.output_folder_path, PROMPTS_FOLDER_NAME, prompt_path)
        )

    def dump_dict(self, filename: str, d: Dict):
        def task(f: str):
            try:
                if not f.endswith(".json"):
                    f += ".json"
                path = os.path.join(self.output_folder_path, f)
                # TODO: handle file already exists
                with open(path, "w") as f:
                    json.dump(d, f, indent=2, sort_keys=False)
            except Exception as e:
                print(e, file=sys.stderr)

        self.executor.submit(task, filename)

    def file_exists(self, filename: str, extension: str = ".json") -> bool:
        if not filename.endswith(extension):
            filename += extension
        return os.path.exists(os.path.join(self.output_folder_path, filename))

    def exists_and_load_json(self, filename: str):
        if self.file_exists(filename, extension="json"):
            if not filename.endswith(".json"):
                filename += ".json"
            with open(os.path.join(self.output_folder_path, filename)) as f:
                return json.load(f)
        return None

    def load_json(self, filename: str) -> Dict:
        if not filename.endswith(".json"):
            filename += ".json"
        with open(os.path.join(self.output_folder_path, filename)) as f:
            return json.load(f)

    def load_dict(self, filename: str) -> Dict:
        return self.load_json(filename)

    def shutdown(self):
        self.executor.shutdown()
