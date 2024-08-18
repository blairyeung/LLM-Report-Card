from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

from core.config import *


def append_to_csv(csv_path: str, rows: List[List[Any]], header: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        rows.insert(0, header)
        f = open(csv_path, "w", newline="")
    else:
        f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    writer.writerows(rows)
    f.close()


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


def find_next_available_folder(folder_name: str) -> str:
    """
    Find the next available folder name by appending -{i} to the folder name.
    """
    # list all files in the folder
    os.makedirs(os.path.dirname(folder_name), exist_ok=True)
    files = os.listdir(os.path.dirname(folder_name))
    # filter out only the folders starting with the folder_name
    files = sorted(
        [f for f in files if f.startswith(os.path.basename(folder_name))], reverse=True
    )
    i = 0
    # try to get the number after the last folder
    for file in files:
        try:
            i = int(file[len(os.path.basename(folder_name)) + 1 :])
            break
        except ValueError:
            continue
    if os.path.isdir(folder_name):
        return f"{folder_name}{i + 1}"
    else:
        return f"{folder_name}-{i + 1}"


class ResourceManager:
    """
    A class to manage resources for a single run of an experiment.
    It manages a directory for the run, and provides methods to read/write files.

    For writing JSONs, it has an async executor to write files in parallel.
    """

    exp_name: str
    output_folder_path: str
    prompt_map: Dict[str, str]
    executor: ThreadPoolExecutor  # for async writing
    is_existing: bool  # whether the folder is existed before creating this rm

    def __init__(
        self,
        exp_name: str,
        name: str = "run",
    ):
        """
        Create a new ResourceManager for the experiment with the given name.
        - When the name is not provided, it will use "run" as the default name.
        - When the name is an existing folder, it will use the folder as the output folder.
        - When the name is not an existing folder, it will create a new folder and copy the prompts.
        """
        self.exp_name = exp_name
        self.prompt_map = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        # assume existing
        self.output_folder_path = os.path.join(OUTPUTS_FOLDER_NAME, exp_name, name)
        # if not existing, create a new folder based on rules
        self.is_existing = os.path.exists(self.output_folder_path)
        if not self.is_existing:
            # self.output_folder_path = find_next_available_folder(
            #     os.path.join(OUTPUTS_FOLDER_NAME, exp_name, name)
            # )
            os.makedirs(self.output_folder_path, exist_ok=False)
            # copy prompts: we maintain a copy of prompts for each method
            shutil.copytree(
                PROMPTS_FOLDER_NAME,
                os.path.join(self.output_folder_path, PROMPTS_FOLDER_NAME),
                dirs_exist_ok=True,
            )
            # create README.md
            open(os.path.join(self.output_folder_path, "README.md"), "w").close()

    def append_to_readme(self, text: str):
        with open(os.path.join(self.output_folder_path, "README.md"), "a") as f:
            f.write(text)

    def get_prompt(self, prompt_path: str) -> str:
        """
        Get the prompt content from the prompt_path.
        """
        if prompt_path in self.prompt_map:
            return self.prompt_map[prompt_path]
        # not in map, read from file
        if not prompt_path.endswith(".txt"):
            prompt_path = prompt_path + ".txt"
        return read_all(
            os.path.join(self.output_folder_path, PROMPTS_FOLDER_NAME, prompt_path)
        )

    def dump_dict(self, filename: str, d: Dict):
        """
        Dump a dictionary to a JSON file in the output folder.
        """

        def task(f: str):
            try:
                if not f.endswith(".json"):
                    f += ".json"
                path = os.path.join(self.output_folder_path, f)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # TODO: handle file already exists
                with open(path, "w") as f:
                    json.dump(d, f, indent=2, sort_keys=False)
            except Exception as e:
                print(e, file=sys.stderr)

        self.executor.submit(task, filename)

    def file_exists(self, filename: str, extension: str = ".json") -> bool:
        """
        Check if the file exists in the output folder.
        """
        if not filename.endswith(extension):
            filename += extension
        return os.path.exists(os.path.join(self.output_folder_path, filename))

    def exists_and_load_json(self, filename: str):
        """
        Check if the file exists in the output folder and load it if it exists
        """
        if self.file_exists(filename, extension="json"):
            if not filename.endswith(".json"):
                filename += ".json"
            with open(os.path.join(self.output_folder_path, filename)) as f:
                return json.load(f)
        return None

    def load_json(self, filename: str) -> Dict:
        """
        Load a JSON file from the output folder.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        with open(os.path.join(self.output_folder_path, filename)) as f:
            return json.load(f)

    def load_dict(self, filename: str) -> Dict:
        """
        Load a dictionary from the output folder.
        """
        return self.load_json(filename)

    def shutdown(self):
        """
        Shutdown the async executor.
        Should be called at the end of the experiment.
        """
        self.executor.shutdown()
