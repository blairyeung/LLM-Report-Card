import csv
import json
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Tuple, Dict

import numpy as np
from tqdm import tqdm

from core.card import GenerativeCard
from core.models import select_model, CostManager
from core.utils import ResourceManager
from dataset.data import Batch


class LikertEvaluator:
    rm: ResourceManager
    cm: CostManager

    dataset: str
    topic: str
    card_format: str
    card_epoch: int
    teacher_model_name: str
    student_model_name: str

    excerpt_model_name: str
    evaluator_model_name: str

    def __init__(self, hp: Dict, rm: ResourceManager, cm: CostManager = None):
        """
        Example hp:
        hp = {
            "topic": "math",
            "student_model_name": "gpt-3.5-turbo",
            "excerpt_model_name": "gpt-3.5-turbo",
            "evaluator_model_name": "gpt-3.5-turbo",
        }
        """
        self.rm = rm
        self.cm = cm if cm is not None else CostManager()

        self.dataset = hp["dataset"]
        self.topic = hp["topic"]
        self.card_format = hp["card_format"]
        self.card_epoch = hp["card_epoch"]
        self.teacher_model_name = hp["teacher_model_name"]
        self.student_model_name = hp["student_model_name"]
        self.excerpt_model_name = hp["excerpt_model_name"]
        self.evaluator_model_name = hp["evaluator_model_name"]

    def main(
        self,
        version: Literal["lite", "full"],
        card: GenerativeCard,
        batch: Batch,
        num_times: int = 3,
        max_workers: int = 60,
    ):
        log = {
            "iterations": [],
        }
        for i in range(num_times):
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.helper, version, card, batch, index)
                    for index in range(len(batch))
                ]
                for future in tqdm(
                    as_completed(futures),
                    desc=f"Likert Evaluation Progress",
                    total=len(futures),
                ):
                    if future.exception() is not None:
                        print(future.exception())
                        continue
                    if future.result() is None:
                        continue
                    index, ratings, sub_log = future.result()
                    results.append(
                        {
                            "index": index,
                            "ratings": ratings,
                            **sub_log,
                        }
                    )
            log["iterations"].append(results)

        # dump results
        result_file = open(
            os.path.join(self.rm.output_folder_path, f"likert_{version}_results.csv"),
            "w",
        )
        writer = csv.writer(result_file)
        writer.writerow(
            [
                "topic",
                "student_model",
                "evaluator_model",
                "excerpt_model",
                "iteration",
                "index",
                "relevance",
                "informativeness",
                "ease_of_understanding",
            ]
        )
        ratings = np.zeros((num_times, len(batch), 3))
        for i, results in enumerate(log["iterations"]):
            for j, result in enumerate(results):
                ratings[i, j, :] = np.array(result["ratings"]).flatten()
                writer.writerow(
                    [
                        self.topic,
                        self.student_model_name,
                        self.evaluator_model_name,
                        self.excerpt_model_name,
                        i,
                        result["index"],
                        ratings[i, j, 0],
                        ratings[i, j, 1],
                        ratings[i, j, 2],
                    ]
                )
        result_file.close()

        # print aggregated results
        output_str = (
            f"eval_likert_{version} on {self.topic} and {self.student_model_name}\n"
            f"Relevance: {float(np.mean(ratings[:, :, 0]))} ± {float(np.std(ratings[:, :, 0]))} | mean = {np.mean(ratings[:, :, 0], axis=1).tolist()}, std = {np.std(ratings[:, :, 0], axis=1).tolist()}\n"
            f"Informativeness: {float(np.mean(ratings[:, :, 1]))} ± {float(np.std(ratings[:, :, 1]))} | mean = {np.mean(ratings[:, :, 1], axis=1).tolist()}, std = {np.std(ratings[:, :, 1], axis=1).tolist()}\n"
            f"Ease of Understanding: {float(np.mean(ratings[:, :, 2]))} ± {float(np.std(ratings[:, :, 2]))} | mean = {np.mean(ratings[:, :, 2], axis=1).tolist()}, std = {np.std(ratings[:, :, 2], axis=1).tolist()}\n"
            f"Cost so far: {self.cm.get_cost()}"
        )
        # print(output_str)
        self.rm.append_to_readme(f"```\n{output_str}\n```")

        # dump the log
        self.rm.dump_dict(f"eval_likert_{version}", log)
        return log

    def helper(
        self,
        version: Literal["lite", "full"],
        card: GenerativeCard,
        batch: Batch,
        index: int,
    ):
        """
        Evaluate the card based on multiple dimensions.
        Lite Version: Don't show the question and model's answer to the evaluator.
        Full Version: Show the question and model's answer to the evaluator.
        The excerpt is based on the relevance of sub-topics to the given question and student's response.
        """
        question = batch.get_question(index)
        model_response = batch.get_model_reasoning(index)
        qa = f"The Question:\n{question}\n\nStudent's Response to the Question:\n{model_response}"
        excerpt, excerpt_log = self.excerpt_card(
            card, question, index, model_response, version
        )

        system_prompt = self.rm.get_prompt(f"eval/likert/{version}/system")
        user_prompt = self.rm.get_prompt(f"eval/likert/{version}/user").format(
            topic=self.topic,
            qa=qa,
            excerpt=excerpt,
        )
        model = select_model(
            self.evaluator_model_name,
            system_prompt,
            self.cm,
        )
        json_obj = model(user_prompt, use_json=True, cache=True)
        if not isinstance(json_obj, dict):
            raise ValueError(f"Invalid JSON: {json_obj}")
        ratings = [
            json_obj["relevance"],
            json_obj["informativeness"],
            json_obj["clarity"],
        ]

        log = {
            "excerpt": excerpt_log,
            "conversation": model.messages,
        }
        return index, ratings, log

    def excerpt_card(
        self,
        card: GenerativeCard,
        question: str,
        question_index: int,
        model_response: str,
        version: str,
    ) -> Tuple[str, Dict]:
        excerpts_file_path = os.path.join(
            "excerpts",
            self.dataset,
            self.topic,
            "press",
            self.card_format,
            self.teacher_model_name,
            self.student_model_name,
            f"epoch_{self.card_epoch}-{version}-excerpts.json",
        )
        if os.path.exists(excerpts_file_path):
            with open(excerpts_file_path, "r") as f:
                excerpts = json.load(f)
                if question_index in excerpts:
                    return excerpts[question_index]["relevant_sub_topics"], {}
        excerpt_model = select_model(
            self.excerpt_model_name,
            self.rm.get_prompt("eval/likert/excerpt/system"),
            self.cm,
        )
        user_prompt = self.rm.get_prompt("eval/likert/excerpt/user").format(
            card=card,
            qa=question,
            response=model_response,
        )
        relevant_sub_topics = excerpt_model(user_prompt, use_json=True, cache=True)[
            "relevant_sub_topics"
        ]
        partial_card = card[relevant_sub_topics]
        log = {
            "conversation": excerpt_model.messages,
        }
        return str(partial_card), log
