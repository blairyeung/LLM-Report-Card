import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset.data import Batch
from core.models import HFAPIModel, NousHermesModel
from core.utils import ResourceManager
from peripherals.elo import EloRating


class PreferentialEvaluator:
    topic: str
    model: str
    rm: ResourceManager
    evaluator_name: str

    def __init__(
        self,
        meta: str,
        topic: str,
        model: str,
        rm: ResourceManager,
        evaluator_name: str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    ):

        self.meta = meta
        self.topic = topic
        self.model = model
        self.rm = rm
        self.evaluator_name = evaluator_name

    def helper(self, batch: Batch, index: int, card1, card2):
        """
        Returns the preferred card (0 or 1), model, index
        """
        qa = batch.get_eval_preferential_str(index)
        system_prompt = self.rm.get_prompt("eval/preference/system").format(
            topic=self.topic,
        )

        if "Nous" in self.evaluator_name:
            model = NousHermesModel(system_prompt, self.evaluator_name)

        else:
            model = HFAPIModel(system_prompt, self.evaluator_name)

        user_prompt = self.rm.get_prompt("eval/preference/user").format(
            qa=qa,
            card1=str(card1),
            card2=str(card2),
        )
        try:
            json_obj = model(user_prompt, use_json=True)
            if isinstance(json_obj, str):
                raise ValueError(f"Response not parsable: {json_obj}")
            preferred_card = ord(json_obj["preferred_card"]) - ord("A")
            assert preferred_card in [0, 1]
        except Exception as e:
            print(e, file=sys.stderr)
            return None
        return preferred_card, model, index

    def eval(self, batch: Batch, card1, card2):
        info_dict = {"details": {}}

        rslt = self.helper(batch, 0, card1, card2)
        count, count_1, count_2 = 0, 0, 0
        with ThreadPoolExecutor(max_workers=int(len(batch) / 1)) as executor:
            futures = [
                executor.submit(self.helper, batch, i, card1, card2)
                for i in range(len(batch))
            ]
            for future in tqdm(as_completed(futures)):
                if future.result() is None:
                    continue
                preferred_card, model, index = future.result()
                count += 1
                if preferred_card == 0:
                    count_1 += 1
                elif preferred_card == 1:
                    count_2 += 1
                info_dict["details"][str(index)] = {
                    "preferred_card": chr(ord("A") + preferred_card),
                    "conversation": model.messages,
                }

        ratio_1 = count_1 / count
        ratio_2 = count_2 / count
        info_dict["preferred_ratio_A"] = ratio_1
        info_dict["preferred_ratio_B"] = ratio_2
        info_dict["A_win"] = count_1
        info_dict["B_win"] = count_2
        return info_dict, (ratio_1, ratio_2), (count_1, count_2)

    def main(
        self, name: str, batch: Batch, card1, card2, player_a: dict, player_b: dict
    ):
        info_dict = {
            "method": "preferential",
            "topic": self.topic,
            "model": self.model,
            "evaluator": self.evaluator_name,
            "iterations": [],
        }
        metrics = []
        for _ in range(1):
            info, metric, count = self.eval(batch, card1, card2)
            info_dict["iterations"].append(info)
            metrics.append(metric)

        ratios_A = [metric[0] for metric in metrics]
        ratios_A_mean = np.mean(ratios_A)
        ratios_A_sd = np.std(ratios_A)
        ratios_B = [metric[1] for metric in metrics]
        ratios_B_mean = np.mean(ratios_B)
        ratios_B_sd = np.std(ratios_B)
        info_dict["metrics"] = {
            "preferred_ratios_A": ratios_A,
            "preferred_ratios_A_mean": ratios_A_mean,
            "preferred_ratios_A_sd": ratios_A_sd,
            "preferred_ratios_B": ratios_B,
            "preferred_ratios_B_mean": ratios_B_mean,
            "preferred_ratios_B_sd": ratios_B_sd,
        }
        print(
            f"{name}\n"
            f"preferred_ratios_A_mean: {ratios_A_mean}, preferred_ratios_A_sd: {ratios_A_sd}, preferred_ratios_A: {ratios_A}\n"
            f"preferred_ratios_B_mean: {ratios_B_mean}, preferred_ratios_B_sd: {ratios_B_sd}, preferred_ratios_B: {ratios_B}"
        )

        # meta='mmlu', topic='high_school_physics', student_model='Mistral-7B-Instruct-v0.2'
        elo_rating = EloRating(
            meta=self.meta, topic=self.topic, student_model=self.model
        )
        match = (player_a, player_b, (count[0], count[1]))
        elo_rating.update_ratings_batch([match])

        self.rm.dump_dict(name, info_dict)

    def plot(self, prefix: str, epoch: int):
        output_folder = self.rm.output_folder_path
        epochs = list(range(1, epoch))
        ratios_A = []
        ratios_B = []
        for e in range(1, epoch):
            filename = output_folder + f"/{prefix}_epoch_{e}_preferential.json"
            with open(filename) as f:
                json_obj = json.load(f)
            json_obj = json_obj["metrics"]
            ratios_A.append(json_obj["preferred_ratios_A"])
            ratios_B.append(json_obj["preferred_ratios_B"])

        # Calculate means and standard deviations for each epoch
        mean_ratios_A = np.mean(ratios_A, axis=1)
        std_ratios_A = np.std(ratios_A, axis=1)
        mean_ratios_B = np.mean(ratios_B, axis=1)
        std_ratios_B = np.std(ratios_B, axis=1)

        # Plot each metric
        plt.figure(figsize=(10, 6))

        # Ratios A
        plt.plot(epochs, mean_ratios_A, label="Current Card", color="blue")
        plt.fill_between(
            epochs,
            np.array(mean_ratios_A) - np.array(std_ratios_A),
            np.array(mean_ratios_A) + np.array(std_ratios_A),
            color="blue",
            alpha=0.2,
        )
        for e, value in enumerate(mean_ratios_A):
            plt.text(
                epochs[e], value, f"{value:.2f}", ha="center", va="bottom", color="blue"
            )

        # Ratios B
        plt.plot(epochs, mean_ratios_B, label="Last Epoch Card", color="red")
        plt.fill_between(
            epochs,
            np.array(mean_ratios_B) - np.array(std_ratios_B),
            np.array(mean_ratios_B) + np.array(std_ratios_B),
            color="red",
            alpha=0.2,
        )
        for e, value in enumerate(mean_ratios_B):
            plt.text(
                epochs[e], value, f"{value:.2f}", ha="center", va="bottom", color="red"
            )

        # Adding legend and labels
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        plt.title("Preferential Evaluation Metrics Over Epochs")
        plt.suptitle(f"Topic: {self.topic}  Target Model: {self.model}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xticks(range(epoch))
        plt.savefig(
            os.path.join(
                self.rm.output_folder_path, f"{prefix}_preferential_metrics.png"
            )
        )
        # plt.show()
