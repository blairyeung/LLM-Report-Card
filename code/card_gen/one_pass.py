# one-shot card generation
# Generate one card based on a batch of questions and completions from the target model.
# Currently, it's only generating the bullet point format cards.
# The most naive way.
from typing import Dict, Optional

from core.card import GenerativeCard
from core.config import GPT_4_MODEL_NAME
from core.models import select_model, CostManager
from core.utils import ResourceManager, get_initial_criteria
from dataset.data import load_batches
from eval.predictive import PredictiveEvaluator


class OnePassMethod:
    hp: Dict

    experiment: str
    name: str
    dataset: str
    topic: str
    model: str

    initial_criteria: str

    rm: ResourceManager
    cm: CostManager

    evaluator_name: str

    def __init__(self, hp: Dict, rm: ResourceManager, cm: Optional[CostManager] = None):
        self.hp = hp
        self.dataset = hp["dataset"]
        self.topic = hp["topic"]
        self.model = hp["model"]
        self.evaluator_name = hp["evaluator_name"]
        self.initial_criteria = hp["initial_criteria"]
        self.rm = rm
        self.cm = cm

        self.rm.dump_dict("hyperparameters", hp)

    def main(self, batch_str: str):
        print(f"One-Pass Method Started on {self.topic} with model {self.model}!")
        system_prompt = self.rm.get_prompt("gen/one_pass/system")
        evaluator = select_model(self.evaluator_name, system_prompt, cm=self.cm)
        user_prompt = self.rm.get_prompt("gen/one_pass/user").format(
            topic=self.topic,
            batch=batch_str,
            criteria=self.initial_criteria,
        )
        # print(user_prompt)
        card = GenerativeCard(d=evaluator(user_prompt, temperature=1.0, use_json=True))

        self.rm.dump_dict("card", card.to_dict())
        self.rm.dump_dict(
            "info",
            {
                "conversation": evaluator.messages,
            },
        )
        print(
            f"One-Pass Method Finished on {self.topic} with model {self.model}! Cost: {self.cm.get_cost()}"
        )
        return card


def run_exp():
    topics = [
        "high_school_mathematics",
        "high_school_chemistry",
        "high_school_physics",
        "machine_learning",
    ]
    models = [
        # "gpt-4o",
        # "gpt-4-turbo",
        "Meta-Llama-3-8B-Instruct",
        "Meta-Llama-3-70B-Instruct",
        "Mistral-7B-Instruct-v0.2",
        "Mixtral-8x7B-Instruct-v0.1",
    ]
    cm = CostManager()
    for topic in topics:
        for model in models:
            hp = {
                "dataset": "mmlu",
                "topic": topic,
                "model": model,
                "evaluator_name": GPT_4_MODEL_NAME,
                "initial_criteria": get_initial_criteria(topic),
            }
            exp_name = "one_pass"
            run_name = f"{topic}/{model}/run"
            rm = ResourceManager(exp_name, run_name)
            method = OnePassMethod(hp, rm, cm=cm)
            train_batch = load_batches(
                f"datasets/mmlu", topic, model, "train", [40], False
            )[0]
            card = method.main(train_batch.get_train_str())
            testing_batch = load_batches(
                f"datasets/mmlu", topic, model, "test", [60], False
            )[0]
            pe = PredictiveEvaluator(topic, model, rm, "mmlu", cm=cm)
            pe.main("predictive_eval_log", testing_batch, card, num_times=3)
    print("Total cost:", cm.get_cost())


if __name__ == "__main__":
    run_exp()
