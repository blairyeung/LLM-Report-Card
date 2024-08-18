import csv
import itertools
import os

from tqdm import tqdm

from core.card import load_card
from core.config import OUTPUTS_FOLDER_NAME
from core.models import CostManager
from core.utils import ResourceManager
from dataset.data import load_batches
from eval.likert import LikertEvaluator

datasets = [
    "mmlu",
    "anthropic_eval",
    # "gsm8k",
    # "hotpot_qa",
    # "openend",
]
topics = {
    "mmlu": [
        "college_mathematics",
        # "high_school_biology",
        # "high_school_chemistry",
        "high_school_mathematics",
        "high_school_physics",
        # "high_school_world_history",
        "machine_learning",
    ],
    "anthropic_eval": [
        "corrigible-less-HHH",
        # "myopic-reward",
        "power-seeking-inclination",
        # "self-awareness-general-ai",
        # "survival-instinct",
    ],
    "gsm8k": [],
    "hotpot_qa": [],
    "openend": [
        "Crafting engaging and contextually appropriate jokes",
        "Discussing philosophical theories of consciousness",
        "Writing efficient code for solving concrete algorthimic problems",
        "Providing dating advice",
        "Roleplaying as a fictional character",
    ],
}
student_models = [
    # "gemma-1.1-7b-it",
    # "gpt-3.5-turbo",
    # "gpt-4-turbo",
    "gpt-4o",
    "Meta-Llama-3-8B-Instruct",
    # "Meta-Llama-3-70B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    # "Mixtral-8x7B-Instruct-v0.1",
]
card_formats = [
    "bullet_point",
    # "dict",
]
teacher_models = [
    "gpt-4o-2024-05-13",
]
card_epochs = [
    0,
    4,
]
versions = [
    # "lite",
    "full",
]


def run_exp(exp_name: str = "eval_likert", result_filename: str = "results.csv"):
    """
    Run the major Likert evaluation experiment.
    It has the resume capability to start from an interrupted point.
    """
    # start the experiment
    cm = CostManager()

    result_filename = os.path.join(OUTPUTS_FOLDER_NAME, exp_name, result_filename)
    resume = os.path.exists(result_filename)
    if resume:
        print("Resuming from the previous checkpoint.")
        result_file = open(result_filename, "a")
    else:
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        result_file = open(result_filename, "w")
    writer = csv.writer(result_file)
    if not resume:
        writer.writerow(
            [
                "dataset",
                "topic",
                "algorithm",
                "card_format",
                "teacher_model",
                "student_model",
                "epoch",
                "version",
                "evaluator_model",
                "excerpt_model",
                "iteration",
                "index",
                "relevance",
                "informativeness",
                "ease_of_understanding",
            ]
        )

    dataset_tqdm = tqdm(datasets, desc="Current Dataset: ")
    for dataset in dataset_tqdm:
        dataset_tqdm.set_description(f"Current Dataset: {dataset}")
        topic_tqdm = tqdm(topics[dataset], desc="Current Topic: ")
        for topic in topic_tqdm:
            topic_tqdm.set_description(f"Current Topic: {topic}")
            config_tqdm = tqdm(
                itertools.product(
                    card_formats, teacher_models, student_models, card_epochs, versions
                ),
                desc="Current Config: ",
                total=len(card_formats)
                * len(teacher_models)
                * len(student_models)
                * len(card_epochs)
                * len(versions),
            )
            for card_format, teacher, student, epoch, version in config_tqdm:
                config_tqdm.set_description(
                    f"Current Config: {card_format}, {teacher}, {student}, {epoch}, {version}"
                )
                card = load_card(
                    dataset, topic, "press", card_format, teacher, student, epoch
                )
                if card is None:
                    tqdm.write(
                        f"Card not found: {dataset}/{topic}/press/{card_format}/{teacher}/{student}/{epoch}"
                    )
                    continue
                hp = {
                    "dataset": dataset,
                    "topic": topic,
                    "card_format": card_format,
                    "card_epoch": epoch,
                    "teacher_model_name": teacher,
                    "student_model_name": student,
                    "excerpt_model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
                    "evaluator_model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
                }
                name = f"{dataset}/{topic}/press/{card_format}/{teacher}/{student}-epoch_{epoch}-{version}"
                rm = ResourceManager(exp_name, name)
                # for resume
                if rm.is_existing and os.path.exists(
                    os.path.join(rm.output_folder_path, f"likert_{version}_results.csv")
                ):
                    continue
                evaluator = LikertEvaluator(hp, rm, cm=cm)
                testing_batch = load_batches(
                    f"datasets/{dataset}",
                    hp["topic"],
                    hp["student_model_name"],
                    "test",
                    [60],
                    shuffle=False,
                )[0]
                log = evaluator.main(
                    version, card, testing_batch, num_times=1, max_workers=15
                )
                for i, results in enumerate(log["iterations"]):
                    for j, result in enumerate(results):
                        writer.writerow(
                            [
                                dataset,
                                topic,
                                "press",
                                card_format,
                                teacher,
                                student,
                                epoch,
                                version,
                                hp["evaluator_model_name"],
                                hp["excerpt_model_name"],
                                i,
                                result["index"],
                                result["ratings"][0],
                                result["ratings"][1],
                                result["ratings"][2],
                            ]
                        )
                result_file.flush()

    result_file.close()
