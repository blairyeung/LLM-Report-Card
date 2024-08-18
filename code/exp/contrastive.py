import time
from typing import List, Dict, Literal, Tuple, Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib import itertools

from core.models import CostManager
from core.utils import ResourceManager
from dataset.data import load_batches
from eval.contrastive_2R2A import Contrastive2R2AEvaluator


def test(
    models: List[str],
    dataset: str,
    topic: str,
    exp_name: str,
    evaluators: List[str],
    cot: bool,
    mode: Literal["simplified", "full", "partial", "logit"],
    # full matrix or mirror the half matrix
    filling_mode: Literal["full", "mirror"],
    model_to_card: Optional[Dict[str, str]] = None,
    num_samples: int = -1,
    max_workers: int = 60,
    model_pairs: List[Tuple[str, str]] = None,
    few_shot: bool = False,
    paraphrased: bool = False,
):
    assert few_shot == (model_to_card is None)
    start_time = time.time()
    cm = CostManager()

    data = {}
    avg_accuracy, total = 0.0, 0
    dataset_acc = {model: float("nan") for model in models}

    # init model pairs
    if model_pairs is None:
        model_pairs = []
        if filling_mode == "full":
            model_pairs = itertools.product(models, models)
            # remove the repeated models
            model_pairs = [
                (model0, model1) for model0, model1 in model_pairs if model0 != model1
            ]
        else:
            for i in range(0, len(models)):
                for j in range(i + 1, len(models)):
                    model_pairs.append((models[i], models[j]))

    for model0, model1 in tqdm(model_pairs):
        print(f"Running contrastive evaluation for {model0} and {model1}")
        batch0 = load_batches(
            f"datasets/{dataset}/", topic, model0, "test", [60], False
        )[0]
        batch1 = load_batches(
            f"datasets/{dataset}/", topic, model1, "test", [60], False
        )[0]
        dataset_acc[model0] = batch0.get_accuracy()
        dataset_acc[model1] = batch1.get_accuracy()

        card0, card1 = None, None
        if model_to_card is not None:
            card0 = model_to_card[model0]
            card1 = model_to_card[model1]

        eval_method = Contrastive2R2AEvaluator(
            dataset,
            topic,
            (batch0, batch1),
            (model0, model1),
            (card0, card1),
            evaluators,
            ResourceManager(exp_name=exp_name, name=f"{model0}-{model1}"),
            cm,
            cot=cot,
            k_shots=3,
            max_workers=max_workers,
            csv_path=f"exp_results/{exp_name}.csv",
            few_shot=few_shot,
            paraphrase=paraphrased,
        )
        metrics = eval_method.main(num_samples=num_samples, mode=mode)[0]
        eval_method.shutdown()
        accuracy = metrics[0]

        data[(model0, model1)] = accuracy
        data[(model0, model0)] = 0.5  # diagonal
        data[(model1, model1)] = 0.5
        avg_accuracy += accuracy
        total += 1
        # time.sleep(60 * 1)
        print(f"Time taken: {time.time() - start_time:.2f}s")
        start_time = time.time()
    avg_accuracy /= total

    plot_heatmap(topic, models, data, dataset_acc, avg_accuracy, exp_name)

    print(data)
    print(cm.get_info_dict())
    return data


def plot_heatmap(topic, models, data, dataset_acc, avg_accuracy, exp_name):
    """
    Plot the heatmap of the full contrastive evaluation.
    :param topic: Topic name.
    :param models: List of model names.
    :param data: Dictionary of model pairs to accuracy.
    :param dataset_acc: Dictionary of model to dataset accuracy.
    :param avg_accuracy: Average accuracy.
    :param exp_name: Experiment name.
    """
    # Format model names with their dataset accuracy
    model_names_with_acc = {
        model: f"{model}\n({dataset_acc[model]:.2f})" for model in models
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(list(data.items()), columns=["index", "value"])
    df[["str1", "str2"]] = pd.DataFrame(
        [
            [model_names_with_acc[model] for model in pair]
            for pair in df["index"].tolist()
        ],
        index=df.index,
    )
    df.drop(columns=["index"], inplace=True)

    # Calculate differences and prepare formatted annotations
    def format_annotations(row):
        model1 = row["str1"].split("\n")[0]
        model2 = row["str2"].split("\n")[0]
        difference = abs(dataset_acc[model1] - dataset_acc[model2])
        return f"{row['value']:.2f}\n({difference:.2f})"

    df["formatted_value"] = df.apply(format_annotations, axis=1)

    # Pivot for color mapping and annotations
    pivot_values = df.pivot(index="str2", columns="str1", values="value")
    pivot_annotations = df.pivot(index="str2", columns="str1", values="formatted_value")

    # Reindex to ensure models are in the correct order
    order = [model_names_with_acc[model] for model in models]
    pivot_values = pivot_values.reindex(index=order, columns=order)
    pivot_annotations = pivot_annotations.reindex(index=order, columns=order)

    # Plot the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        pivot_values.astype(float),  # Ensure this is float for colormap calculation
        annot=pivot_annotations,  # Use formatted annotations
        cmap=sns.color_palette("Blues", as_cmap=True),
        fmt="",  # Set format to empty since annotations are strings
        annot_kws={"fontsize": 10, "va": "top", "ha": "center"},
    )
    plt.title(
        f"Topic: {topic}\n"
        f"Heatmap of 2C2A Contrastive Evaluation on Generative Cards\n"
        f"Avg Accuracy: {avg_accuracy:.2f}"
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=10, style="italic")
    plt.yticks(rotation=0, ha="right", fontsize=10, style="italic")
    plt.savefig(f"outputs/{exp_name}/heatmap.png")
    # plt.show()
