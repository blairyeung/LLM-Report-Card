import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import jsonlines
import numpy as np
import pandas as pd
# sns
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.card import GenerativeCard
from core.config import *
from core.data import load_batches, Batch
from core.models import CostManager, select_model
from core.utils import ResourceManager


def compute_baseline_accuracy(topic, batches, models):
    # batch1 = load_mmlu_batches("datasets/mmlu/", topic, model1, "test", [60], False)[0]
    # batch2 = load_mmlu_batches("datasets/mmlu/", topic, model2, "test", [60], False)[0]

    batch1, batch2 = batches
    model1, model2 = models

    batch_1_correct = [
        batch1.get_true_answer(i) == batch1.get_model_answer(i)
        for i in range(len(batch1))
    ]
    batch_2_correct = [
        batch2.get_true_answer(i) == batch2.get_model_answer(i)
        for i in range(len(batch2))
    ]

    correct = 0
    agreement = 0

    for i in range(len(batch_1_correct)):
        if batch_1_correct[i] == batch_2_correct[i]:
            # if same, 50% change correct
            correct += 0.5
            agreement += 1
        else:
            if batch_1_correct[i]:
                correct += 1

    # print(f'Accuracy for {model1} and {model2} on {topic} is:')
    # print(model1, sum(batch_1_correct) / len(batch_1_correct) * 100)
    # print(model2, sum(batch_2_correct) / len(batch_2_correct) * 100)
    # print('Agreement:', agreement / len(batch_1_correct) * 100)
    # print('Actual accuracy:', correct / len(batch_1_correct) * 100)

    oracle_accuracy = max(
        correct / len(batch_1_correct), 1 - correct / len(batch_1_correct)
    )
    oracle_str = f"{oracle_accuracy:.2f}%"
    # 2 decimal, add %
    model_1_accuracy = sum(batch_1_correct) / len(batch_1_correct) * 100
    model_2_accuracy = sum(batch_2_correct) / len(batch_2_correct) * 100
    agreement = agreement / len(batch_1_correct) * 100
    return oracle_accuracy, oracle_str, model_1_accuracy, model_2_accuracy, agreement


def plot_same_matrix(models, topic):
    matrix = np.zeros((len(models), len(models)))
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            batch1 = load_batches("datasets/mmlu/", topic, model1, "test", [60], False)[
                0
            ]
            batch2 = load_batches("datasets/mmlu/", topic, model2, "test", [60], False)[
                0
            ]

            batch_same = [batch1[i][3] == batch2[i][3] for i in (range(len(batch1)))]

            matrix[i, j] = sum(batch_same)

    # reflect the matrix
    matrix = matrix + matrix.T

    # sns plot heat map
    sns.heatmap(matrix, annot=True, xticklabels=models, yticklabels=models)
    # labels each block with value

    plt.show()


if __name__ == "__main__":
    meta = "mmlu"
    topic = "self-awareness-general-ai"
    topic = "power-seeking-inclination"
    topic = "high_school_mathematics"

    models = [
        "Mistral-7B-Instruct-v0.2",
        "Mixtral-8x7B-Instruct-v0.1",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf",
        "gpt-3.5-turbo-1106",
        "Meta-Llama-3-70B-Instruct",
        GPT_4_MODEL_NAME,
    ]

    # plot_same_matrix(models, topic)

    model1 = models[0]
    model2 = models[-1]

    batch1 = load_batches(f"datasets/{meta}", topic, model1, "test", [60], False)[0]
    batch2 = load_batches(f"datasets/{meta}", topic, model2, "test", [60], False)[0]

    rslt = compute_baseline_accuracy(topic, [batch1, batch2], [model1, model2])

    print(rslt)

    model2 = models[1]
    model1 = models[4]

    batch1 = load_batches(f"datasets/{meta}", topic, model1, "test", [60], False)[0]
    batch2 = load_batches(f"datasets/{meta}", topic, model2, "test", [60], False)[0]

    rslt = compute_baseline_accuracy(topic, [batch1, batch2], [model1, model2])

    print(rslt)

    # batch_1_correct = [b[2] == b[3] for b in batch1]
    # batch_2_correct = [b[2] == b[3] for b in batch2]

    # correct = 0
    # diff = 0

    # for i in range(len(batch_1_correct)):
    #     if batch_1_correct[i] == batch_2_correct[i]:
    #         # if same, 50% change correct
    #         correct += 0.5
    #     else:
    #         diff += 1
    #         if batch_1_correct[i]:
    #             correct += 1

    # print(f'Accuracy for {model1} and {model2} on {topic} is:')
    # print(model1, sum(batch_1_correct) / len(batch_1_correct) * 100)
    # print(model2, sum(batch_2_correct) / len(batch_2_correct) * 100)

    # print('Actual accuracy:', correct / len(batch_1_correct) * 100)
    # print('Different:', diff)
    # # print('Theoretical accuracy:',(1 - 0.5 * )) * 100)

    # batch_same = [batch1[i][3] == batch2[i][3] for i in (range(len(batch1)))]

    # print('Same:', sum(batch_same))
