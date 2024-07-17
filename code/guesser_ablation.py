import os

from eval.eval_contrastive_full import test as test_full

# change working dir to the content root
content_root = "/home/scott/llmeval/LLM-Eval-NIPS"
os.chdir(content_root)
print("Current Working Directory: ", os.getcwd())

# subset of topics and models used for ablation
topics = [
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "machine_learning",
]
models = [
    "gpt-4o",
    # "gpt-4-turbo",
    "Meta-Llama-3-70B-Instruct",
    # "gpt-3.5-turbo",
    "Meta-Llama-3-8B-Instruct",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    # "gemma-1.1-7b-it",
]

guessers = [
    # "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


def get_model_cards(topic: str, card_format: str, epoch: int):
    topic_cards = {
        "high_school_mathematics": {
            "dict": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gemma-1.1-7b-it/05-12_19-38-06_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-3.5-turbo/05-12_19-38-06_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-4-turbo/05-12_19-38-06_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/gpt-4o/05-18_11-03-15_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-12_19-38-06_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-12_19-38-06_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-12_19-38-06_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_mathematics/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-12_19-33-08_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
            "bullet_point": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/gemma-1.1-7b-it/05-13_02-39-38_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/gpt-3.5-turbo/05-13_02-39-38_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/gpt-4-turbo/05-13_02-39-38_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/gpt-4o/05-20_10-27-51_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/Meta-Llama-3-8B-Instruct/05-13_02-39-38_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/Meta-Llama-3-70B-Instruct/05-13_02-39-38_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/Mistral-7B-Instruct-v0.2/05-13_10-35-17_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_mathematics/prog-reg/bullet_point/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_10-35-17_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
        },
        "high_school_physics": {
            "dict": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_12-37-20_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_12-37-20_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-4-turbo/05-13_12-37-20_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/gpt-4o/05-18_11-05-33_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_12-37-20_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_12-37-20_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-13_12-24-57_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_physics/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_12-24-57_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
            "bullet_point": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/gemma-1.1-7b-it/05-13_12-34-07_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/gpt-3.5-turbo/05-13_12-34-07_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/gpt-4-turbo/05-13_12-34-07_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/gpt-4o/05-20_10-25-44_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/Meta-Llama-3-8B-Instruct/05-13_12-34-07_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/Meta-Llama-3-70B-Instruct/05-13_12-34-07_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/Mistral-7B-Instruct-v0.2/05-13_12-21-13_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_physics/prog-reg/bullet_point/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_12-21-13_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
        },
        "high_school_chemistry": {
            "dict": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_22-53-43_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_22-53-43_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-4-turbo/05-13_22-53-43_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/gpt-4o/05-18_11-08-10_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_22-53-43_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_22-53-43_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-13_22-53-43_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_chemistry/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_22-53-43_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
            "bullet_point": {
                "gemma-1.1-7b-it": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/gemma-1.1-7b-it/05-20_19-27-53_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/gpt-3.5-turbo/05-20_19-27-53_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/gpt-4-turbo/05-20_19-27-53_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/gpt-4o/05-20_19-27-53_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/Meta-Llama-3-8B-Instruct/05-20_19-27-53_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/Meta-Llama-3-70B-Instruct/05-20_19-27-53_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/Mistral-7B-Instruct-v0.2/05-20_19-27-52_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/high_school_chemistry/prog-reg/bullet_point/gpt/Mixtral-8x7B-Instruct-v0.1/05-20_19-27-52_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
        },
        "machine_learning": {
            "dict": {
                "gemma-1.1-7b-it": f"outputs/generative/machine_learning/prog-reg/dict/gpt/gemma-1.1-7b-it/05-13_22-50-27_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/machine_learning/prog-reg/dict/gpt/gpt-3.5-turbo/05-13_22-50-27_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/machine_learning/prog-reg/dict/gpt/gpt-4-turbo/05-13_22-50-27_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/machine_learning/prog-reg/dict/gpt/gpt-4o/05-18_23-18-00_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/machine_learning/prog-reg/dict/gpt/Meta-Llama-3-8B-Instruct/05-13_22-50-27_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/machine_learning/prog-reg/dict/gpt/Meta-Llama-3-70B-Instruct/05-13_22-50-27_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/machine_learning/prog-reg/dict/gpt/Mistral-7B-Instruct-v0.2/05-13_22-50-27_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/machine_learning/prog-reg/dict/gpt/Mixtral-8x7B-Instruct-v0.1/05-13_22-50-27_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
            "bullet_point": {
                "gemma-1.1-7b-it": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/gemma-1.1-7b-it/05-20_19-32-00_gemma-1.1-7b-it_main/cards/epoch_{epoch}_card.json",
                "gpt-3.5-turbo": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/gpt-3.5-turbo/05-20_19-32-00_gpt-3.5-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4-turbo": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/gpt-4-turbo/05-20_19-32-00_gpt-4-turbo_main/cards/epoch_{epoch}_card.json",
                "gpt-4o": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/gpt-4o/05-20_19-32-00_gpt-4o_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-8B-Instruct": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/Meta-Llama-3-8B-Instruct/05-20_19-32-00_Meta-Llama-3-8B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Meta-Llama-3-70B-Instruct": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/Meta-Llama-3-70B-Instruct/05-20_19-32-00_Meta-Llama-3-70B-Instruct_main/cards/epoch_{epoch}_card.json",
                "Mistral-7B-Instruct-v0.2": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/Mistral-7B-Instruct-v0.2/05-20_19-32-00_Mistral-7B-Instruct-v0.2_main/cards/epoch_{epoch}_card.json",
                "Mixtral-8x7B-Instruct-v0.1": f"outputs/generative/machine_learning/prog-reg/bullet_point/gpt/Mixtral-8x7B-Instruct-v0.1/05-20_19-32-00_Mixtral-8x7B-Instruct-v0.1_main/cards/epoch_{epoch}_card.json",
            },
        },
    }
    return topic_cards[topic][card_format]


def run_ablation():
    exp_name = "ablation/guesser"
    for topic in topics:
        for guesser in guessers:
            model_pairs = None
            if topic == "high_school_mathematics":
                model_pairs = [('Meta-Llama-3-70B-Instruct', 'Mistral-7B-Instruct-v0.2'), ('Meta-Llama-3-8B-Instruct', 'gpt-4o'), ('Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-70B-Instruct'), ('Meta-Llama-3-8B-Instruct', 'Mixtral-8x7B-Instruct-v0.1'), ('Meta-Llama-3-8B-Instruct', 'Mistral-7B-Instruct-v0.2'), ('Mixtral-8x7B-Instruct-v0.1', 'gpt-4o'), ('Mixtral-8x7B-Instruct-v0.1', 'Meta-Llama-3-70B-Instruct'), ('Mixtral-8x7B-Instruct-v0.1', 'Meta-Llama-3-8B-Instruct'), ('Mixtral-8x7B-Instruct-v0.1', 'Mistral-7B-Instruct-v0.2'), ('Mistral-7B-Instruct-v0.2', 'gpt-4o'), ('Mistral-7B-Instruct-v0.2', 'Meta-Llama-3-70B-Instruct'), ('Mistral-7B-Instruct-v0.2', 'Meta-Llama-3-8B-Instruct'), ('Mistral-7B-Instruct-v0.2', 'Mixtral-8x7B-Instruct-v0.1')]

            test_full(
                models,
                get_model_cards(topic, "bullet_point", 4),
                "mmlu",
                topic,
                f"{exp_name}/{topic}/{guesser.split('/')[1]}",
                [guesser],
                False,
                "partial",
                "full",
                120,
                20,
                model_pairs,
            )


if __name__ == "__main__":
    run_ablation()
