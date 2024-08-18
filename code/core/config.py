import os 

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# a list of Hugging Face API tokens, separated by commas
HF_API_TOKENS = os.getenv("HF_API_TOKENS").split(",")

GROQ_API_KEYS = []

# use exact version for better reproducibility
GPT_4_MODEL_NAME = "gpt-4o-2024-05-13"
GPT_3_MODEL_NAME = "gpt-3.5-turbo-0125"
# GPT_4_MODEL_NAME = GPT_3_MODEL_NAME  # temporary
GPT_MODEL_NAME = GPT_4_MODEL_NAME

CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
CLAUDE_3_MODEL_NAME = CLAUDE_3_5_SONNET

GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"

REG_WORD_LIM = 768
REG_CRITERIA_LIM = 12

BIG_SEPARATOR = "====="
SMALL_SEPARATOR = "-----"

OUTPUTS_FOLDER_NAME = "outputs"
PROMPTS_FOLDER_NAME = "prompts"

PSEUDO_MODEL_NAMES = ["A", "B"]

ALL_AVAILABLE_METAS = ["mmlu", "anthropic_eval", "gsm8k", "hotpot_qa", "openend"]

ALL_STUDENT_MODELS = [
    "gemma-1.1-7b-it",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1",
]

DATASET_TO_TOPICS = {
    "mmlu": [
        "college_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_world_history",
        "machine_learning",
    ],
    "anthropic_eval": [
        "corrigible-less-HHH",
        "myopic-reward",
        "power-seeking-inclination",
        "self-awareness-general-ai",
        "survival-instinct",
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


PLOT_HEIGHT = 6
PLOT_WIDTH = 7
PLOT_SIZE = (PLOT_WIDTH, PLOT_HEIGHT)

PLOT_TITLE_FONT_SIZE = 18
PLOT_AXIS_FONT_SIZE = 16
PLOT_TICK_FONT_SIZE = 15
PLOT_LABEL_FONT_SIZE = 10
PLOT_LEGEND_FONT_SIZE = 12

PLOT_COLORS = ['#BF2642', '#F2C4D0', '#0E3B40', '#F2D194', '#CEF2EC']