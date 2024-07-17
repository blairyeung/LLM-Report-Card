import os 

try:
    OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
except:
    OPENAI_API_KEY = None

try:
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
except:
    CLAUDE_API_KEY = None

try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
except:
    GEMINI_API_KEY = None

try:
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")
except:
    QWEN_API_KEY = None


HF_API_TOKENS = [
    os.getenv("HF_API_TOKEN"),
]

GROQ_API_KEYS = [
    None
]


GPT_4_MODEL_NAME = "gpt-4o"
GPT_3_MODEL_NAME = "gpt-3.5-turbo"
GPT_MODEL_NAME = GPT_4_MODEL_NAME

LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

ALL_MODELS = [
    'gpt-4o',
    'claude-3-opus-20240229',
    'gpt-4-turbo',
    'Meta-Llama-3-70B-Instruct',
    'gpt-3.5-turbo',
    'Meta-Llama-3-8B-Instruct',
    'Mixtral-8x7B-Instruct-v0.1',
    'gemma-1.1-7b-it',
    'Mistral-7B-Instruct-v0.2',
]

MODELS_2_ELO = {
    'gpt-4o': 1287.055,
    'claude-3-opus-20240229': 1248.078,
    'gpt-4-turbo': 1246.366,
    'Meta-Llama-3-70B-Instruct': 1208.031602,
    'gpt-3.5-turbo': 1108.074137,
    'Meta-Llama-3-8B-Instruct': 1154.47553,
    'Mixtral-8x7B-Instruct-v0.1': 1114,
    'gemma-1.1-7b-it': 1084.464441,
    'Mistral-7B-Instruct-v0.2': 1074.293606,
}

CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-5-sonnet-20240620"
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
CLAUDE_3_MODEL_NAME = CLAUDE_3_SONNET

GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"

REG_WORD_LIM = 1024
REG_CRITERIA_LIM = 12

BIG_SEPARATOR = "====="
SMALL_SEPARATOR = "-----"

OUTPUTS_FOLDER_NAME = "outputs"
PROMPTS_FOLDER_NAME = "prompts"

PSEUDO_MODEL_NAMES = ["A", "B"]

ALL_AVAILABLE_METAS = ["mmlu", "anthropic-eval", "gsm8k", "hotpot_qa", "openend"]

ALL_STUDENT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    CLAUDE_3_OPUS,
    GPT_4_MODEL_NAME,
    'gpt-4-turbo',
    GPT_3_MODEL_NAME,
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-1.1-7b-it",
]

ALL_MMLU_TOPICS = [
    "high_school_mathematics",
    "college_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "high_school_world_history",
    "machine_learning",
]

ALL_ANTHROPIC_TOPICS = [
    "corrigible-less-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    "survival-instinct",
    "self-awareness-general-ai",
]

ALL_OPENEND_TOPICS = [
    "Crafting engaging and contextually appropriate jokes",
    "Discussing philosophical theories of consciousness",
    "Writing efficient code for solving concrete algorthimic problems",
    "Providing dating advice",
    "Roleplaying as a fictional character",
]
