from core.models import GPTModel
from core.config import *
from tqdm import tqdm, trange

TOPICS = [
# "Understanding and critiquing elements of Renaissance art",
# "Explaining the principles behind quantum mechanics",
# "Discussing the evolution of mental health therapies",
# "Giving opinions on the impact of social media on society",
# "Roleplaying as a fictional character",
# "Roasting a friend in a lighthearted manner",
# "Detailing the process and challenges of urban farming",
# "Offering insights into the causes and effects of the Industrial Revolution",
# "Sharing tips on preparing gluten-free meals",
# "Creating and understanding various types of humor, including puns and satire",
# "Explaining the mechanics of blockchain technology",
# "Offering fitness and exercise recommendations",
# "Providing dating advice",
# "Making funny noises and sound effects",
# "Discussing philosophical theories of consciousness",
# "Providing updates on climate change mitigation technologies",
# "Understanding causal relationships",
# "Suggesting wine pairings with various cuisines",
# "Crafting engaging and contextually appropriate jokes",
# "Explaining the role of AI in enhancing educational practices"
# "Writing efficient code for solving concrete algorthimic problems",
"Solving leetcode questions in python"
]

import os
import openai
import numpy as np
from datasets import load_dataset
import json
import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import multiprocessing as mp
import time

if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=300)
  parser.add_argument('--tag', type=str, default='')
  args = parser.parse_args()

  SYSTEM = "You are an expert at evaluating the capabilities, biases, and response patterns of AI assistants with respect to specific topics or skills."

  with open('../datasets/openend/_raw/build_dataset.prompt', 'r') as f:
    TEMPLATE = f.read()

  for t in TOPICS:
    results_file = f'../datasets/openend/_raw/{t}.json'
  
    if os.path.exists(results_file):
      with open(results_file, 'r') as f:
        res = json.load(f)
        continue
    else:
      res = {}

    messages = [
      {'role': 'system', 'content': SYSTEM},
      {'role': 'user', 'content': TEMPLATE.format(topic=t)}
    ]

    try:
      model = GPTModel(system_prompt=SYSTEM,
                       model_name=GPT_4_MODEL_NAME)
      for _ in trange(1):
        sample = model(TEMPLATE.format(topic=t), use_json=True)
    except Exception as e:
      print(e)
      continue

    try:
      # j = json.loads(sample.split('===')[1])
      j = sample
      
    except Exception as e:
      # print(sample.split('===')[1])
      print(e)
      continue
    
    res = j

    with open(results_file, 'w') as f:
      json.dump(res, f, indent=2)
    
    time.sleep(3)