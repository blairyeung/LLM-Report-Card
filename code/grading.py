from core.models import Llama3Model, select_model, YIModel
from core.config import LLAMA_MODEL_NAME, HF_API_TOKENS
import json, jsonlines, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from tqdm import tqdm, trange


def get_last_grade(response):
    matches = re.findall(r'grade: (\d+)', response.lower())
    if matches:
        grade = matches[-1].lower()
        return grade
    else:
        return None
    
def change_dir():   
    # Get the current working directory
    current_dir = os.getcwd()
    print("Current Directory:", current_dir)

    # Change the current working directory to the parent directory
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

def extractor(completion):
    extractor_name = ''

def helper(topic, index, question, completion, criteria):
    
    grader = Llama3Model(model_name=LLAMA_MODEL_NAME,
                     system_prompt=f"Review the answer to the following {topic} question CRITICALLY. Provide constructive feedback based on the following guidelines: \n- Highlight the strengths and what was done well.\n- Point out areas that are partially correct or relevant but need improvement.\n- Identify aspects that are incorrect, irrelevant, or of poor quality. Be objective and fair in your review.")
        
    grader.api_key_index = index % len(HF_API_TOKENS)
    user_prompt = f"Question: {question}\n\nAnswer: {completion}\n\nReview this answer critically and objectively based on the criteria provided.{str(criteria)}"

    response = grader(user_prompt=user_prompt, max_new_tokens=1024)

    # assume the response is in the form of "Grade: [1-10]"
    # extract the grade using re, convert to lower case
    # grade = get_last_grade(response)
    
    return index, response

def grade_dataset(models, topic):
    # only Openended
    for model in models:
        for partition in ['train', 'test']:
            path = f'datasets/openend/{topic}/{model}-{partition}.jsonl'
            data = []
            with jsonlines.open(path) as reader:
                for obj in reader:
                    data.append(obj)

            questions = [obj['question'] for obj in data]
            completions = [obj['completion'] for obj in data]
            max_worker = 40

            m = model
            with ThreadPoolExecutor(max_workers=max_worker) as executor:
                futures = [executor.submit(helper, topic, j, questions[j], completions[j]) for j in range(len(questions))]
                results = {}
                for future in tqdm(as_completed(futures), desc=f'Processing {model}'):
                    index, grade = future.result()
                    results[index] = {'grade': grade}

            # print(results)
        
            for i in range(len(data)):
                data[i]['comment'] = results[i]['grade']

            # print average grade
            total = 0
            for obj in data:
                total += int(obj['grade'])
            print(f'{model} {partition} average grade: {total/len(data)}')

            with jsonlines.open(path, 'w') as writer:
                for obj in data:
                    writer.write(obj)



if __name__ == '__main__':
    change_dir()

    models = ['gpt-4o', 'claude-3-opus-20240229', 'gpt-4-turbo', 'Meta-Llama-3-70B-Instruct', 'gpt-3.5-turbo', 'Meta-Llama-3-8B-Instruct', 'Mixtral-8x7B-Instruct-v0.1', 'gemma-1.1-7b-it', 'Mistral-7B-Instruct-v0.2']
    models = [models[-1]]

    topic = 'Solving leetcode questions in python'

    # grade_dataset(models, topic)
    model = YIModel(model_name='01-ai/Yi-1.5-34B-Chat', system_prompt='You are a helpful assistant')
    model.api_key_index = 0
    print(model('你的中文如何'))