from abc import ABC, abstractmethod
from typing import Tuple, Iterable
import copy

from jsonlines import jsonlines

from models import *
from utils import *
from utils import get_choice_str, get_choice_int


class Batch(ABC):

    @abstractmethod
    def get_train_str(self, model: str = None,
                      include_model_answer: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_refine_str(self, index: int, include_model_answer: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_eval_predictive_str(self, index: int, model: str = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_eval_preferential_str(self, index: int, model: str = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_accuracy(self, model: str = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_oracle(self, model: str = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_question(self, index: int, model: str = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_true_answer(self, index: int, model: str = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model_answer(self, index: int, model: str = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model_reasoning(self, index: int, model: str = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def set_model(self, model: str):
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError


class AnthropicBatch(Batch):
    model: Optional[str]
    # model to (question, true answer, false answer, model answer, model reasoning)
    raw: Dict[str, List[Tuple[str, str, str, str, str]]]

    def __init__(self, raw: Dict[str, List[Tuple[str, str, str, str, str]]], model: str = None):
        self.model = model
        self.raw = raw

    def get_train_str(self, model: str = None, include_model_answer: bool = True) -> str:
        s = ''
        for i in range(len(self)):
            s += f'Question:\n{self.get_question(i, model)}\n\n'
            s += f'Proper Choice: {self.get_true_answer(i, model)}\n'
            s += f'Improper Choice: {self.get_false_answer(i, model)}\n'
            if include_model_answer:
                s += f"Student's Choice: {self.get_model_answer(i, model)}\n"
                s += f"Student's Analysis:\n{self.get_model_reasoning(i, model)}\n"
            s += f'\n{SMALL_SEPARATOR}\n'
        return s

    def get_refine_str(self, index: int, include_model_answer: bool = True) -> str:
        return self.get_train_str(self.model, include_model_answer)

    def get_eval_predictive_str(self, index: int, model: str = None) -> str:
        s = f'Question:\n{self.get_question(index, model)}\n\n'
        s += f'Proper (Ground Truth) Choice: {self.get_true_answer(index, model)}\n'
        s += f'Improper Choice: {self.get_false_answer(index, model)}\n'
        return s

    def get_eval_preferential_str(self, index: int, model: str = None) -> str:
        s = f'Question:\n{self.get_question(index, model)}\n\n'
        s += f'Proper (Ground Truth) Choice: {self.get_true_answer(index, model)}\n'
        s += f'Improper Choice: {self.get_false_answer(index, model)}\n'
        s += f"Student's Choice: {self.get_model_answer(index, model)}\n"
        s += f"Student's Reasoning:\n{self.get_model_reasoning(index, model)}\n"
        return s

    def get_accuracy(self, model: str = None) -> float:
        if model is None:
            assert self.model is not None
            model = self.model
        correct = 0
        for i in range(len(self)):
            if self.get_true_answer(i, model) == self.get_model_answer(i, model):
                correct += 1
        return correct / len(self.raw)

    def get_oracle(self, model: str = None) -> float:
        acc = self.get_accuracy(model)
        return max(acc, 1 - acc)

    def shuffle(self):
        for model in self.raw:
            random.shuffle(self.raw[model])

    def get_question(self, index: int, model: str = None) -> str:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][0]

    def get_true_answer(self, index: int, model: str = None) -> Any:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][1]

    def get_false_answer(self, index: int, model: str = None) -> Any:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][2]

    def get_model_answer(self, index: int, model: str = None) -> str:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][3]

    def get_model_reasoning(self, index: int, model: str = None) -> str:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][4]

    def set_model(self, model: str):
        self.model = model

    def __len__(self):
        assert self.model is not None
        return len(self.raw[self.model])

    def __getitem__(self, item):
        assert self.model is not None
        return self.raw[self.model][item]

    def __iter__(self):
        assert self.model is not None
        return iter(self.raw[self.model])

    def __add__(self, other):
        raise NotImplementedError


class MMLUBatch(Batch):
    model: Optional[str]
    # model to (question, choices, true answer, model answer, model reasoning)
    raw: Dict[str, List[Tuple[str, List[str], int, int, str]]]

    def __init__(self, raw: Dict[str, List[Tuple[str, List[str], int, int, str]]], model: str = None):
        self.model = model
        self.raw = raw

    def get_train_str(self, model: str = None,
                      include_model_answer: bool = True) -> str:
        s = ''
        for i in range(len(self)):
            s += f'Question:\n{self.get_question(i, model)}\n'
            s += f'Ground Truth Answer: {get_choice_str(self.get_true_answer(i, model))}\n'
            if include_model_answer:
                s += f"Student's Answer: {get_choice_str(self.get_model_answer(i, model))}\n"
                s += f"Student's Reasoning:\n{self.get_model_reasoning(i, model)}\n"
            s += f'\n{SMALL_SEPARATOR}\n'
        return s

    def get_refine_str(self, index: int, include_model_answer: bool = True) -> str:
        return self.get_train_str(self.model, include_model_answer)

    def get_eval_predictive_str(self, index: int, model: str = None) -> str:
        q = self.get_question(index, model)
        s = f'Question:\n{q}\n\n'
        return s

    def get_eval_preferential_str(self, index: int, model: str = None) -> str:
        q = self.get_question(index, model)
        s = f'Question:\n{q}\n\n'
        s += f"Student's Answer: {get_choice_str(self.get_model_answer(index, model))}\n"
        s += f"Student's Reasoning:\n{self.get_model_reasoning(index, model)}\n"
        return s

    def get_eval_coverage_str(self, index: int, model: str = None) -> str:
        q = self.get_question(index, model)
        s = f'Question:\n{q}\n\n'
        s += f'Ground Truth Answer: {get_choice_str(self.get_true_answer(index, model))}\n'
        s += f"Student's Answer: {get_choice_str(self.get_model_answer(index, model))}\n"
        s += f"Student's Reasoning:\n{self.get_model_reasoning(index, model)}\n"
        return s

    def get_eval_coverage_batch_str(self, indices: Iterable[int], model: str = None) -> str:
        s = ''
        for i in indices:
            s += self.get_eval_coverage_str(i, model)
            s += f'\n{SMALL_SEPARATOR}\n'
        return s

    def sample(self, k: int) -> Self:
        # return MMLUBatch(random.sample(self.raw, k), self.model)
        raise NotImplementedError

    def get_accuracy(self, model: str = None) -> float:
        if model is None:
            assert self.model is not None
            model = self.model
        correct = 0
        for i in range(len(self)):
            if self.get_true_answer(i, model) == self.get_model_answer(i, model):
                correct += 1
        return correct / len(self)

    def get_oracle(self, model: str = None) -> float:
        acc = self.get_accuracy(model)
        return max(acc, 1 - acc)

    def get_choices(self, index: int, model: str = None) -> List[str]:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][1]

    def get_question(self, index: int, model: str = None) -> str:
        if model is None:
            assert self.model is not None
            model = self.model
        s = f'{self.raw[model][index][0]}\n\n'
        for i, choice in enumerate(self.get_choices(index, model)):
            s += f'{get_choice_str(i)}. {choice}\n'
        return s

    def get_true_answer(self, index: int, model: str = None) -> int:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][2]

    def get_model_answer(self, index: int, model: str = None) -> int:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][3]

    def get_model_reasoning(self, index: int, model: str = None) -> str:
        if model is None:
            assert self.model is not None
            model = self.model
        return self.raw[model][index][4]

    def shuffle(self):
        for model in self.raw:
            random.shuffle(self.raw[model])

    def set_model(self, model: str):
        self.model = model

    def __len__(self):
        assert self.model is not None
        return len(self.raw[self.model])

    def __getitem__(self, item):
        assert self.model is not None
        return self.raw[self.model][item]

    def __iter__(self):
        assert self.model is not None
        return iter(self.raw[self.model])

    def __add__(self, other) -> Self:
        new_batch = MMLUBatch(copy.deepcopy(self.raw), self.model)
        for model in self.raw:
            new_batch.raw[model] += copy.deepcopy(other.raw[model])
        return new_batch


def reshuffle_batches(batches: List[MMLUBatch]) -> List[MMLUBatch]:
    batch_nums = [len(batch) for batch in batches]
    target_model = batches[0].model
    meta_batch = batches[0]
    for batch in batches[1:]:
        meta_batch += batch
    meta_batch.shuffle()
    # split the meta batch into batches
    new_batches = []
    start = 0
    for batch_num in batch_nums:
        raw = {}
        for model in meta_batch.raw:
            raw[model] = meta_batch.raw[model][start:start + batch_num]
        start += batch_num
        new_batch = MMLUBatch(raw, target_model)
        new_batches.append(new_batch)
    for batch in new_batches:
        batch.set_model(target_model)
    return new_batches


def balance_batches(batches: List[MMLUBatch]) -> List[MMLUBatch]:
    # TODO: finish this
    batch_nums = [len(batch) for batch in batches]
    meta_batch = batches[0]
    model = meta_batch.model
    for batch in batches[1:]:
        assert batch.model == model
        meta_batch += batch
    meta_acc = meta_batch.get_accuracy()
    raw = meta_batch.raw[model]
    correct_raw = [obj for obj in raw if obj[2] == obj[3]]
    incorrect_raw = [obj for obj in raw if obj[2] != obj[3]]
    new_batches = []
    start = 0
    for batch_num in batch_nums:
        correct_count = int(batch_num * meta_acc)
        incorrect_count = batch_num - correct_count
        correct_batch_raw = correct_raw[start:start + correct_count]
        incorrect_batch_raw = incorrect_raw[start:start + incorrect_count]
        start += batch_num
        new_batch = MMLUBatch({model: correct_batch_raw + incorrect_batch_raw}, model)
        new_batches.append(new_batch)
    return new_batches


def load_batches(folder_name: str, topic: str, batch_nums: List[int],
                 shuffle: bool = False, target_model: str = None) -> List[Batch]:
    if 'anthropic' in folder_name:
        return load_anthropic_batches(folder_name, topic, batch_nums, shuffle, target_model)
    elif 'mmlu' in folder_name:
        return load_mmlu_batches(folder_name, topic, batch_nums, shuffle, target_model)
    else:
        raise ValueError(f'Invalid folder name: {folder_name}')


def load_anthropic_batches(folder_name: str, topic: str, batch_nums: List[int],
                           shuffle: bool = False, target_model: str = None) -> List[AnthropicBatch]:
    assert len(batch_nums) >= 2, 'Must have at least 2 batches!'
    folder_name = os.path.join(folder_name, topic)
    train = {}
    test = {}
    for model in MODELS:
        train[model] = []
        train_filename = os.path.join(folder_name, f'{topic}_{model}_train.jsonl')
        test_filename = os.path.join(folder_name, f'{topic}_{model}_test.jsonl')
        if not os.path.exists(train_filename) or not os.path.exists(test_filename):
            continue
        assert os.path.exists(train_filename) and os.path.exists(test_filename)
        with jsonlines.open(train_filename) as lines:
            train[model] = [(obj['question'], obj['answer_matching_behavior'], obj['answer_not_matching_behavior'],
                             obj[model]['answer'], obj[model]['reasoning']) for obj in lines]
            if shuffle:
                random.shuffle(train[model])
        with jsonlines.open(test_filename) as lines:
            test[model] = [(obj['question'], obj['answer_matching_behavior'], obj['answer_not_matching_behavior'],
                            obj[model]['answer'], obj[model]['reasoning']) for obj in lines]
    # for each model, split the train set into batches
    result = []
    start = 0
    for num in batch_nums[:-1]:
        raw_batch = {}
        for model in MODELS:
            if model not in train:
                continue
            raw_batch[model] = train[model][start:start + num]
        start += num
        result.append(AnthropicBatch(raw_batch, target_model))
    # add the test set
    raw_batch = {}
    for model in MODELS:
        if model not in test:
            continue
        raw_batch[model] = test[model][:batch_nums[-1]]
    result.append(AnthropicBatch(raw_batch, target_model))
    return result


MODELS = [
    'gpt-3.5-turbo-1106',
    'Mixtral-8x7B-Instruct-v0.1',
    'Mistral-7B-Instruct-v0.2',
    'Llama-2-13b-chat-hf',
    'Llama-2-70b-chat-hf',
]


def load_mmlu_batches(folder_name: str, topic: str, batch_nums: List[int],
                      shuffle: bool = False, target_model: str = None) -> List[MMLUBatch]:
    assert len(batch_nums) >= 2, 'Must have at least 2 batches!'
    folder_name = os.path.join(folder_name, topic)
    train = {}
    test = {}
    for model in MODELS:
        train_filename = os.path.join(folder_name, f'{topic}_{model}_train.jsonl')
        test_filename = os.path.join(folder_name, f'{topic}_{model}_test.jsonl')
        if not os.path.exists(train_filename) or not os.path.exists(test_filename):
            continue
        assert os.path.exists(train_filename) and os.path.exists(test_filename)
        train[model] = []
        with jsonlines.open(train_filename) as lines:
            train[model] = [(obj['question'], obj['choices'], obj['answer'],
                             obj[model]['answer'], obj[model]['reasoning']) for obj in lines]
            if shuffle:
                random.shuffle(train[model])
        with jsonlines.open(test_filename) as lines:
            test[model] = [(obj['question'], obj['choices'], obj['answer'],
                            obj[model]['answer'], obj[model]['reasoning']) for obj in lines]
    # for each model, split the train set into batches
    result = []
    start = 0
    for num in batch_nums[:-1]:
        raw_batch = {}
        for model in MODELS:
            if model not in train:
                continue
            raw_batch[model] = train[model][start:start + num]
        start += num
        result.append(MMLUBatch(raw_batch, target_model))
    # add the test set
    raw_batch = {}
    for model in MODELS:
        if model not in test:
            continue
        raw_batch[model] = test[model][:batch_nums[-1]]
    result.append(MMLUBatch(raw_batch, target_model))
    return result


def validate_dataset(folder_name: str, topic: str, model: str, verbose: bool = True):
    # TODO: update this
    filename = os.path.join(folder_name, f'{topic}.jsonl')
    with jsonlines.open(filename) as lines:
        raw = [obj for obj in lines]
    valid = True
    for obj in raw:
        assert 'models' in obj
        if model not in obj['models']:
            if valid and verbose:  # first time
                print(f'=====Dataset is invalid for model {model}!====')
            valid = False
            if verbose:
                print(obj['question'])
                print('==================================')
    return valid


def gpt_generate_answers(filename: str, topic: str):
    with jsonlines.open(filename) as lines:
        raw = [obj for obj in lines]

    rslt_df = copy.deepcopy(raw)

    def f(j, obj):
        model = GPTModel(f'Your are an expert in {topic}. Your task is to answer multiple-choice question.', GPT_3_MODEL_NAME)
        user_prompt = f'Question:\n{obj["question"]}\n\n'
        for k, choice in enumerate(obj['choices']):
            user_prompt += f'{get_choice_str(k)}. {choice}\n'
        user_prompt += """
You must write your answer in the following JSON format:
{
    "analysis": your analysis for the question,
    "choice": EXACTLY one of A, B, C, or D, or N (none of above).
}
You should write your analysis first before reaching to a final answer.
Make sure in "choice", only put exactly one token of A, B, C, or D, or N (none of above). 
                """
        try:
            r = model(user_prompt, use_json=True, timeout=10)
            # print(model.messages) 
            # e = json.loads(r)
            e = r
            # print(r)
            return j, get_choice_int(e['choice']), e['analysis']
        except Exception as e:
            print(e, file=sys.stderr)

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for i, obj in enumerate(rslt_df):
            # if 'models' not in obj:
            #     # obj['models'] = {}
            #     pass
            # if GPT_3_MODEL_NAME in obj['models'] and isinstance(obj['models'][GPT_3_MODEL_NAME], dict):
            #     continue
            futures.append(executor.submit(f, i, obj))
        for future in tqdm(as_completed(futures), 'Generating Answers'):
            if future.result() is None:
                continue
            i, answer, reason = future.result()
            rslt_df[i].pop('Mistral-7B-Instruct-v0.2')
            rslt_df[i][GPT_3_MODEL_NAME] = {
                'answer': answer,
                'reasoning': reason
            }



    new_file_name = filename.replace('Mistral-7B-Instruct-v0.2', GPT_3_MODEL_NAME)
    jsonlines.open(new_file_name, 'w').write_all(rslt_df)


def random_generate_answers(filename: str):
    with jsonlines.open(filename) as lines:
        raw = [obj for obj in lines]
    for obj in raw:
        if 'models' not in obj:
            obj['models'] = {}
        obj['models']['random'] = {
            'answer': random.randint(0, 3),
            'reasoning': ''
        }
    jsonlines.open(filename, 'w').write_all(raw)


def hf_models_generate_answers(filename: str, model_name: str):
    # filename = os.path.join('../datasets/mmlu', f'{topic}.jsonl')
    with jsonlines.open(filename) as lines:
        raw = [obj for obj in lines]

    rslt_df = copy.deepcopy(raw)

    # with jsonlines.open(filename) as lines:
        # raw = [obj for obj in lines]

    def helper(index: int, obj: Dict):
        q, choices, true_answer = obj['question'], obj['choices'], obj['answer']
        qa = q + '\n\n'
        for k, choice in enumerate(obj['choices']):
            qa += f'{get_choice_str(k)}. {choice}\n'
        qa += '\n'

        system_prompt = read_all(f'prompts/data/system.txt').format(topic=topic)
        user_prompt = read_all(f'prompts/data/user.txt').format(qa=qa)
        model = Llama3Model(system_prompt, model_name)
        r = None
        try:
            r = model(user_prompt, use_json=True, timeout=30, temperature=0.3, cache=False)
            return index, get_choice_int(r['answer'][0]), r['reasoning']
        except Exception as e:
            print(f'Exception: {e}\nQuestion: {q}\nResponse: {r}')
            return None

    max_workers = 30
    if 'llama' in model_name:
        max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, obj in enumerate(raw):  # for each question
            # if 'models' not in obj:
            #     obj['models'] = {}
            # if model_name.split('/')[1] in obj['models']:
            #     continue  # already generated
            futures.append(executor.submit(helper, i, obj))
        for future in tqdm(as_completed(futures), 'Generating Answers'):
            if future.result() is None:
                continue
            i, answer, reason = future.result()
            rslt_df[i].pop('Mistral-7B-Instruct-v0.2')
            rslt_df[i][model_name.split('/')[1]] = {
                'answer': answer,
                'reasoning': reason
            }

        
        new_file_name = filename.replace('Mistral-7B-Instruct-v0.2', model_name.split('/')[-1])
        jsonlines.open(new_file_name, 'w').write_all(rslt_df)
    load_mmlu_batches('../datasets/mmlu', topic, [50, 50])


def _generate_data():
    topics = [
        'abstract_algebra',
        'astronomy',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_physics',
        'computer_security',
        'elementary_mathematics',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_mathematics',
        'high_school_physics',
        'high_school_statistics',
        'human_aging',
        'human_sexuality',
        'miscellaneous',
        'high_school_world_history',
        'machine_learning',
    ]
    model_names = [
        'mistralai/Mistral-7B-Instruct-v0.2',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
        # 'meta-llama/Llama-2-7b-chat-hf'
    ]
    for model_name in model_names:
        for topic in topics:
            print(f'==================== {model_name} {topic} ====================')
            hf_models_generate_answers(topic, model_name)
            print('Validating dataset................')
            if not validate_dataset('../datasets/mmlu', topic, model_name.split('/')[1]):
                print(f'Invalid: {model_name} {topic}!')


if __name__ == '__main__':
    topics = ['high_school_chemistry',
              'high_school_physics',
              'high_school_mathematics',
              'high_school_biology',
              'machine_learning',]
    # model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model = 'groq/llama3-8b-8192'
    for topic in tqdm(topics):
       for t in ['train', 'test']:
            # gpt_generate_answers(f'datasets/mmlu/{topic}/Mistral-7B-Instruct-v0.2-{t}.jsonl', topic)
            hf_models_generate_answers(f'datasets/mmlu/{topic}/Mistral-7B-Instruct-v0.2-{t}.jsonl', model)
            time.sleep(15)
