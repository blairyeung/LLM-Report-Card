

from __future__ import annotations


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
"Generating efficient algorithms for solving complex problems",
]

import copy
import json
from anthropic import Anthropic
import logging
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict, Union, Optional, Self

import numpy as np
import torch
from groq import Groq
from huggingface_hub import InferenceClient
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

ROLE_SYSTEM = 'system'
ROLE_USER = 'user'
ROLE_ASSISTANT = 'assistant'

SUPPORTED_OPENAI_MODELS = ['gpt-4-turbo']
SUPPORTED_MISTRAL_MODELS = ['mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2']
SUPPORTED_NOUS_MODELS = ['NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO']
SUPPORTED_LLAMA_MODELS = ['meta-llama/Llama-2-70b-chat-hf',
                          'meta-llama/Llama-2-13b-chat-hf',
                          'meta-llama/Llama-2-7b-chat-hf']


SUPPORTED_GROQ_MODELS = ['groq/llama3-8b-8192', 'groq/llama3-70b-8192', 'groq/mixtral-8x7b-32768', 'groq/gemma-7b-it']
SUPPORTED_LLAMA3_MODELS = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]


# suppress logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class Model(ABC):
    name: str
    messages: List[Dict[str, str]]
    system_prompt: str

    def __init__(self, model_name: str, system_prompt: str):
        self.name = model_name
        self.system_prompt = system_prompt
        self.messages = [
            {'role': ROLE_SYSTEM, 'content': system_prompt}
        ]

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[str, Dict]:
        raise NotImplementedError

    def add_message(self, role: str, content: str):
        assert role in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]
        self.messages.append({'role': role, 'content': content})

    def clear_conversations(self):
        self.messages.clear()
        self.add_message(ROLE_SYSTEM, self.system_prompt)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class CostManager:
    #                              in , out
    tokens: defaultdict[str, List[int]]

    def __init__(self):
        self.tokens = defaultdict(lambda: [0, 0])

    def record_tokens(self, in_tokens: int, out_tokens: int, model_name: str = GPT_4_MODEL_NAME):
        self.tokens[model_name][0] += in_tokens
        self.tokens[model_name][1] += out_tokens

    def get_cost(self) -> float:
        return (self.tokens[GPT_4_MODEL_NAME][0] / 1000 * 0.01 +
                self.tokens[GPT_4_MODEL_NAME][1] / 1000 * 0.03 +
                self.tokens[GPT_3_MODEL_NAME][0] / 1000 * 0.001 +
                self.tokens[GPT_3_MODEL_NAME][1] / 1000 * 0.002)

    def get_info_dict(self) -> Dict:
        return {'total_cost': self.get_cost(), 'details': self.tokens}

class ClaudeModel(Model):
    client: Anthropic
    def __init__(self, system_prompt: str,
                 model_name: str = CLAUDE_3_MODEL_NAME,
                 tm: CostManager = None,
                 api_key: str = CLAUDE_API_KEY):
        # super().__init__(model_name, system_prompt)
        self.name = model_name
        self.system_prompt = system_prompt
        self.messages = []
        self.client = Anthropic(api_key=api_key)

        self.total_in_tokens = 0
        self.total_out_tokens = 0
        self.tm = None
        self.name = model_name

        self.json_transcriber = GPTModel('You are a good json transcriber, you are good at copying texts. Copy the text to the given json format.',
                                          GPT_3_MODEL_NAME,
                                          tm=tm)

    # def __del__(self):
    #     self.client.close()

    def __call__(self, user_prompt: str, *args,
                use_json: bool = False,
                temperature: float = 0.,
                timeout: float = None,
                **kwargs) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, returns a json dict.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(temperature, use_json, timeout)
        # content = response.choices[0].message.content
        content = response.content[0].text
        self.add_message(ROLE_ASSISTANT, content)
        # record tokens
        # if self.tm is not None:
        #     self.tm.record_tokens(response.usage.prompt_tokens, response.usage.completion_tokens, self.name)
        # self.total_out_tokens += response.usage.completion_tokens
        # self.total_in_tokens += response.usage.prompt_tokens


        if use_json:
            json_obj = None
            try:
                json_obj = self.extract_json(content)
                # assert every single criterion has all the fields
                for criterion in json_obj:
                    assert all([field in json_obj[criterion] for field in ['overview', 'thinking_pattern', 'strength', 'weakness']])

            except:
            # if True:
                prompt = f'Please convert the following text to a json format:\n\n{content}'
                prompt += '''
                {{
                "criterion_name_1": {{
                    "overview": "...",
                    "thinking_pattern": "...",
                    "strength": "...",
                    "weakness": "..."
                }},
                "criterion_name_2": {{
                    "overview": "...",
                    "thinking_pattern": "...",
                    "strength": "...",
                    "weakness": "..."
                }},
                ...
                }}
                Notes:
                - If a criterion/sub-criterion is empty, leave all its fields blank but include it in the JSON.
                    '''
                json_obj = self.json_transcriber(content, use_json=True)

            return json_obj
        
        return content
    
    def extract_json(self, text):
        # match first { and last }
        starting_ind = text.find('{')
        ending_ind = text.rfind('}')
        if starting_ind == -1 or ending_ind == -1:
            return None
        return json.loads(text[starting_ind:ending_ind+1])

    def get_response(self, temperature: float,
                        use_json: bool,
                        timeout: float = None) -> ChatCompletion:
        message = self.client.messages.create(
                max_tokens=4096,
                system=self.system_prompt,
                messages=self.messages,
                model=self.name,
        )
        return message

@classmethod
def from_messages(cls, messages: List[Dict[str, str]], model_name: str, tm: CostManager = None) -> Self:
    assert (len(messages) >= 1 and
            all([m['role'] in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT] for m in messages]) and
            messages[0]['role'] == ROLE_USER)
    system_prompt = messages[0]['content']
    model = cls(system_prompt, model_name, tm=tm)
    model.messages = copy.deepcopy(messages)
    return model

class GPTModel(Model):
    client: OpenAI

    total_in_tokens: int
    total_out_tokens: int
    tm: Optional[CostManager]

    def __init__(self, system_prompt: str,
                 model_name: str = GPT_MODEL_NAME,
                 tm: CostManager = None,
                 api_key: str = OPENAI_API_KEY,
                 base_url: str = None):
        super().__init__(model_name, system_prompt)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.total_in_tokens = 0
        self.total_out_tokens = 0
        self.tm = tm

    def __del__(self):
        self.client.close()

    def __call__(self, user_prompt: str, *args,
                 use_json: bool = False,
                 temperature: float = 0.,
                 timeout: float = None,
                 **kwargs) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, returns a json dict.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(temperature, use_json, timeout)
        content = response.choices[0].message.content
        self.add_message(ROLE_ASSISTANT, content)
        # record tokens
        if self.tm is not None:
            self.tm.record_tokens(response.usage.prompt_tokens, response.usage.completion_tokens, self.name)
        self.total_out_tokens += response.usage.completion_tokens
        self.total_in_tokens += response.usage.prompt_tokens
        if use_json:
            return json.loads(content)
        return content

    def get_response(self, temperature: float,
                     use_json: bool,
                     timeout: float = None) -> ChatCompletion:
        if use_json:
            return self.client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
                response_format={'type': 'json_object'},
            )
        else:
            return self.client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
            )

    @classmethod
    def from_messages(cls, messages: List[Dict[str, str]], model_name: str, tm: CostManager = None) -> Self:
        assert (len(messages) >= 1 and
                all([m['role'] in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT] for m in messages]) and
                messages[0]['role'] == ROLE_SYSTEM)
        system_prompt = messages[0]['content']
        model = cls(system_prompt, model_name, tm=tm)
        model.messages = copy.deepcopy(messages)
        return model


class HFAPIModel(Model):

    def __call__(self, user_prompt: str, *args,
                 use_json: bool = False,
                 temperature: float = 0,
                 timeout: float = None,
                 cache: bool = False,
                 json_retry_count: int = 5,
                 **kwargs) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, will try its best to return a json dict, but not guaranteed.
        If we cannot parse the JSON, we will return the response string directly.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(temperature, use_json, timeout, cache)
        if use_json:
            for i in range(json_retry_count):
                # cache only if both instruct to do and first try
                response = self.get_response(temperature, use_json, timeout, cache and i == 0)
                json_obj = self.find_first_valid_json(response)
                if json_obj is not None:
                    response = json_obj
                    break
        self.add_message(ROLE_ASSISTANT, response)
        return response

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(max=15), reraise=True)  # retry if exception
    def get_response(self, temperature: float, use_json: bool, timeout: float, cache: bool) -> str:
        client = InferenceClient(model=self.name, token=random.choice(HF_API_TOKENS), timeout=timeout)
        if not cache:
            client.headers["x-use-cache"] = "0"

        r = client.text_generation(self.format_messages(),
                                   do_sample=temperature > 0,
                                   temperature=temperature if temperature > 0 else None,
                                   stop_sequences=['<|eot_id|>'],
                                   max_new_tokens=1024)
        
        return r

    @abstractmethod
    def format_messages(self) -> str:
        raise NotImplementedError

    def get_short_name(self) -> str:
        """
        Returns the last part of the model name.
        For example, "mistralai/Mixtral-8x7B-Instruct-v0.1" -> "Mixtral-8x7B-Instruct-v0.1"
        """
        return self.name.split('/')[-1]

    @staticmethod
    def find_first_valid_json(s) -> Optional[Dict]:
        s = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', lambda m: m.group(0)[1:], s)  # remove all invalid escapes chars
        for i in range(len(s)):
            if s[i] != '{':
                continue
            for j in range(i + 1, len(s) + 1):
                if s[j - 1] != '}':
                    continue
                try:
                    potential_json = s[i:j]
                    json_obj = json.loads(potential_json, strict=False)
                    return json_obj  # Return the first valid JSON object found
                except json.JSONDecodeError:
                    pass  # Continue searching if JSON decoding fails
        return None  # Return None if no valid JSON object is found


class MistralModel(HFAPIModel):

    def __init__(self, system_prompt: str, model_name: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1') -> None:
        assert model_name in ['mistralai/Mixtral-8x7B-Instruct-v0.1',
                              'mistralai/Mistral-7B-Instruct-v0.2',
                              'mistralai/Mixtral-8x22B-Instruct-v0.1'], 'Model not supported'
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        # mistral doesn't support system prompt, so we need to convert it to user prompt
        if messages[0]['role'] == ROLE_SYSTEM:
            assert len(self.messages) >= 2
            messages = [{'role'   : ROLE_USER,
                         'content': messages[0]['content'] + '\n' + messages[1]['content']}] + messages[2:]
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=4096)
        # print(r)
        return r


class NousHermesModel(HFAPIModel):

    def __init__(self, system_prompt: str, model_name: str = 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO') -> None:
        assert model_name in ['NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'], 'Model not supported'
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]['role'] == ROLE_SYSTEM and messages[1]['role'] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=4096)
        # print(r)
        return r


class LlamaModel(HFAPIModel):

    def __init__(self, system_prompt: str, model_name: str = 'meta-llama/Llama-2-70b-chat-hf') -> None:
        assert model_name in ['meta-llama/Llama-2-70b-chat-hf',
                              'meta-llama/Llama-2-13b-chat-hf',
                              'meta-llama/Llama-2-7b-chat-hf',
                              'meta-llama/Meta-Llama-3-8B',
                              'meta-llama/Meta-Llama-3-8B-instruct',
                              'meta-llama/Meta-Llama-3-70B'
                              'meta-llama/Meta-Llama-3-70B-Instruct'], 'Model not supported'
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        """
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {user_message} [/INST]
        """
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        r = f'<s>[INST] <<SYS>>\n{messages[0]["content"]}\n<</SYS>>\n\n{messages[1]["content"]} [/INST]'
        for msg in messages[2:]:
            role, content = msg['role'], msg['content']
            if role == ROLE_SYSTEM:
                assert ValueError
            elif role == ROLE_USER:
                if r.endswith('</s>'):
                    r += '<s>'
                r += f'[INST] {content} [/INST]'
            elif role == ROLE_ASSISTANT:
                r += f'{content}</s>'
            else:
                raise ValueError
        return r


class GroqModel(Model):

    def __init__(self, system_prompt: str,
                 model_name: str = GPT_MODEL_NAME, ):
        assert model_name in SUPPORTED_GROQ_MODELS
        model_name = model_name.split("/")[1]
        super().__init__(model_name, system_prompt)

    def __call__(self, user_prompt: str, *args,
                 use_json: bool = False,
                 temperature: float = 0.,
                 timeout: float = None,
                 **kwargs) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, returns a json dict.
        """
        client = Groq(api_key=random.choice(GROQ_API_KEYS))
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(client, temperature, use_json, timeout)
        content = response.choices[0].message.content
        self.add_message(ROLE_ASSISTANT, content)
        if use_json:
            return json.loads(content)
        return content

    def get_response(self, client: Groq,
                     temperature: float,
                     use_json: bool,
                     timeout: float = None) -> ChatCompletion:
        if use_json:
            return client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
                response_format={'type': 'json_object'},
            )
        else:
            return client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
            )



class Llama3Model(HFAPIModel):

    def __init__(self, system_prompt: str, model_name: str= 'meta-llama/Meta-Llama-3-70B-Instruct') -> None:
        # assert model_name in ['meta-llama/Meta-Llama-3-8B',
        #                       'meta-llama/Meta-Llama-3-8B-instruct',
        #                       'meta-llama/Meta-Llama-3-70B'
        #                       'meta-llama/Meta-Llama-3-70B-Instruct'], 'Model not supported'
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:

        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]['role'] == ROLE_SYSTEM and messages[1]['role'] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        # print(messages)
        tokenizer.eos_token = '<|eot_id|>'
        tokenizer.add_special_tokens({'eos_token': '<|eot_id|>'})
        r = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=4096)
        # print('-' * 50)
        # print(r)
        # print('-' * 50)
        return r
       


class GemmaModel(HFAPIModel):

    def __init__(self, system_prompt: str, model_name: str = 'google/gemma-7b-it') -> None:
        assert model_name in ['google/gemma-7b-it'], 'Model not supported'
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]['role'] == ROLE_SYSTEM and messages[1]['role'] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, max_length=4096)
        return r


# TODO: refactor HFLocalModel
CHAT_FORMATS = {
    "mistralai" : "<s>[INST] {prompt} [/INST]",
    "openchat"  : "GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:",
    "meta-llama": """[INST] <<SYS>>
You answer questions directly.
<</SYS>>
{prompt}[/INST]""",
    "mosaicml"  : """<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant""",
    "lmsys"     : "USER: {prompt}\nASSISTANT:",
}


class HFLocalModel:

    def __init__(self, model_name: str, bits: int = 4) -> None:
        self.model_name = model_name

        device_map = {"": 0}
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=getattr(torch, "float16"),
                bnb_4bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map)

        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type='nf8',
                bnb_8bit_compute_dtype=getattr(torch, "float16"),
                bnb_8bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                quantization_config=bnb_config,
                device_map=device_map)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side='left')
        self.tokenizer.add_special_tokens(
            {'pad_token': self.tokenizer.eos_token})

        self.pipe = pipeline(task="text-generation", model=self.model,
                             tokenizer=self.tokenizer, max_length=2048)

    def prob_eval(self, sentences, post_fix=" ", labels=None, ground_truth=[]):

        assert len(ground_truth) == len(sentences)

        inf_rslt, logits_batched, sentences = self.prob_inference(sentences, post_fix, labels)

        correct_list = []
        for i in range(len(sentences)):
            correct_list.append(np.argmax(logits_batched[i]) == labels.index(ground_truth[i]))
        correct = np.sum(correct_list)
        return correct, logits_batched, sentences

    def prob_inference(self, sentences, post_fix=" ", labels=None):
        if labels is None:
            labels = ['A', 'B']

        sentences = [CHAT_FORMATS[self.model_name.split('/')[0]].format(prompt=s) for s in sentences]
        sentences = [s + post_fix for s in sentences]

        with torch.no_grad():
            encoded_tokenized = self.tokenizer(sentences, padding=True,
                                               return_tensors="pt").to("cuda")
            rslt = self.model(encoded_tokenized.input_ids)

            logits = rslt.logits

            logits_batched = np.zeros(shape=(len(sentences), len(labels)))

            inf_rslt = []
            for i in range(len(sentences)):
                probs = [logits[i, -1, self.tokenizer.encode(labels[j])[1]].item() for j in range(len(labels))]
                probs = np.exp(probs) / np.sum(np.exp(probs))
                logits_batched[i] = np.array(probs)
                inf_rslt.append(np.argmax(logits_batched[i]))

            return inf_rslt, logits_batched, sentences

    def __call__(self, prompt) -> Any:
        llm_type = self.model_name.split('/')[0]
        prompt_format = CHAT_FORMATS[llm_type]
        formatted_prompt = prompt_format.format(prompt=prompt)
        result = self.pipe(formatted_prompt)
        return result[0]['generated_text'][len(formatted_prompt) + 1:]

    def eval(self):
        self.model.eval()


# TODO: consider remove this HFEPModel
LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

MISTRAL_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>> {user_message} [/INST]"""

YI_34B_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""


def extract_json(text: str) -> Dict:
    # json_string_match = re.search(r"json\s+(.+?)\s+", text, re.DOTALL)

    # Assume it's goind to be like: "Guess": "A" or "Guess": "B"

    # Now it's either true or false

    # print(text)
    text = text.replace('\\', '\\\\')

    try:
        rslt = json.loads(text)
    except Exception as e:
        # print(e)
        # print(text)
        rslt = None
    return rslt


# TODO: consider remove this HFEPModel
class HFEPModel:

    def __init__(self, system_prompt: str, model_name: str) -> None:
        self.system_prompt = system_prompt
        self.model_name = model_name

    def encode_hfapi(self, input: list):
        assert 0 < len(input) <= 2

        if "llama" in self.model_name:
            if len(input) == 1:
                system_prompt = self.system_prompt
                assert input[0]["role"] == "user"
                user_message = input[0]["content"]
            elif len(input) == 2:
                assert input[0]["role"] == "system"
                system_prompt = input[0]["content"]
                assert input[1]["role"] == "user"
                user_message = input[1]["content"]
            else:
                raise ValueError("Unreachable code")

            return LLAMA_TEMPLATE.format(
                system_prompt=system_prompt, user_message=user_message
            )

        elif "mistral" in self.model_name or "mixtral" in self.model_name:
            if len(input) == 1:
                system_prompt = self.system_prompt
                assert input[0]["role"] == "user"
                user_message = input[0]["content"]
            elif len(input) == 2:
                assert input[0]["role"] == "system"
                assert input[1]["role"] == "user"
                user_message = f"{input[0]['content']}\n\n{input[1]['content']}"
            else:
                raise ValueError("Unreachable code")

            return MISTRAL_TEMPLATE.format(
                system_prompt=system_prompt,
                user_message=user_message)

        elif "Yi" or 'TheBloke' in self.model_name:
            if len(input) == 1:
                system_prompt = "You are a helpful assistant."
                assert input[0]["role"] == "user"
                user_message = input[0]["content"]
            elif len(input) == 2:
                assert input[0]["role"] == "system"
                system_prompt = input[0]["content"]
                assert input[1]["role"] == "user"
                user_message = input[1]["content"]
            else:
                raise ValueError("Unreachable code")

            return YI_34B_TEMPLATE.format(
                system_prompt=system_prompt, user_message=user_message
            )

        else:
            raise ValueError("Invalid model name")

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(10))
    def hfapi_decode(
            self,
            input: list,
            max_length: int = 4096,
            temp: float = 0.1,
            index=1
    ):
        # Get endpoint
        """Suggested models:
        - meta-llama/Llama-2-70b-chat-hf
        - mistralai/Mixtral-8x7B-Instruct-v0.1
        - mistralai/Mistral-7B-Instruct-v0.2
        """
        assert temp > 0

        input = self.encode_hfapi(input)
        SILVIU_TOKEN = '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'
        BLAIR_TOKEN = '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'
        tokens = [
            SILVIU_TOKEN,
            BLAIR_TOKEN,
        ]
        model = InferenceClient(model=self.model_name, token=random.choice(tokens))
        model.headers["x-use-cache"] = "0"  # dont cache
        output = model.post(
            json={
                "inputs"    : input,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature"   : temp,
                    "do_sample"     : True if temp > 0 else False,
                },
            }
        )

        # This no longer works
        text = json.loads(output)[0]["generated_text"].strip()

        if "[/INST]" in text:
            assert text.count("[/INST]") == 1
            text = text.split("[/INST]")[1].strip()
            return text
        return text

    def __call__(self, prompt, use_json=False, timeout=10) -> Any:
        if type(prompt) == str or (type(prompt) == list and len(prompt) == 1):
            prompt = [{"role": "user", "content": prompt}]
            # print('Single promp')
            # print(f'{prompt = }')
            rslt = self.hfapi_decode(prompt)

            # print(rslt)

            if use_json:
                if type(prompt) == str:
                    rslt = extract_json(rslt)

                else:
                    # rslt = [extract_json(r) for r in rslt]
                    for i in range(len(rslt)):
                        try:
                            rslt[i] = extract_json(rslt[i])
                        except Exception as e:
                            print(e)
                            print(rslt[i])

            return rslt

        elif type(prompt) == list:

            # print('Batched prompt')
            # print(f'{prompt = }')

            assert len(prompt) > 0
            prompts = []
            for p in prompt:
                prompts.append([{"role": "user", "content": p}])

            rslts = self._batched_inference(prompts)

            if use_json:
                rslts = [extract_json(rslt) for rslt in rslts]

            return rslts

        else:
            raise ValueError("Invalid prompt type")

    def _batched_inference(self, prompts, batch_size=8):
        """Inference with batching"""
        with ThreadPoolExecutor(max_workers=max(batch_size, len(prompts))) as executor:
            # Need to make sure order is preserved
            futures = [executor.submit(self.hfapi_decode, prompts[i]) for i in range(len(prompts))]
            # results = [future.result() for future in futures]
            # results = [None] * len(prompts)
            results = []
            # wait(futures)

            # Preserve order

            for future in tqdm(as_completed(futures)):
                result = future.result()
                if result is not None:
                    # print(result)
                    results.append(result)

        return results




import os
import openai
import numpy as np
from datasets import load_dataset
import json
import tqdm
# import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import multiprocessing as mp
import time

dotenv.load_dotenv(override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORGANIZATION')
if os.getenv('OPENAI_API_TYPE') is not None:
  openai.api_type = os.getenv('OPENAI_API_TYPE')
if os.getenv('OPENAI_API_BASE') is not None:
  openai.api_base = os.getenv('OPENAI_API_BASE')
if os.getenv('OPENAI_API_VERSION') is not None:
  openai.api_version = os.getenv('OPENAI_API_VERSION')


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=(retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)),

)
def chat_decode(input: list, max_length: int = 4096, temp: float = 0.3, stop: str | list[str] | None = None, n: int = 1, engine='gpt-4-turbo'):
    if openai.api_type == 'azure':
      response = openai.ChatCompletion.create(
        engine=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    else:
      response = openai.ChatCompletion.create(
        model=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    
    return [response["choices"][j]["message"]["content"] for j in range(len(response["choices"]))]

if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=300)
  parser.add_argument('--tag', type=str, default='')
  args = parser.parse_args()

  SYSTEM = "You are an expert at evaluating the capabilities, biases, and response patterns of AI assistants with respect to specific topics or skills."

  with open('./RPR/topicgen/build_dataset.prompt', 'r') as f:
    TEMPLATE = f.read()


  for t in TOPICS:
    results_file = f'./RPR/topicgen/dataset_{t}.json'
  
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
      model = GPTModel(system_prompt=SYSTEM, model_name='gpt-4-turbo')
      sample = model(messages[1], use_json=False, temperature=0.3, timeout=10)

    except Exception as e:
      print(e)
      continue

    try:
      j = json.loads(sample.split('===')[1])
      
    except Exception as e:
      print(sample.split('===')[1])
      print(e)
      continue
    
    res = j

    with open(results_file, 'w') as f:
      json.dump(res, f, indent=2)
    
    time.sleep(3)