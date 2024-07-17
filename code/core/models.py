from __future__ import annotations

import copy
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, Dict, Union, Optional, Self
import time
from dashscope import Generation

import google.generativeai as genai
import numpy as np
from http import HTTPStatus
import torch
from anthropic import Anthropic
from groq import Groq
from huggingface_hub import InferenceClient
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from .config import *

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

SUPPORTED_OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]

SUPPORTED_CLAUDE_MODELS = [
    CLAUDE_3_OPUS,
    CLAUDE_3_SONNET,
    CLAUDE_3_HAIKU,
]

SUPPORTED_YI_MODELS = [
    '01-ai/Yi-1.5-34B-Chat'
]

SUPPORTED_QWEN_API_MODELS = [
 'qwen2-72b-instruct',
]

SUPPORTED_MISTRAL_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"
]
SUPPORTED_NOUS_MODELS = ["NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"]
SUPPORTED_LLAMA_MODELS = [
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
]

SUPPORTED_GROQ_MODELS = [
    "groq/llama3-8b-8192",
    "groq/llama3-70b-8192",
    "groq/mixtral-8x7b-32768",
    "groq/gemma-7b-it",
]

SUPPORTED_LLAMA3_MODELS = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]

SUPPORTED_PHI_MODELS = [
    'microsoft/Phi-3-mini-4k-instruct']

# suppress logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def select_model(
    model_name: str, system_prompt: str, cm: CostManager = None, **kwargs
) -> Model:
    if model_name in SUPPORTED_OPENAI_MODELS:
        return GPTModel(
            system_prompt,
            model_name,
            tm=cm,
            api_key=kwargs.get("api_key", OPENAI_API_KEY),
            base_url=kwargs.get("base_url", None),
        )
    elif model_name in SUPPORTED_YI_MODELS:
        return YIModel(system_prompt, model_name)
    elif model_name in SUPPORTED_CLAUDE_MODELS:
        return ClaudeModel(system_prompt, model_name, tm=cm)
    elif model_name in SUPPORTED_MISTRAL_MODELS:
        return MistralModel(system_prompt, model_name)
    elif model_name in SUPPORTED_NOUS_MODELS:
        return NousHermesModel(system_prompt, model_name)
    elif model_name in SUPPORTED_LLAMA_MODELS:
        return LlamaModel(system_prompt, model_name)
    elif model_name in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]:
        return Llama3Model(system_prompt, model_name)
    elif model_name in SUPPORTED_GROQ_MODELS:
        return GroqModel(system_prompt, model_name)
    elif model_name in ["google/gemma-1.1-7b-it", 'google/gemma-2-27b-it', 'google/gemma-2-9b-it']:
        return GemmaModel(system_prompt, model_name)
    elif model_name in SUPPORTED_PHI_MODELS:
        return PhiModel(system_prompt, model_name)
    elif model_name in SUPPORTED_QWEN_API_MODELS:
        return QwenAPIModel(system_prompt, model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


class Model(ABC):
    name: str
    messages: List[Dict[str, str]]
    system_prompt: str

    def __init__(self, model_name: str, system_prompt: str):
        self.name = model_name
        self.system_prompt = system_prompt
        self.messages = [{"role": ROLE_SYSTEM, "content": system_prompt}]

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Union[str, Dict]:
        raise NotImplementedError

    def get_logits(self, prompt, post_fix=" "):
        raise NotImplementedError

    def add_message(self, role: str, content: str):
        assert role in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]
        self.messages.append({"role": role, "content": content})

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

    def record_tokens(
        self, in_tokens: int, out_tokens: int, model_name: str = GPT_4_MODEL_NAME
    ):
        self.tokens[model_name][0] += in_tokens
        self.tokens[model_name][1] += out_tokens

    def get_cost(self) -> float:
        return (
            self.tokens[GPT_4_MODEL_NAME][0] / 1000 * 0.005
            + self.tokens[GPT_4_MODEL_NAME][1] / 1000 * 0.015
            + self.tokens[GPT_3_MODEL_NAME][0] / 1000 * 0.001
            + self.tokens[GPT_3_MODEL_NAME][1] / 1000 * 0.002
        )

    def get_info_dict(self) -> Dict:
        return {"total_cost": self.get_cost(), "details": self.tokens}


class ClaudeModel(Model):
    client: Anthropic

    def __init__(
        self,
        system_prompt: str,
        model_name: str = CLAUDE_3_MODEL_NAME,
        tm: CostManager = None,
        api_key: str = CLAUDE_API_KEY,
    ):
        # super().__init__(model_name, system_prompt)
        self.name = model_name
        self.system_prompt = system_prompt
        self.messages = []
        self.client = Anthropic(api_key=api_key)

        self.total_in_tokens = 0
        self.total_out_tokens = 0
        self.tm = None
        self.name = model_name

        self.json_transcriber = GPTModel(
            "You are a good json transcriber, you are good at copying texts. Copy the text to the given json format.",
            GPT_3_MODEL_NAME,
            tm=tm,
        )

    # def __del__(self):
    #     self.client.close()

    def __call__(
        self,
        user_prompt: str,
        *args,
        use_json: bool = False,
        temperature: float = 0.0,
        timeout: float = None,
        **kwargs,
    ) -> Union[str, Dict]:
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
                # for criterion in json_obj:
                #     assert all(
                #         [
                #             field in json_obj[criterion]
                #             for field in [
                #                 "overview",
                #                 "thinking_pattern",
                #                 "strength",
                #                 "weakness",
                #             ]
                #         ]
                #     )
                #     pass

            except:
                # if True:
                prompt = (
                    f"Please convert the following text to a json format:\n\n{content}"
                )
                prompt += """
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
                    """
                json_obj = self.json_transcriber(content, use_json=True)

            return json_obj

        return content

    def extract_json(self, text):
        # match first { and last }
        starting_ind = text.find("{")
        ending_ind = text.rfind("}")
        if starting_ind == -1 or ending_ind == -1:
            # use gpt-3.5 to transcribe the text to json
            transcriber = GPTModel(
                "You are a good json transcriber, you are good at copying texts. Copy the text to the given json format.",
                GPT_3_MODEL_NAME,
            )
            return transcriber(text, use_json=True)

        return json.loads(text[starting_ind : ending_ind + 1])

    def get_response(
        self, temperature: float, use_json: bool, timeout: float = None
    ) -> ChatCompletion:
        message = self.client.messages.create(
            max_tokens=4096,
            system=self.system_prompt,
            messages=self.messages,
            model=self.name,
        )
        return message

    @classmethod
    def from_messages(
        cls, messages: List[Dict[str, str]], model_name: str, tm: CostManager = None
    ) -> Self:
        assert (
            len(messages) >= 1
            and all(
                [
                    m["role"] in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]
                    for m in messages
                ]
            )
            and messages[0]["role"] == ROLE_USER
        )
        system_prompt = messages[0]["content"]
        model = cls(system_prompt, model_name, tm=tm)
        model.messages = copy.deepcopy(messages)
        return model


class GPTModel(Model):
    client: OpenAI

    total_in_tokens: int
    total_out_tokens: int
    tm: Optional[CostManager]

    def __init__(
        self,
        system_prompt: str,
        model_name: str = GPT_MODEL_NAME,
        tm: CostManager = None,
        api_key: str = OPENAI_API_KEY,
        base_url: str = None,
    ):
        super().__init__(model_name, system_prompt)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.total_in_tokens = 0
        self.total_out_tokens = 0
        self.tm = tm

    def __del__(self):
        self.client.close()

    def __call__(
        self,
        user_prompt: str,
        *args,
        use_json: bool = False,
        temperature: float = 0.0,
        timeout: float = None,
        **kwargs,
    ) -> Union[str, Dict]:
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
            self.tm.record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                self.name,
            )
        self.total_out_tokens += response.usage.completion_tokens
        self.total_in_tokens += response.usage.prompt_tokens
        if use_json:
            try:
                return json.loads(content)
            except:
                # manually parse the json, check first { and last }
                starting_ind = content.find("{")
                ending_ind = content.rfind("}")
                if starting_ind != -1 and ending_ind != -1:
                    return json.loads(content[starting_ind : ending_ind + 1])
                else:
                    raise ValueError("Cannot parse JSON")
        return content

    def get_response(
        self, temperature: float, use_json: bool, timeout: float = None
    ) -> ChatCompletion:
        if use_json:
            return self.client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
        else:
            return self.client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
            )

    @classmethod
    def from_messages(
        cls, messages: List[Dict[str, str]], model_name: str, tm: CostManager = None
    ) -> Self:
        assert (
            len(messages) >= 1
            and all(
                [
                    m["role"] in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]
                    for m in messages
                ]
            )
            and messages[0]["role"] == ROLE_SYSTEM
        )
        system_prompt = messages[0]["content"]
        model = cls(system_prompt, model_name, tm=tm)
        model.messages = copy.deepcopy(messages)
        return model


class HFAPIModel(Model):

    def __init__(self, model_name: str, system_prompt: str) -> None:
        super().__init__(model_name, system_prompt)
        self.api_key_index = random.randint(0, len(HF_API_TOKENS))

    def __call__(
        self,
        user_prompt: str,
        *args,
        use_json: bool = False,
        temperature: float = 0,
        timeout: float = None,
        cache: bool = False,
        json_retry_count: int = 5,
        max_new_tokens=1024,
        api_key_index: int = 0,
        **kwargs,
    ) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, will try its best to return a json dict, but not guaranteed.
        If we cannot parse the JSON, we will return the response string directly.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(
            temperature, use_json, timeout, cache, max_new_tokens
        )
        if use_json:
            found_json = False
            for i in range(json_retry_count):
                # cache only if both instruct to do and first try
                response = self.get_response(
                    temperature, use_json, timeout, cache and i == 0, max_new_tokens
                )
                json_obj = self.find_first_valid_json(response)
                if json_obj is not None:
                    response = json_obj
                    found_json = True
                    break

            if not found_json:
                # if cannot parse json, use another model to transcribe
                transcriber = Llama3Model(system_prompt="Transcribe the following text to JSON format.", 
                                          model_name="meta-llama/Meta-Llama-3-8B-Instruct")
                
                response = transcriber(f'{response}\nMake sure that your output is a VALID json, do not put extra tokens', use_json=False)
                # try to laod this again
                json_obj = self.find_first_valid_json(response)
                if json_obj is not None:
                    response = json_obj
                else:
                    raise ValueError("Cannot parse JSON")
                    
                    
        self.api_key_index = api_key_index
        self.add_message(ROLE_ASSISTANT, response)
        return response

    @retry(
        stop=stop_after_attempt(6), wait=wait_random_exponential(max=15), reraise=True
    )  # retry if exception
    def get_response(
        self,
        temperature: float,
        use_json: bool,
        timeout: float,
        cache: bool,
        max_new_tokens=1024,
    ) -> str:
        client = InferenceClient(
            model=self.name, token=HF_API_TOKENS[self.api_key_index  % len(HF_API_TOKENS)], timeout=timeout
        )
        if not cache:
            client.headers["x-use-cache"] = "0"

        r = client.text_generation(
            self.format_messages(),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            stop_sequences=["<|eot_id|>"],
            max_new_tokens=max_new_tokens,
        )

        return r

    def get_logits(self, prompt, post_fix=" "):
        self.messages.append({"role": ROLE_USER, "content": prompt})
        r = self.format_messages()
        client = InferenceClient(model=self.name, token=random.choice(HF_API_TOKENS))
        r += post_fix
        output = client.post(
            json={
                "inputs": r,
                "parameters": {"top_n_tokens": 5, "details": True, "max_new_tokens": 1},
            }
        )
        top_tokens = json.loads(output)[0]["details"]["top_tokens"][0]
        top_tokens = {t["text"]: t["logprob"] for t in top_tokens}

        return top_tokens

    @abstractmethod
    def format_messages(self) -> str:
        raise NotImplementedError

    def get_short_name(self) -> str:
        """
        Returns the last part of the model name.
        For example, "mistralai/Mixtral-8x7B-Instruct-v0.1" -> "Mixtral-8x7B-Instruct-v0.1"
        """
        return self.name.split("/")[-1]

    @staticmethod
    def find_first_valid_json(s) -> Optional[Dict]:
        s = re.sub(
            r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', lambda m: m.group(0)[1:], s
        )  # remove all invalid escapes chars
        for i in range(len(s)):
            if s[i] != "{":
                continue
            for j in range(i + 1, len(s) + 1):
                if s[j - 1] != "}":
                    continue
                try:
                    potential_json = s[i:j]
                    json_obj = json.loads(potential_json, strict=False)
                    return json_obj  # Return the first valid JSON object found
                except json.JSONDecodeError:
                    pass  # Continue searching if JSON decoding fails
        return None  # Return None if no valid JSON object is found


class MistralModel(HFAPIModel):

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ) -> None:
        assert model_name in [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1"
        ], "Model not supported"
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        # mistral doesn't support system prompt, so we need to convert it to user prompt
        if messages[0]["role"] == ROLE_SYSTEM:
            assert len(self.messages) >= 2
            messages = [
                {
                    "role": ROLE_USER,
                    "content": messages[0]["content"] + "\n" + messages[1]["content"],
                }
            ] + messages[2:]
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        # print(r)
        return r
    
    

class PhiModel(HFAPIModel):
    
        def __init__(
            self,
            system_prompt: str,
            model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        ) -> None:
            assert model_name in [
                "microsoft/Phi-3-mini-4k-instruct",
                "microsoft/Phi-3-medium-128k-instruct",
                "microsoft/Phi-3-medium-4k-instruct",
                'microsoft/Phi-3-small-128k-instruct',
                'microsoft/Phi-3-small-8k-instruct',
            ], "Model not supported"
            super().__init__(model_name, system_prompt)
    
        def format_messages(self) -> str:
            messages = self.messages
            assert len(messages) >= 2  # must be at least a system and a user
            assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
            tokenizer = AutoTokenizer.from_pretrained(self.name)
            r = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, max_length=4096
            )
            return r
        
        def __call__(self, user_prompt: str, *args, use_json: bool = False, temperature: float = 0, timeout: float = None, cache: bool = False, json_retry_count: int = 5, max_new_tokens=1024, **kwargs) -> str | Dict:
            rslt = super().__call__(user_prompt, *args, use_json=use_json, temperature=temperature, timeout=timeout, cache=cache, json_retry_count=json_retry_count, max_new_tokens=max_new_tokens, **kwargs)
            rslt = rslt.replace("<|end|>", "")
            return rslt

class NousHermesModel(HFAPIModel):

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "ap",
    ) -> None:
        assert model_name in [
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        ], "Model not supported"
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        # print(r)
        return r


class LlamaModel(HFAPIModel):

    def __init__(
        self, system_prompt: str, model_name: str = "meta-llama/Llama-2-70b-chat-hf"
    ) -> None:
        assert model_name in [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-instruct",
            "meta-llama/Meta-Llama-3-70B" "meta-llama/Meta-Llama-3-70B-Instruct",
        ], "Model not supported"
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
            role, content = msg["role"], msg["content"]
            if role == ROLE_SYSTEM:
                assert ValueError
            elif role == ROLE_USER:
                if r.endswith("</s>"):
                    r += "<s>"
                r += f"[INST] {content} [/INST]"
            elif role == ROLE_ASSISTANT:
                r += f"{content}</s>"
            else:
                raise ValueError
        return r


class GroqModel(Model):

    def __init__(
        self,
        system_prompt: str,
        model_name: str = GPT_MODEL_NAME,
    ):
        assert model_name in SUPPORTED_GROQ_MODELS
        model_name = model_name.split("/")[1]
        super().__init__(model_name, system_prompt)

    def __call__(
        self,
        user_prompt: str,
        *args,
        use_json: bool = False,
        temperature: float = 0.0,
        timeout: float = None,
        **kwargs,
    ) -> Union[str, Dict]:
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

    def get_response(
        self, client: Groq, temperature: float, use_json: bool, timeout: float = None
    ) -> ChatCompletion:
        if use_json:
            return client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
        else:
            return client.chat.completions.create(
                messages=self.messages,
                model=self.name,
                temperature=temperature,
                timeout=timeout,
            )


class Llama3Model(HFAPIModel):

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    ) -> None:
        super().__init__(model_name, system_prompt)
        self.api_key_index = random.randint(0, len(HF_API_TOKENS))

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        # print(messages)
        tokenizer.eos_token = "<|eot_id|>"
        tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        return r

    def get_logits(self, prompt, post_fix=" "):
        self.messages.append({"role": ROLE_USER, "content": prompt})
        r = self.format_messages()
        client = InferenceClient(model=self.name, token=random.choice(HF_API_TOKENS))
        r += post_fix
        output = client.post(
            json={
                "inputs": r,
                "parameters": {"top_n_tokens": 5, "details": True, "max_new_tokens": 1},
            }
        )
        top_tokens = json.loads(output)[0]["details"]["top_tokens"][0]
        top_tokens = {t["text"]: t["logprob"] for t in top_tokens}

        return top_tokens

class YIModel(HFAPIModel):
    def __init__(
    self,
    system_prompt: str,
    model_name: str,
    ) -> None:
        super().__init__(model_name, system_prompt)
        self.prompt_template = """
        <|System|>
        {sys_prompt}
        <|Human|>
        {user_prompt}
        <|Assistant|>
        """

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        # print(r)
        return r


class GemmaModel(HFAPIModel):

    def __init__(
        self, system_prompt: str, model_name: str = "google/gemma-1.1-7b-it"
    ) -> None:
        assert model_name in ["google/gemma-1.1-7b-it", 'google/gemma-2-27b-it', 'google/gemma-2-9b-it'], "Model not supported"
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        # assert len(messages) >= 2  # must be at least a system and a user
        # assert messages[0]['role'] == ROLE_SYSTEM and messages[1]['role'] == ROLE_USER
        # make sure no system prompt exists, if yes, replace with user
        if messages[0]["role"] == ROLE_SYSTEM:
            assert len(self.messages) >= 2
            messages = [
                {
                    "role": ROLE_USER,
                    "content": messages[0]["content"] + "\n" + messages[1]["content"],
                }
            ] + messages[2:]
        tokenizer = AutoTokenizer.from_pretrained(self.name,
                                                  token=HF_API_TOKENS[self.api_key_index % len(HF_API_TOKENS)])
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        return r


# TODO: refactor HFLocalModel
CHAT_FORMATS = {
    "mistralai": "<s>[INST] {prompt} [/INST]",
    "openchat": "GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:",
    "meta-llama": """[INST] <<SYS>>
You answer questions directly.
<</SYS>>
{prompt}[/INST]""",
    "mosaicml": """<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant""",
    "lmsys": "USER: {prompt}\nASSISTANT:",
}


class HFLocalModel:

    def __init__(self, model_name: str, bits: int = 4) -> None:
        self.model_name = model_name

        device_map = {"": 0}
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, "float16"),
                bnb_4bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
            )

        elif bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=getattr(torch, "float16"),
                bnb_8bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=2048,
        )

    def prob_eval(self, sentences, post_fix=" ", labels=None, ground_truth=[]):

        assert len(ground_truth) == len(sentences)

        inf_rslt, logits_batched, sentences = self.prob_inference(
            sentences, post_fix, labels
        )

        correct_list = []
        for i in range(len(sentences)):
            correct_list.append(
                np.argmax(logits_batched[i]) == labels.index(ground_truth[i])
            )
        correct = np.sum(correct_list)
        return correct, logits_batched, sentences

    def prob_inference(self, sentences, post_fix=" ", labels=None):
        if labels is None:
            labels = ["A", "B"]

        sentences = [
            CHAT_FORMATS[self.model_name.split("/")[0]].format(prompt=s)
            for s in sentences
        ]
        sentences = [s + post_fix for s in sentences]

        with torch.no_grad():
            encoded_tokenized = self.tokenizer(
                sentences, padding=True, return_tensors="pt"
            ).to("cuda")
            rslt = self.model(encoded_tokenized.input_ids)

            logits = rslt.logits

            logits_batched = np.zeros(shape=(len(sentences), len(labels)))

            inf_rslt = []
            for i in range(len(sentences)):
                probs = [
                    logits[i, -1, self.tokenizer.encode(labels[j])[1]].item()
                    for j in range(len(labels))
                ]
                probs = np.exp(probs) / np.sum(np.exp(probs))
                logits_batched[i] = np.array(probs)
                inf_rslt.append(np.argmax(logits_batched[i]))

            return inf_rslt, logits_batched, sentences

    def __call__(self, prompt) -> Any:
        llm_type = self.model_name.split("/")[0]
        prompt_format = CHAT_FORMATS[llm_type]
        formatted_prompt = prompt_format.format(prompt=prompt)
        result = self.pipe(formatted_prompt)
        return result[0]["generated_text"][len(formatted_prompt) + 1 :]

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
    text = text.replace("\\", "\\\\")

    try:
        rslt = json.loads(text)
    except Exception as e:
        # print(e)
        # print(text)
        rslt = None
    return rslt


class GeminiModel(Model):
    def __init__(
        self, system_prompt: str, model_name: str = "gemini-1.5-pro-latest"
    ) -> None:
        super().__init__(model_name, system_prompt)
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=model_name, system_instruction=system_prompt
        )

    def __call__(self, user_prompt: str, use_json=False, *args):
        # does nto support multi_turn
        response = self.model.generate_content(user_prompt)

        return response.text

class QwenAPIModel(Model):
    def __init__(self, system_prompt: str, model_name: str, api_key: str =QWEN_API_KEY, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(model_name, system_prompt)
        self.api_key = api_key
        self.base_url = base_url

    def __call__(self, user_prompt: str, *args, use_json: bool = False, **kwargs) -> Union[str, Dict]:
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response()
        content = response
        self.add_message(ROLE_ASSISTANT, content)
        time.sleep(30)
        return content
        

    def get_response(self) -> str:
        try:
            response = Generation.call(
                model=self.name,
                messages=self.messages,
                api_key=self.api_key,
                base_url=self.base_url,
                result_format='message',
                seed=random.randint(1, 10000)
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API call failed with status code {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    @classmethod
    def from_messages(cls, messages: List[Dict[str, str]], model_name: str, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> 'QwenAPIModel':
        assert len(messages) >= 1 and messages[0]["role"] == ROLE_SYSTEM
        system_prompt = messages[0]["content"]
        model = cls(system_prompt, model_name, api_key, base_url)
        model.messages = messages.copy()
        return model


if __name__ == "__main__":
    model = GPTModel(system_prompt="You are a helpful assistant.", model_name="gpt-3.5-turbo")
    # print(model("hello!"))
    rslt = (model('How many parameters do you have? respond in json: {{"parameter": "value"}}'))
    print(rslt)