from __future__ import annotations

import copy
import json
import logging
import random
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Union, Optional, Self

import huggingface_hub.errors
from anthropic import Anthropic
from anthropic.types import Message
from groq import Groq
from huggingface_hub import InferenceClient
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception,
    wait_random_exponential,
)
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from core.config import *

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

SUPPORTED_OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
]

SUPPORTED_CLAUDE_MODELS = [
    CLAUDE_3_5_SONNET,
    CLAUDE_3_OPUS,
    CLAUDE_3_SONNET,
    CLAUDE_3_HAIKU,
]

SUPPORTED_MISTRAL_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
SUPPORTED_NOUS_MODELS = ["NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"]
SUPPORTED_LLAMA_MODELS = [
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
]

SUPPORTED_GROQ_MODELS = [  # no longer supported
    # "groq/llama3-8b-8192",
    # "groq/llama3-70b-8192",
    # "groq/mixtral-8x7b-32768",
    # "groq/gemma-7b-it",
]

SUPPORTED_LLAMA3_MODELS = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
]

SUPPORTED_PHI_MODELS = ["microsoft/Phi-3-mini-4k-instruct"]

# suppress logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def select_model(
    model_name: str, system_prompt: str, cm: CostManager = None, **kwargs
) -> Model:
    if model_name in SUPPORTED_OPENAI_MODELS:
        return GPTModel(
            system_prompt,
            model_name,
            cm=cm,
            api_key=kwargs.get("api_key", OPENAI_API_KEY),
            base_url=kwargs.get("base_url", None),
        )
    elif model_name in SUPPORTED_CLAUDE_MODELS:
        return ClaudeModel(
            system_prompt,
            model_name,
            cm=cm,
            api_key=kwargs.get("api_key", CLAUDE_API_KEY),
        )
    elif model_name in SUPPORTED_MISTRAL_MODELS:
        return MistralModel(system_prompt, model_name)
    elif model_name in SUPPORTED_NOUS_MODELS:
        return NousHermesModel(system_prompt, model_name)
    elif model_name in SUPPORTED_LLAMA_MODELS:
        return LlamaModel(system_prompt, model_name)
    elif model_name in SUPPORTED_LLAMA3_MODELS:
        return Llama3Model(system_prompt, model_name)
    elif model_name in ["google/gemma-1.1-7b-it"]:
        return GemmaModel(system_prompt, model_name)
    elif model_name in SUPPORTED_PHI_MODELS:
        return PhiModel(system_prompt, model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


def select_model_from_messages(
    model_name: str,
    messages: List[Dict[str, str]],
    cm: CostManager = None,
    **kwargs,
):
    assert (
        len(messages) >= 1
        and all(
            [m["role"] in [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT] for m in messages]
        )
        and messages[0]["role"] == ROLE_SYSTEM
    )
    system_prompt = messages[0]["content"]
    model = select_model(model_name, system_prompt, cm, **kwargs)
    model.messages = copy.deepcopy(messages)
    return model


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


# cost per 1000000 (1M) tokens
MODEL_TO_COST = {
    #          in, out
    "gpt-4o": (5, 15),
    "gpt-4o-2024-05-13": (5, 15),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o-mini-2024-07-18": (0.15, 0.6),
    "gpt-4-turbo": (10, 30),
    "gpt-3.5-turbo": (0.5, 1.5),
    CLAUDE_3_5_SONNET: (3, 15),
    CLAUDE_3_OPUS: (15, 75),
    CLAUDE_3_HAIKU: (0.25, 1.25),
    CLAUDE_3_SONNET: (3, 15),
}


class CostManager:
    #                              in , out
    tokens: defaultdict[str, List[int]]

    def __init__(self):
        self.tokens = defaultdict(lambda: [0, 0])

    def record_tokens(self, in_tokens: int, out_tokens: int, model_name: str):
        self.tokens[model_name][0] += in_tokens
        self.tokens[model_name][1] += out_tokens

    def get_cost(self) -> float:
        total_cost = 0.0
        for model_name in self.tokens:
            in_tokens, out_tokens = self.tokens[model_name]
            in_cost, out_cost = MODEL_TO_COST[model_name]
            total_cost += (
                in_tokens / 1000000 * in_cost + out_tokens / 1000000 * out_cost
            )
        return total_cost

    def get_info_dict(self) -> Dict:
        return {"total_cost": self.get_cost(), "details": self.tokens}

    @classmethod
    def from_dict(cls, d: Dict) -> Self:
        cm = cls()
        for model_name in d["details"]:
            cm.tokens[model_name] = d["details"][model_name]
        return cm

    def __add__(self, other):
        if not isinstance(other, CostManager):
            raise ValueError("Can only add CostManager objects")
        new_cm = CostManager()
        new_cm.tokens = copy.deepcopy(self.tokens)
        for model_name in other.tokens:
            new_cm.tokens[model_name][0] += other.tokens[model_name][0]
            new_cm.tokens[model_name][1] += other.tokens[model_name][1]
        return new_cm


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


class ClaudeModel(Model):
    client: Anthropic
    cm: Optional[CostManager]

    def __init__(
        self,
        system_prompt: str,
        model_name: str = CLAUDE_3_MODEL_NAME,
        cm: CostManager = None,
        api_key: str = CLAUDE_API_KEY,
    ):
        super().__init__(model_name, system_prompt)
        self.client = Anthropic(api_key=api_key)
        self.cm = cm

    def __del__(self):
        self.client.close()

    def __call__(
        self,
        user_prompt: str,
        use_json: bool = False,
        temperature: float = 1.0,
        timeout: float = None,
    ) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, returns a json dict.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(temperature, timeout)
        content = response.content[0].text
        self.add_message(ROLE_ASSISTANT, content)
        # record tokens
        if self.cm is not None:
            self.cm.record_tokens(
                response.usage.input_tokens,
                response.usage.output_tokens,
                self.name,
            )

        if use_json:
            json_obj = find_first_valid_json(content)
            return json_obj

        return content

    def get_response(self, temperature: float, timeout: float = None) -> Message:
        message = self.client.messages.create(
            max_tokens=4096,
            temperature=temperature,
            system=self.system_prompt,
            messages=self.messages[1:],  # messages shouldn't include system here
            model=self.name,
            timeout=timeout,
        )
        return message


class GPTModel(Model):
    client: OpenAI
    cm: Optional[CostManager]

    def __init__(
        self,
        system_prompt: str,
        model_name: str = GPT_MODEL_NAME,
        cm: CostManager = None,
        api_key: str = OPENAI_API_KEY,
        base_url: str = None,
    ):
        super().__init__(model_name, system_prompt)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.cm = cm

    def __del__(self):
        self.client.close()

    def __call__(
        self,
        user_prompt: str,
        use_json: bool = False,
        temperature: float = 1.0,
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
        if self.cm is not None:
            self.cm.record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                self.name,
            )
        if use_json:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
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


def after_retry(retry_state):
    # Check if it is not the first attempt and if the last attempt raised an exception
    if retry_state.attempt_number > 1 and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        exception_type = type(exception).__name__  # Get the type name of the exception
        exception_message = str(exception)  # Get the message of the exception
        # Print both the type and message of the exception
        print(
            f"Retry #{retry_state.attempt_number - 1} due to {exception_type}: {exception_message}"
        )


def should_retry(exception):
    """Return True if the retry should occur, False otherwise."""
    return not isinstance(exception, huggingface_hub.errors.ValidationError)


# round robin index for HF API tokens
hf_token_index: int = 0

_tokenizer_cache = {}
_tokenizer_cache_lock: threading.Lock = threading.Lock()


def get_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    with _tokenizer_cache_lock:
        if model_name in _tokenizer_cache:  # need double-check
            return _tokenizer_cache[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in SUPPORTED_LLAMA3_MODELS and "3.1" not in model_name:
            # only for llama3, not 3.1
            tokenizer.eos_token = "<|eot_id|>"
            tokenizer.add_special_tokens({"eos_token": "<|eot_id|>"})
        _tokenizer_cache[model_name] = tokenizer
    return tokenizer


class HFAPIModel(Model):

    def __call__(
        self,
        user_prompt: str,
        use_json: bool = False,
        temperature: float = 1.0,
        timeout: float = 60 * 5,
        cache: bool = True,
        json_retry_count: int = 5,
        max_new_tokens=2048,
    ) -> Union[str, Dict]:
        """
        Returns the model's response.
        If use_json = True, will try its best to return a json dict, but not guaranteed.
        If we cannot parse the JSON, we will return the response string directly.
        """
        self.add_message(ROLE_USER, user_prompt)
        response = self.get_response(temperature, timeout, cache, max_new_tokens)
        if use_json:
            for i in range(json_retry_count):
                json_obj = find_first_valid_json(response)
                if json_obj is not None:
                    response = json_obj
                    break
                print(f"Failed to parse JSON, retrying... {i + 1}/{json_retry_count}")
                response = self.get_response(
                    temperature, timeout, False, max_new_tokens
                )
        self.add_message(ROLE_ASSISTANT, response)
        return response

    @retry(
        stop=stop_after_attempt(1000),
        wait=wait_random_exponential(1.5, 30, 2, 1),
        reraise=True,
        after=after_retry,
        retry=retry_if_exception(should_retry),
    )  # retry if exception
    def get_response(
        self,
        temperature: float,
        timeout: float,
        cache: bool,
        max_new_tokens=1024,
    ) -> str:
        global hf_token_index
        hf_token_index = (hf_token_index + 1) % len(HF_API_TOKENS)
        # print(hf_token_index)
        client = InferenceClient(
            model=self.name,
            token=HF_API_TOKENS[hf_token_index],
            timeout=timeout,
            # model=self.name, token=HF_API_TOKENS[6], timeout=timeout
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
        # print(r)

        return r

    # def get_logit(self, prompt):
    #     client = InferenceClient(model=self.name, token=random.choice(HF_API_TOKENS))
    #     self.add_message(ROLE_USER, prompt)
    #     r = client.text_generation(
    #         self.format_messages(),
    #         do_sample=False,
    #         max_new_tokens=1,
    #         details=True,
    #         best_of=2,
    #     )
    #     print(r)

    @abstractmethod
    def format_messages(self) -> str:
        raise NotImplementedError

    def get_short_name(self) -> str:
        """
        Returns the last part of the model name.
        For example, "mistralai/Mixtral-8x7B-Instruct-v0.1" -> "Mixtral-8x7B-Instruct-v0.1"
        """
        return self.name.split("/")[-1]


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
        tokenizer = get_tokenizer(self.name)
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
        ], "Model not supported"
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = get_tokenizer(self.name)
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        return r

    def __call__(
        self,
        user_prompt: str,
        *args,
        use_json: bool = False,
        temperature: float = 1.0,
        timeout: float = None,
        cache: bool = False,
        json_retry_count: int = 5,
        max_new_tokens=1024,
        **kwargs,
    ) -> str | Dict:
        rslt = super().__call__(
            user_prompt,
            use_json=use_json,
            temperature=temperature,
            timeout=timeout,
            cache=cache,
            json_retry_count=json_retry_count,
            max_new_tokens=max_new_tokens,
        )
        rslt = rslt.replace("<|end|>", "")
        return rslt


class NousHermesModel(HFAPIModel):

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    ) -> None:
        assert model_name in [
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        ], "Model not supported"
        super().__init__(model_name, system_prompt)

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = get_tokenizer(self.name)
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
        temperature: float = 1.0,
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

    def format_messages(self) -> str:
        messages = self.messages
        assert len(messages) >= 2  # must be at least a system and a user
        assert messages[0]["role"] == ROLE_SYSTEM and messages[1]["role"] == ROLE_USER
        tokenizer = get_tokenizer(self.name)
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


class GemmaModel(HFAPIModel):

    def __init__(
        self, system_prompt: str, model_name: str = "google/gemma-1.1-7b-it"
    ) -> None:
        assert model_name in ["google/gemma-1.1-7b-it"], "Model not supported"
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
        tokenizer = get_tokenizer(self.name)
        r = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, max_length=4096
        )
        return r


if __name__ == "__main__":
    cm = CostManager()
    m = ClaudeModel("You are a helpful assistant.", model_name=CLAUDE_3_HAIKU, cm=cm)
