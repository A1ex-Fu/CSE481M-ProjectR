import model_init
from transformers import pipeline
from langchain import HuggingFacePipeline
import json
from langchain.schema import LLMResult, Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
from typing import Optional, List, Any, Mapping
import requests
import os
from openai import OpenAI

def read_json(path):

    with open(path, 'r') as fr:
        file = json.load(fr)
    return file


def llm_init_langchain(config, max_new_tokens, seed):

    if config['model_type'] == 'deepseek':
        return DeepSeekLLM(
            api_key="sk-6624f2437ba84a0dab5fb0586c6d283b",  # Or from config
            model_name=config['model_type'],
            max_tokens=max_new_tokens,
            temperature=0.7  # Adjust as needed
        )

    elif config['model_type'] == 'llama2':
        model, tokenizer = model_init.llama(config['model_path'], load_in_4bit=False)
        

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            device_map = 'auto'
        )

        return HuggingFacePipeline(pipeline=pipe)

    elif config['model_type'] == 'gpt4-turbo-128k':

        llm = model_init.gpt(config['env_file_path'], config['deployment_name'], config['model_version'], max_new_tokens, seed)

    elif config['model_type'] == 'llama-2-chat-70b' or config['model_type'] == 'llama-3-instruct-70b':

        model, tokenizer = model_init.llama(config['model_path'], load_in_4bit=True)
        print(model.hf_device_map)
        text_pipeline = pipeline(task="text-generation",
                                 model=model,
                                 tokenizer=tokenizer,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False)

        llm = HuggingFacePipeline(pipeline=text_pipeline)

    elif config['model_type'] == 'Mixtral-8x7B-Instruct-v0.1':

        model, tokenizer = model_init.mixtral(config['model_path'], load_in_4bit=True)

        text_pipeline = pipeline(task="text-generation",
                                 model=model,
                                 tokenizer=tokenizer,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False)

        llm = HuggingFacePipeline(pipeline=text_pipeline)

    else:
        raise ValueError('Model type {} not supported', config['model_type'])

    return llm

class DeepSeekLLM(BaseLLM):
    api_key: str
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        all_generations: List[List[Generation]] = []
        for prompt in prompts:

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            all_generations.append([Generation(text=response.choices[0].message.content)])
        
        return LLMResult(generations=all_generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._generate([prompt], stop, run_manager, **kwargs).generations[0][0].text
    
    async def _acall(self, *args, **kwargs):
        raise NotImplementedError("Async calls not implemented for DeepSeekLLM")
