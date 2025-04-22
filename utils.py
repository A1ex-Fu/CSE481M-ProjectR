import model_init
from transformers import pipeline
from langchain import HuggingFacePipeline
import json


def read_json(path):

    with open(path, 'r') as fr:
        file = json.load(fr)
    return file


def llm_init_langchain(config, max_new_tokens, seed):

    if config['model_type'] == 'llama2':
        model, tokenizer = model_init.llama(config['model_path'], load_in_4bit=False)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

        return HuggingFacePipeline(pipeline=pipe)

    elif config['model_type'] == 'gpt4-turbo-128k':

        llm = model_init.gpt(config['env_file_path'], config['deployment_name'], config['model_version'], max_new_tokens, seed)

    elif config['model_type'] == 'llama-2-chat-70b' or config['model_type'] == 'llama-3-instruct-70b':

        model, tokenizer = model_init.llama(config['model_path'], load_in_4bit=True)

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
