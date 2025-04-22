from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI


def llama(model_path, load_in_4bit):

    # logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=load_in_4bit)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def mixtral(model_path, load_in_4bit):

    # logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=load_in_4bit)

    return model, tokenizer


def gpt(env_file_path, deployment_name, model_version, max_new_tokens, seed):

    load_dotenv(env_file_path)

    model = AzureChatOpenAI(deployment_name=deployment_name, openai_api_version='2023-12-01-preview', model_version=model_version, max_tokens=max_new_tokens, model_kwargs={"seed": seed})

    return model


