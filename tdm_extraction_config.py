import ml_collections
from datetime import datetime
import json


def get_config(args):
    """
        Configuration elements for related work paragraph generations
    """

    config = ml_collections.ConfigDict()

    # Environmental variables path for keys
    config.env_file_path = args.env_file_path

    # Experiment id
    config.exp_id = args.exp_id

    # Processed documents path
    config.processed_docs_path = args.processed_docs_path

    # Whether processed documents are used
    config.is_preprocessed_doc = args.is_preprocessed_doc

    # Whether few-shot experiment
    config.is_few_shot = args.is_few_shot

    # Papers file
    config.papers_path = args.papers_path

    # Model type
    config.model_type = args.model_type

    # Model version (if GPT type)
    config.model_version = args.model_version

    # Deployment name (if GPT type)
    config.deployment_name = args.deployment_name

    # Model path
    config.model_path = args.model_path

    # Output path
    config.output_path = args.output_path + config.exp_id + '-' + config.model_type + '-' + datetime.now().strftime('%d.%m.%Y-%H_%M_%S') + '/'

    # Prompt file
    # with open(args.prompt_file) as fr:
    #     prompts = json.load(fr)
    with open(args.prompt_file, "r", encoding="utf-8") as fr:
        prompts = json.load(fr)


    config.system_prompt = prompts['tdm-extraction-system-prompt']
    config.few_shot_system_prompt = prompts['few-shot-extraction-system-prompt']
    config.query_prompt = prompts['query']

    # The maximum number of tokens to generate
    config.max_new_tokens = args.max_new_tokens

    # Seed
    config.seed = args.seed

    return config
