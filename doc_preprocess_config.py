import ml_collections
from datetime import datetime
import json


def get_config(args):
    """
        Configuration elements for related work paragraph generations
    """

    config = ml_collections.ConfigDict()

    # Experiment id
    config.process_id = args.process_id

    # Papers file
    config.papers_path = args.papers_path

    # # Output path
    # config.output_path = args.output_path + config.process_id + '-' + datetime.now().strftime('%d.%m.%Y-%H:%M:%S') + '/'

    # # Prompt file
    # # with open(args.prompt_file) as fr:
    # #     prompts = json.load(fr)
    # with open(args.prompt_file, 'r', encoding='utf-8') as fr:
    #     prompts = json.load(fr)

    # Output path with updated timestamp format (replacing colon with underscore)
    config.output_path = args.output_path + config.process_id + '-' + datetime.now().strftime('%d.%m.%Y-%H_%M_%S') + '/'

    # Prompt file
    with open(args.prompt_file, 'r', encoding='utf-8') as fr:
        prompts = json.load(fr)


    config.query = prompts['query']

    return config
