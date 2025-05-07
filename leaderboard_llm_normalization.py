import json
import torch
import random
import argparse
import tqdm
import utils
from prompting import NormalizationPrompt
from langchain.callbacks import get_openai_callback


def main(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = utils.read_json(args.tdm_config_file)
    gold_leaderboards = utils.read_json(args.gold_leaderboard_file)
    masked_normalized_output = utils.read_json(args.masked_normalized_file)
    prompts = utils.read_json(args.prompt_file)

    system_prompt = prompts['leaderboard-normalization-system-prompt']

    leaderboard_tdm_set = {'(' + gold_leaderboard['Task'] + ', ' + gold_leaderboard['Dataset'] + ', ' + gold_leaderboard['Metric'] + ')' for gold_leaderboard in gold_leaderboards}

    prompt = NormalizationPrompt(system_prompt, config['model_type'])

    leaderboard_normalized_output = {}
    costs = []

    llm = utils.llm_init_langchain(config, max_new_tokens=args.max_new_tokens, seed=args.seed)

    for i, paper in enumerate(tqdm.tqdm(masked_normalized_output, total=len(masked_normalized_output))):
        tdms = masked_normalized_output[paper]['normalized_output']

        normalized_tdms = []
        for tdm in tdms:
            normalized_tdm = tdm
            if (tdm['Task'] is not None) and (tdm['Dataset'] is not None) and (tdm['Metric'] is not None):

                model_tuple = '(' + tdm['Task'] + ', ' + tdm['Dataset'] + ', ' + tdm['Metric'] + ')'

                if model_tuple not in leaderboard_tdm_set:
                    if config['model_type'] == 'gpt4-turbo-128k':
                        with get_openai_callback() as cb:
                            answer = llm.invoke(prompt.prompt_template.invoke({'items': '{' + ", ".join(leaderboard_tdm_set) + '}', 'input': model_tuple})).content
                            costs.append({'prompt_tokens': cb.prompt_tokens,
                                          'completion_tokens': cb.completion_tokens,
                                          'total_cost': cb.total_cost})

                    elif config['model_type'] == 'llama-2' or config['model_type'] == 'llama-2-chat-70b' or config['model_type'] == 'llama-3-instruct-70b':
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': '{' + ", ".join(leaderboard_tdm_set) + '}', 'input': model_tuple}))[2:]

                    elif config['model_type'] == 'Mixtral-8x7B-Instruct-v0.1':
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': '{' + ", ".join(leaderboard_tdm_set) + '}', 'input': model_tuple}))[1:]
                        if (answer[0] == '"' and answer[-1] == '"') or (answer[0] == "'" and answer[-1] == "'"):
                            answer = answer[1:-1]
                    elif config['model_type'] == 'deepseek':
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': '{' + ", ".join(leaderboard_tdm_set) + '}', 'input': model_tuple}))
                    else:
                        raise ValueError('Model type {} not supported', config['model_type'])

                    if answer in leaderboard_tdm_set:
                        tuple = answer[1:-1].split(', ')
                        normalized_tdm['Task'] = tuple[0]
                        normalized_tdm['Dataset'] = tuple[1]
                        normalized_tdm['Metric'] = tuple[2]

            normalized_tdms.append(normalized_tdm)

        leaderboard_normalized_output[paper] = {'normalized_output': normalized_tdms, 'source_documents': masked_normalized_output[paper]['source_documents']}

    if len(costs) > 0:
        costs.append({'experiment_cost': sum([cost['total_cost'] for cost in costs])})
        with open(args.leaderboard_normalization_path + 'leaderboard_normalization_costs.txt', 'w') as fw:
            json.dump(costs, fw, indent=4)

    with open(args.leaderboard_normalization_path + 'leaderboard_normalized_output.json', "w") as fw:
        json.dump(leaderboard_normalized_output, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_leaderboard_file', required=True, type=str)
    parser.add_argument('--tdm_config_file', required=True, type=str)
    parser.add_argument('--masked_normalized_file', required=True, type=str)
    parser.add_argument('--leaderboard_normalization_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--max_new_tokens', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)

    main(parser.parse_args())
