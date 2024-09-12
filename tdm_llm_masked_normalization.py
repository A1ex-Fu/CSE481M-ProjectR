import json
import torch
import random
import argparse
import tqdm
import os

import utils
from prompting import NormalizationPrompt
from langchain.callbacks import get_openai_callback


def sample_tdms(gold_leaderboards, gold_data, seed, output_path, cold_start, sub_folder):

    # Extracting all gold individual TDM items
    labels_dict = {'Task': set(), 'Dataset': set(), 'Metric': set()}

    if not cold_start:
        leaderboard_tasks = {}
        for leaderboard in gold_leaderboards:
            if leaderboard['Task'] not in leaderboard_tasks:
                leaderboard_tasks[leaderboard['Task']] = 1
            else:
                leaderboard_tasks[leaderboard['Task']] += 1

        # Leaderboard bottleneck is task type. There are 8 tasks in 27 leaderboards for leaderboard threshold 3 and 16 tasks in 62 leaderboards for leaderboard threshold 2.
        # We sample the tasks among the ones that do not belong to the highest and the lowest number of leaderboards. Task sampling ratio is around 1/3.
        task_sampling_pool = [task for task in leaderboard_tasks if leaderboard_tasks[task] != max(leaderboard_tasks.values()) and leaderboard_tasks[task] != min(leaderboard_tasks.values())]

        random.seed(seed)
        masked_tasks = random.sample(task_sampling_pool, k=round(len(leaderboard_tasks)/3))
        # We are also masking the datasets that are in the same leaderboard with masked tasks
        masked_datasets = {leaderboard['Dataset'] for leaderboard in gold_leaderboards if leaderboard['Task'] in masked_tasks}

        masked_leaderboards = [leaderboard for leaderboard in gold_leaderboards if (leaderboard['Task'] in masked_tasks or leaderboard['Dataset'] in masked_datasets)]

        for paper in gold_data:
            for item in gold_data[paper]['TDMs']:
                if item['Task'] not in masked_tasks:
                    labels_dict['Task'].add(item['Task'])

                if item['Dataset'] not in masked_datasets:
                    labels_dict['Dataset'].add(item['Dataset'])

                labels_dict['Metric'].add(item['Metric'])

        os.makedirs(output_path + sub_folder, exist_ok=True)
        with open(output_path + sub_folder + 'partially_masked_leaderboards.json', "w") as fw:
            json.dump(masked_leaderboards, fw, indent=4)

    else:
        # In cold start setting all task, datasets and metrics are masked, so all leaderboards as well.
        os.makedirs(output_path + sub_folder, exist_ok=True)
        with open(output_path + sub_folder + 'cold_start_masked_leaderboards.json', "w") as fw:
            json.dump(gold_leaderboards, fw, indent=4)

    return labels_dict


def main(args):

    torch.manual_seed(args.seed)

    config = utils.read_json(args.tdm_config_file)
    gold_data = utils.read_json(args.gold_tdm_file)
    gold_leaderboards = utils.read_json(args.gold_leaderboard_file)
    model_output = utils.read_json(args.tdm_output_file)
    prompts = utils.read_json(args.prompt_file)

    if args.cold_start:
        if args.cold_start_seed is not None:
            sub_folder = "cold_start_" + str(args.cold_start_seed) + "/"
        else:
            sub_folder = "cold_start/"
    else:
        sub_folder = "partially_masked_normalization/"

    system_prompt = prompts['normalization-system-prompt']
    masking_system_prompt = prompts['masked-normalization-system-prompt']

    labels_dict = sample_tdms(gold_leaderboards, gold_data, args.seed, args.normalization_output_path, args.cold_start, sub_folder)

    normalized_output = {}
    costs = []

    llm = utils.llm_init_langchain(config, max_new_tokens=args.max_new_tokens, seed=args.seed)

    if args.cold_start and args.cold_start_seed is not None:
        random.seed(args.cold_start_seed)
        paper_list = random.sample(list(model_output.keys()), len(model_output))
    else:
        paper_list = list(model_output.keys())

    for i, paper in enumerate(tqdm.tqdm(paper_list, total=len(paper_list))):
        tdms = json.loads(model_output[paper]['output'])

        normalized_tdms = []
        for tdm in tdms:
            normalized_tdm = {}
            for key in tdm:
                if key != 'Result' and tdm[key] not in labels_dict[key]:
                    if not args.cold_start:
                        if key == 'Task' or key == 'Dataset':
                            prompt = NormalizationPrompt(masking_system_prompt, config['model_type'])
                        elif key == 'Metric':
                            prompt = NormalizationPrompt(system_prompt, config['model_type'])
                        else:
                            raise ValueError('Key error: {}', key)
                    else:
                        prompt = NormalizationPrompt(masking_system_prompt, config['model_type'])

                    if config['model_type'] == 'gpt4-turbo-128k':
                        with get_openai_callback() as cb:
                            answer = llm.invoke(prompt.prompt_template.invoke({'items': str(labels_dict[key]), 'input': tdm[key]})).content
                            costs.append({'prompt_tokens': cb.prompt_tokens,
                                          'completion_tokens': cb.completion_tokens,
                                          'total_cost': cb.total_cost})

                    elif config['model_type'] == 'llama-2-chat-70b' or config['model_type'] == 'llama-3-instruct-70b':
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': str(labels_dict[key]), 'input': tdm[key]}))[2:]

                    elif config['model_type'] == 'Mixtral-8x7B-Instruct-v0.1':
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': str(labels_dict[key]), 'input': tdm[key]}))[1:]
                        if (answer[0] == '"' and answer[-1] == '"') or (answer[0] == "'" and answer[-1] == "'"):
                            answer = answer[1:-1]
                    else:
                        raise ValueError('Model type {} not supported', config['model_type'])

                    if answer in labels_dict[key]:
                        normalized_tdm[key] = answer
                    else:
                        if not args.cold_start:
                            if (key == 'Task' or key == 'Dataset') and (answer == tdm[key]):
                                normalized_tdm[key] = answer
                                labels_dict[key].add(answer)
                            else:
                                normalized_tdm[key] = None
                        else:
                            if answer == tdm[key]:
                                normalized_tdm[key] = answer
                                labels_dict[key].add(answer)
                            else:
                                normalized_tdm[key] = None

                else:
                    normalized_tdm[key] = str(tdm[key])

            normalized_tdms.append(normalized_tdm)

        normalized_output[paper] = {'normalized_output': normalized_tdms, 'source_documents': model_output[paper]['source_documents']}

    if len(costs) > 0:
        costs.append({'experiment_cost': sum([cost['total_cost'] for cost in costs])})
        if args.cold_start:
            cost_output_file = 'cold_start_masked_normalization_costs.txt'
        else:
            cost_output_file = 'partially_masked_normalization_costs.txt'
        with open(args.normalization_output_path + sub_folder + cost_output_file, 'w') as fw:
            json.dump(costs, fw, indent=4)

    if args.cold_start:
        normalized_output_file = 'cold_start_masked_normalized_output.json'
    else:
        normalized_output_file = 'partially_masked_normalized_output.json'

    with open(args.normalization_output_path + sub_folder + normalized_output_file, "w") as fw:
        json.dump(normalized_output, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_tdm_file', required=True, type=str)
    parser.add_argument('--gold_leaderboard_file', required=True, type=str)
    parser.add_argument('--tdm_config_file', required=True, type=str)
    parser.add_argument('--tdm_output_file', required=True, type=str)
    parser.add_argument('--normalization_output_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--cold_start', action='store_true')
    parser.add_argument('--cold_start_seed', default=None) # 3 version (No change, 0, 42)
    parser.add_argument('--max_new_tokens', default=15, type=int)
    parser.add_argument('--seed', default=0, type=int)

    main(parser.parse_args())
