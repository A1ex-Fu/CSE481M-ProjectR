import json
import torch
import random
import argparse
import tqdm
import os
import utils
from prompting import NormalizationPrompt
from langchain.callbacks import get_openai_callback


def main(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = utils.read_json(args.tdm_output_path + 'config.json')
    gold_data = utils.read_json(args.gold_tdm_path)
    model_output = utils.read_json(args.tdm_output_path + 'output.json')
    prompts = utils.read_json(args.prompt_file)

    if config['is_few_shot']:
        system_prompt = prompts['few-shot-normalization-system-prompt']
        print('Few shot experiment...')
    else:
        system_prompt = prompts['normalization-system-prompt']
        print('Zero shot experiment...')

    labels_dict = {'Task': set(), 'Dataset': set(), 'Metric': set()}

    for paper in gold_data:
        for item in gold_data[paper]['TDMs']:
            labels_dict['Task'].add(item['Task'])
            labels_dict['Dataset'].add(item['Dataset'])
            labels_dict['Metric'].add(item['Metric'])

    prompt = NormalizationPrompt(system_prompt, config['model_type'])

    normalized_output = {}
    costs = []

    llm = utils.llm_init_langchain(config, max_new_tokens=args.max_new_tokens, seed=args.seed)

    for i, paper in enumerate(tqdm.tqdm(model_output, total=len(model_output))):
        tdms = json.loads(model_output[paper]['output'])

        normalized_tdms = []
        for tdm in tdms:
            normalized_tdm = {}
            for key in tdm:
                if key != 'Result' and tdm[key] not in labels_dict[key]:
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
                    elif config['model_type'] == 'deepseek':
                        # print(prompt.prompt_template.invoke({'items': str(labels_dict[key]), 'input': tdm[key]}))
                        # print()
                        answer = llm.invoke(prompt.prompt_template.invoke({'items': str(labels_dict[key]), 'input': tdm[key]}))
                    else:
                        raise ValueError('Model type {} not supported', config['model_type'])

                    # Drop EOS token
                    answer = answer[:-4] if answer.endswith("</s>") else answer

                    # if answer in labels_dict[key]:
                    normalized_tdm[key] = answer
                    # else:
                    #     print("UH OH BAD")
                    #     print(answer)
                    #     print(labels_dict[key])
                    #     print()
                    #     normalized_tdm[key] = None
                else:
                    normalized_tdm[key] = str(tdm[key])

            normalized_tdms.append(normalized_tdm)

        normalized_output[paper] = {'normalized_output': normalized_tdms, 'source_documents': model_output[paper]['source_documents']}
        # test to see whether find closest match is possible
        # print(normalized_output[paper])

    os.makedirs(args.tdm_output_path + 'normalization/', exist_ok=True)

    if len(costs) > 0:
        costs.append({'experiment_cost': sum([cost['total_cost'] for cost in costs])})
        with open(args.tdm_output_path + 'normalization/normalization_costs.txt', 'w') as fw:
            json.dump(costs, fw, indent=4)

    with open(args.tdm_output_path + 'normalization/normalized_output.json', "w") as fw:
        json.dump(normalized_output, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_tdm_path', required=True, type=str)
    parser.add_argument('--tdm_output_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--max_new_tokens', default=200, type=int)
    parser.add_argument('--seed', default=0, type=int)

    main(parser.parse_args())
