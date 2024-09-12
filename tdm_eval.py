import argparse
import utils
import json


def calculate_exact_match(gold_data, normalized_output):

    recall_scores = {}
    precision_scores = {}
    for paper in normalized_output:
        exact_matches = 0
        # unique_output_tdms = [dict(t) for t in {tuple(d.items()) for d in output_tdms[paper]}]
        # for output_tdm in unique_output_tdms:
        for output_tdm in normalized_output[paper]['normalized_output']:
            found = False
            for gold_tdm in gold_data[paper]['TDMs']:
                if (output_tdm['Task'] == gold_tdm['Task']) and (output_tdm['Dataset'] == gold_tdm['Dataset']) and (output_tdm['Metric'] == gold_tdm['Metric']) and ((str(gold_tdm['Result']) in output_tdm['Result']) or (output_tdm['Result']) in str(gold_tdm['Result'])):
                    exact_matches += 1
                    found = True
                if found:
                    break

        recall_scores[paper] = exact_matches / len(gold_data[paper]['TDMs'])
        if len(normalized_output[paper]['normalized_output']) != 0:
            precision_scores[paper] = exact_matches / len(normalized_output[paper]['normalized_output'])
        else:
            precision_scores[paper] = 0

    return recall_scores, precision_scores


def calculate_exact_match_wo_results(gold_data, normalized_output):

    recall_scores = {}
    precision_scores = {}
    for paper in normalized_output:
        exact_matches = 0
        output_set = set()
        gold_set = set()
        for tdm in normalized_output[paper]['normalized_output']:
            output_set.add(tuple({'Task': tdm['Task'], 'Dataset': tdm['Dataset'], 'Metric': tdm['Metric']}.items()))
        unique_output_tdms = [dict(t) for t in output_set]

        for tdm in gold_data[paper]['TDMs']:
            gold_set.add(tuple({'Task': tdm['Task'], 'Dataset': tdm['Dataset'], 'Metric': tdm['Metric']}.items()))
        unique_gold_tdms = [dict(t) for t in gold_set]

        for output_tdm in unique_output_tdms:
            found = False
            for gold_tdm in unique_gold_tdms:
                if (output_tdm['Task'] == gold_tdm['Task']) and (output_tdm['Dataset'] == gold_tdm['Dataset']) and (output_tdm['Metric'] == gold_tdm['Metric']):
                    exact_matches += 1
                    found = True
                if found:
                    break

        recall_scores[paper] = exact_matches / len(unique_gold_tdms)
        if len(normalized_output[paper]['normalized_output']) != 0:
            precision_scores[paper] = exact_matches / len(unique_output_tdms)
        else:
            precision_scores[paper] = 0

    return recall_scores, precision_scores


def calculate_ind_match(gold_data, normalized_output, item):

    recall_scores = {}
    precision_scores = {}
    for paper in normalized_output:
        if item in ['Task', 'Dataset', 'Metric']:
            unique_output_items = {output_tdm[item] for output_tdm in normalized_output[paper]['normalized_output']}
            unique_gold_items = {gold_tdm[item] for gold_tdm in gold_data[paper]['TDMs']}

            recall_scores[paper] = len(unique_output_items.intersection(unique_gold_items)) / len(unique_gold_items)
            if len(unique_output_items) != 0:
                precision_scores[paper] = len(unique_output_items.intersection(unique_gold_items)) / len(unique_output_items)
            else:
                precision_scores[paper] = 0

        elif item == 'Result':
            matches = 0
            unique_output_items = {output_tdm[item] for output_tdm in normalized_output[paper]['normalized_output']}
            unique_gold_items = {gold_tdm[item] for gold_tdm in gold_data[paper]['TDMs']}
            for unique_output_item in unique_output_items:
                found = False
                for unique_gold_item in unique_gold_items:
                    if (str(unique_gold_item) in unique_output_item) or (unique_output_item in str(unique_gold_item)):
                        matches += 1
                        found = True
                    if found:
                        break

            recall_scores[paper] = matches / len(unique_gold_items)
            if len(unique_output_items) != 0:
                precision_scores[paper] = matches / len(unique_output_items)
            else:
                precision_scores[paper] = 0

        else:
            raise ValueError('Unknown item: {}'.format(item))

    return recall_scores, precision_scores


def save_results(eval_results_path, eval_values_path, recall_scores, precision_scores, recall_scores_wo_result, precision_scores_wo_results, ind_scores):

    eval_values = {'Exact Match Recall': recall_scores,
                   'Exact Match Precision': precision_scores,
                   'Exact Match Recall wo Results': recall_scores_wo_result,
                   'Exact Match Recall wo Precision': precision_scores_wo_results,
                   'Individual': ind_scores}

    eval_results = {'Exact Match Recall': sum(recall_scores.values()) / len(recall_scores),
                    'Exact Match Precision': sum(precision_scores.values()) / len(precision_scores),
                    'Exact Match Recall wo Results': sum(recall_scores_wo_result.values()) / len(recall_scores_wo_result),
                    'Exact Match Precision wo Results': sum(precision_scores_wo_results.values()) / len(precision_scores_wo_results),
                    'Individual': {key: {metric: sum(ind_scores[key][metric].values()) / len(ind_scores[key][metric]) for metric in ind_scores[key]} for key in ind_scores}}

    with open(eval_values_path, 'w') as fw:
        json.dump(eval_values, fw, indent=4)

    with open(eval_results_path, 'w') as fw:
        json.dump(eval_results, fw, indent=4)


def main(args):

    with open(args.gold_data_path, 'r') as fr:
        gold_data = json.load(fr)

    with open(args.normalized_tdm_output_path, 'r') as fr:
        normalized_output = json.load(fr)

    recall_scores, precision_scores = calculate_exact_match(gold_data, normalized_output)

    recall_scores_wo_result, precision_scores_wo_results = calculate_exact_match_wo_results(gold_data, normalized_output)

    ind_scores = {}

    for item in ['Task', 'Dataset', 'Metric', 'Result']:
        ind_recall_scores, ind_precision_scores = calculate_ind_match(gold_data, normalized_output, item)
        ind_scores[item] = {'Recall': ind_recall_scores, 'Precision': ind_precision_scores}

    save_results(args.eval_results_path,
                 args.eval_values_path,
                 recall_scores, precision_scores,
                 recall_scores_wo_result,
                 precision_scores_wo_results,
                 ind_scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_data_path', required=True, type=str)
    parser.add_argument('--normalized_tdm_output_path', required=True, type=str)
    parser.add_argument('--eval_results_path', required=True, type=str)
    parser.add_argument('--eval_values_path', required=True, type=str)

    main(parser.parse_args())
