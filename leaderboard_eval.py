import argparse
import json
import rbo
import utils


def construct_output_leaderboards(gold_leaderboards, normalized_output):

    output_leaderboards = {str((gold_leaderboard['Task'], gold_leaderboard['Dataset'], gold_leaderboard['Metric'])): {} for gold_leaderboard in gold_leaderboards}
    for gold_leaderboard in gold_leaderboards:
        for paper in gold_leaderboard['Leaderboard']:
            if gold_leaderboard['Leaderboard'][paper] < 1:
                gold_leaderboard['Leaderboard'][paper] *= 100

    for paper in normalized_output:
        for tdm in normalized_output[paper]['normalized_output']:
            tdm_tuple = str((tdm['Task'], tdm['Dataset'], tdm['Metric']))
            if tdm_tuple in output_leaderboards:
                result = tdm['Result']
                if '±' in result:
                    result = result[:result.index('±')]
                if '%' in result:
                    result = result[:result.index('%')]

                try:
                    result = float(result)
                except:
                    continue

                if tdm['Metric'] in ['AVG', 'Accuracy', 'BERTScore', 'BLEU', 'BLEU-4', 'Exact Match (EM)', 'F-Score (F-S)', 'F1', 'Fuzzy B-Cubed (FBC)', 'Fuzzy normalized mutual information (FNMI)', 'Labeled Attachment Score', 'METEOR', "Matthew's Correlation Coefficient (MCC)", 'NIST-4', 'Overall-Accuracy', 'Precision', 'ROGUE-1', 'ROGUE-2', 'ROGUE-L', 'Recall', 'Sent-Accuracy', 'Spearman Correlation', 'TER', 'Unlabeled Attachment Score', 'V-Measure (V-M)']:
                    if result < 1:
                        result *= 100

                output_leaderboards[tdm_tuple][paper] = result

    return output_leaderboards


def evaluate(output_leaderboards, gold_leaderboards):

    eval_by_tuples = {'Paper Coverage': {}, 'Result Coverage': {}, 'RBO': {}}
    for gold_leaderboard in gold_leaderboards:
        tdm_key = str((gold_leaderboard['Task'], gold_leaderboard['Dataset'], gold_leaderboard['Metric']))
        gold_papers = set(gold_leaderboard['Leaderboard'].keys())
        output_papers = set(output_leaderboards[str((gold_leaderboard['Task'], gold_leaderboard['Dataset'], gold_leaderboard['Metric']))].keys())

        eval_by_tuples['Paper Coverage'][tdm_key] = len(gold_papers.intersection(output_papers))/len(gold_papers)
        eval_by_tuples['Result Coverage'][tdm_key] = len([i for i in output_leaderboards[tdm_key].values() if i in gold_leaderboard['Leaderboard'].values()]) / len(gold_leaderboard['Leaderboard'].values())
        if gold_leaderboard['Metric'] in ['Perplexity', 'TER']:
            eval_by_tuples['RBO'][tdm_key] = rbo.RankingSimilarity(sorted(list(set(gold_leaderboard['Leaderboard'].values()))), sorted(list(set(output_leaderboards[tdm_key].values())))).rbo()
        else:
            eval_by_tuples['RBO'][tdm_key] = rbo.RankingSimilarity(sorted(list(set(gold_leaderboard['Leaderboard'].values())), reverse=True), sorted(list(set(output_leaderboards[tdm_key].values())), reverse=True)).rbo()
    return eval_by_tuples


def save_results(eval_scores, eval_results_path, eval_values_path):

    avg_results = {}
    for leaderboard_type in eval_scores:
        avg_results[leaderboard_type] = {}
        for key in eval_scores[leaderboard_type]:
            avg_results[leaderboard_type][key] = sum(eval_scores[leaderboard_type][key].values()) / len(eval_scores[leaderboard_type][key])

    with open(eval_values_path, 'w') as fw:
        json.dump(eval_scores, fw, indent=4)

    with open(eval_results_path, 'w') as fw:
        json.dump(avg_results, fw, indent=4)


def main(args):

    eval_scores = {}

    gold_leaderboards = utils.read_json(args.gold_leaderboards_file)
    normalized_output = utils.read_json(args.normalized_tdm_output_path)

    output_leaderboards = construct_output_leaderboards(gold_leaderboards, normalized_output)
    print(output_leaderboards)
    print()
    print(gold_leaderboards)
    # eval_scores['gold_results'] = evaluate(output_leaderboards, gold_leaderboards)

    # if args.masked_leaderboards_file != "":
    #     masked_leaderboards = utils.read_json(args.masked_leaderboards_file)

    #     masked_output_leaderboards = construct_output_leaderboards(masked_leaderboards, normalized_output)
    #     eval_scores['masked_results'] = evaluate(masked_output_leaderboards, masked_leaderboards)

    # save_results(eval_scores, args.eval_results_file, args.eval_values_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_leaderboards_file', required=True, type=str)
    parser.add_argument('--masked_leaderboards_file', default='', type=str)
    parser.add_argument('--normalized_tdm_output_path', required=True, type=str)
    parser.add_argument('--eval_results_file', required=True, type=str)
    parser.add_argument('--eval_values_file', required=True, type=str)

    main(parser.parse_args())
