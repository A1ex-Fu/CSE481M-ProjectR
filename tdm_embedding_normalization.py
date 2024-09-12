import json
import torch
import random
import argparse
import tqdm
import os
import utils
from sentence_transformers import SentenceTransformer, util


def main(args):

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    gold_data = utils.read_json(args.gold_tdm_path)
    model_output = utils.read_json(args.tdm_output_path + 'manually_refined_output.json')

    labels_dict = {'Task': set(), 'Dataset': set(), 'Metric': set()}

    for paper in gold_data:
        for item in gold_data[paper]['TDMs']:
            labels_dict['Task'].add(item['Task'])
            labels_dict['Dataset'].add(item['Dataset'])
            labels_dict['Metric'].add(item['Metric'])

    labels_dict['Task'] = list(labels_dict['Task'])
    labels_dict['Dataset'] = list(labels_dict['Dataset'])
    labels_dict['Metric'] = list(labels_dict['Metric'])

    encoder = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)

    corpus_embeddings = {'Task': encoder.encode(labels_dict['Task'], convert_to_tensor=True),
                         'Dataset': encoder.encode(labels_dict['Dataset'], convert_to_tensor=True),
                         'Metric': encoder.encode(labels_dict['Metric'], convert_to_tensor=True)}


    normalized_output = {}

    for i, paper in enumerate(tqdm.tqdm(model_output, total=len(model_output))):
        tdms = json.loads(model_output[paper]['output'])

        normalized_tdms = []
        for tdm in tdms:
            normalized_tdm = {}
            for key in tdm:
                if key != 'Result' and tdm[key] not in labels_dict[key]:

                    candidate_embedding = encoder.encode(tdm[key], convert_to_tensor=True)
                    normalized_tdm[key] = labels_dict[key][util.semantic_search(candidate_embedding, corpus_embeddings[key], top_k=1)[0][0]['corpus_id']]
                else:
                    normalized_tdm[key] = str(tdm[key])

            normalized_tdms.append(normalized_tdm)

        normalized_output[paper] = {'normalized_output': normalized_tdms, 'source_documents': model_output[paper]['source_documents']}

    os.makedirs(args.tdm_output_path + 'embedding_normalization/', exist_ok=True)

    with open(args.tdm_output_path + 'embedding_normalization/embedding_normalized_output.json', "w") as fw:
        json.dump(normalized_output, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_tdm_path', required=True, type=str)
    parser.add_argument('--tdm_output_path', required=True, type=str)
    parser.add_argument('--seed', default=0, type=int)

    main(parser.parse_args())
