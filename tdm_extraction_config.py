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
    # config.few_shot_system_prompt = prompts['few-shot-extraction-system-prompt']
    config.few_shot_system_prompt ="""
    You are a highly accurate information extraction system. You are given snippets of text from research papers. Your task is to identify and extract tuples containing the name of the task addressed in the paper, utilized datasets, evaluation metrics, and corresponding results. Extract these tuples for only the best results obtained by proposed methods of the paper, not baselines. Please use JSON format for each different tuple.

Example format: [{"Task": "Task name", "Dataset": "Dataset name", "Metric": "Metric name", "Result": "Result score"}].

Your answer will immediately start with the JSON object satisfying the given template and contain nothing else.

Here are 5 examples to guide your extraction:

**Example 1**

**Input:**

"Table 1: Results on the WNUT 2016 dataset.\nModel  F1\nSystem A 52.41\nOurs 53.48"

**Output:**

```json
[{"Task": "Named Entity Recognition", "Dataset": "WNUT 2016", "Metric": "F1", "Result": "53.48"}]
Example 2

Input:

"Table 4: POS tagging accuracy of our model on test data from WSJ proportion of PTB, together with top-performance systems. The neural net- work based models are marked with \u2021.\nModel Acc.\nToutanova et al. (2003) 97.27\nThis paper 97.55"

Output:

JSON

[{"Task": "POS Tagging", "Dataset": "WSJ", "Metric": "Accuracy", "Result": "97.55"}]
Example 3

Input:

"Table 1: Results on WMT14 En\u2192Fr.\nThe num- bers before and after \u2019\u00b1\u2019 are the mean and stan- dard deviation of test BLEU score over an evalua- tion window.\nModel Test BLEU\n(Zhou et al. 2019) 53.43\nOurs 53.48"

Output:

JSON

[{"Task": "Machine Translation", "Dataset": "WMT14 En\u2192Fr", "Metric": "BLEU", "Result": "53.48"}]
Example 4

Input:

"Table 5: Results of our models, with various feature sets, compared to other published results.\nModel CoNLL-2003 Prec. Recall F1\nBLSTM-CNN + emb + lex 91.39 91.85 91.62"

Output:

JSON

[{"Task": "Named Entity Recognition", "Dataset": "CoNLL-2003", "Metric": "F1", "Result": "91.62"}]
Example 5

Input:

"Table 4: F1 score of models trained to predict document-at-a-time.\nModel F1\nBi-LSTM-CRF (sent) 90.43 \u00b1 0.12 \nID-CNN 90.65 \u00b1 0.15"

Output:

JSON

[{"Task": "Named Entity Recognition", "Dataset": "CoNLL", "Metric": "F1", "Result": "90.65"}]
Now, complete the extraction on the following input.

Input:

"Table 4: POS tagging accuracy of our model on test data from WSJ proportion of PTB, together with top-performance systems. The neural net- work based models are marked with \u2021.\nModel Acc.\nCollobert et al. (2011)\u2021 97.29\nSantos and Zadrozny (2014)\u2021 97.32\nThis paper 97.55"
    """
    config.query_prompt = prompts['query']

    # The maximum number of tokens to generate
    config.max_new_tokens = args.max_new_tokens

    # Seed
    config.seed = args.seed

    return config
