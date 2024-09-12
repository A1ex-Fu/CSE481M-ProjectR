from pdf_loader import Documents, PreProcessedDocuments
from prompting import TDMPrompt
from langchain.chains import RetrievalQA
import tdm_extraction_config
import argparse
import torch
import random
import os
import json
import tqdm
import pickle
from langchain.callbacks import get_openai_callback
import utils


def main(config):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    with open(config.output_path + 'config.json', 'w') as fw:
        json.dump(config.to_dict(), fw, indent=2)

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    llm = utils.llm_init_langchain(config, max_new_tokens=config.max_new_tokens, seed=config.seed)

    if config.is_few_shot:
        prompt = TDMPrompt(config.few_shot_system_prompt, config.model_type)
        print('Few shot experiment...')
    else:
        prompt = TDMPrompt(config.system_prompt, config.model_type)
        print('Zero shot experiment...')

    outputs = {}
    costs = []
    if config.is_preprocessed_doc:
        print('Using preprocessed documents...')
        with open(config.processed_docs_path, 'rb') as fr:
            docs_dict = pickle.load(fr)

        for i, paper in enumerate(tqdm.tqdm(docs_dict, total=len(docs_dict))):

            preprocessed_docs = PreProcessedDocuments(docs=docs_dict[paper])
            retriever = preprocessed_docs.init_retriever()

            chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=retriever,
                                                return_source_documents=True,
                                                chain_type_kwargs={"prompt": prompt.prompt_template})

            if config.model_type == 'gpt4-turbo-128k':
                with get_openai_callback() as cb:
                    result = chain.invoke(config.query_prompt)
                    costs.append({'prompt_tokens': cb.prompt_tokens,
                                  'completion_tokens': cb.completion_tokens,
                                  'total_cost': cb.total_cost})

            else:
                result = chain.invoke(config.query_prompt)

            outputs[paper] = {'output': result['result'],
                              'source_documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in result['source_documents']]}

    else:
        print('Preprocessing documents first before extraction...')
        paper_list = os.listdir(config.papers_path)

        for paper in tqdm.tqdm(paper_list, total=len(paper_list)):
            documents = Documents(pdf_directory=os.path.join(config.papers_path, paper))
            retriever = documents.init_retriever()

            chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=retriever,
                                                return_source_documents=True,
                                                chain_type_kwargs={"prompt": prompt.prompt_template})

            if config.model_type == 'gpt4-turbo-128k':
                with get_openai_callback() as cb:
                    result = chain.invoke(config.query_prompt)
                    costs.append({'prompt_tokens': cb.prompt_tokens,
                                  'completion_tokens': cb.completion_tokens,
                                  'total_cost': cb.total_cost})

            else:
                result = chain.invoke(config.query_prompt)

            outputs[paper] = {'output': result['result'],
                              'source_documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in result['source_documents']]}

            documents.db.delete_collection()

    if len(costs) > 0:
        costs.append({'experiment_cost': sum([cost['total_cost'] for cost in costs])})
        with open(config.output_path + 'costs.txt', 'w') as fw:
            json.dump(costs, fw, indent=4)

    with open(config.output_path + 'output.json', "w") as fw:
        json.dump(outputs, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_file_path', required=True, type=str)
    parser.add_argument('--exp_id', required=True, type=str)
    parser.add_argument('--processed_docs_path', required=True, type=str)
    parser.add_argument('--papers_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)
    parser.add_argument('--model_version', required=True, type=str)
    parser.add_argument('--deployment_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--is_preprocessed_doc', action='store_true')
    parser.add_argument('--is_few_shot', action='store_true')
    parser.add_argument('--max_new_tokens', default=1024, type=int)
    parser.add_argument('--seed', default=0, type=int)

    main(tdm_extraction_config.get_config(parser.parse_args()))
