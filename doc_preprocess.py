# from pdf_loader import Documents
# import argparse
# import os
# import tqdm
# import pickle
# import doc_preprocess_config
# import json


# def main(config):

#     if not os.path.exists(config.output_path):
#         os.makedirs(config.output_path)
#     with open(config.output_path + 'config.json', 'w') as fw:
#         json.dump(config.to_dict(), fw, indent=2)

#     paper_list = os.listdir(config.papers_path)

#     retrieved_docs = {}

#     for paper in tqdm.tqdm(paper_list, total=len(paper_list)):

#         documents = Documents(pdf_directory=os.path.join(config.papers_path, paper))

#         retriever = documents.init_retriever()

#         retrieved_docs[paper] = retriever.invoke(config.query)

#         documents.db.delete_collection()

#     with open(config.output_path + 'processed_docs.pkl', 'wb') as fw:
#         pickle.dump(retrieved_docs, fw)


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--process_id', required=True, type=str)
#     parser.add_argument('--papers_path', required=True, type=str)
#     parser.add_argument('--prompt_file', default='prompts.json', type=str)
#     parser.add_argument('--output_path', required=True, type=str)

#     main(doc_preprocess_config.get_config(parser.parse_args()))

from pdf_loader import Documents
import argparse
import os
import tqdm
import pickle
import doc_preprocess_config
import json


def main(config):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    with open(config.output_path + 'config.json', 'w') as fw:
        json.dump(config.to_dict(), fw, indent=2)

    paper_list = os.listdir(config.papers_path)

    retrieved_docs = {}

    for paper in tqdm.tqdm(paper_list, total=len(paper_list)):

        documents = Documents(pdf_directory=os.path.join(config.papers_path, paper))

        retriever = documents.init_retriever()

        retrieved_docs[paper] = retriever.invoke(config.query)

        documents.db.delete_collection()

    with open(config.output_path + 'processed_docs.pkl', 'wb') as fw:
        pickle.dump(retrieved_docs, fw)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--process_id', required=True, type=str)
    parser.add_argument('--papers_path', required=True, type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--output_path', required=True, type=str)

    main(doc_preprocess_config.get_config(parser.parse_args()))
