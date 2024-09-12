from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from typing import List


class PreProcessedDocuments:

    def __init__(self, docs: List[Document]):
        self.docs = docs

    def init_retriever(self):
        return CustomDummyRetriever(external_docs=self.docs)


class Documents:

    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.tables = None
        self.split_docs = None
        self.db = None
    
    def init_retriever(self, chunk_size=2000, chunk_overlap=200, model_name='multi-qa-mpnet-base-dot-v1'):
        # Max seq. length ~500 tokens => ~2000 chars (0.45 page)
        self.tables, paper_text = self.extract_text_and_tables()
        self.split_docs = RecursiveCharacterTextSplitter(separators=[" ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True).split_documents(paper_text)
        self.db = Chroma.from_documents(documents=self.split_docs, embedding=HuggingFaceEmbeddings(model_name=model_name))

        return CustomRetriever(vector_retriever=self.db.as_retriever(search_type="similarity"), external_docs=self.tables)

    def extract_text_and_tables(self):

        elements = partition_pdf(filename=self.pdf_directory, strategy='hi_res', infer_table_structure=True)
        text_by_page = {}
        tables = []
        captions = []
        for i, el in enumerate(elements):
            if el.category == "Table":
                try:
                    if (elements[i-1].text not in captions) and ((elements[i-1].category == 'FigureCaption') or (elements[i-1].text[:5] == 'Table')):
                        tables.append(Document(page_content=elements[i-1].text + '\n' + el.text,
                                               metadata={'filename': el.metadata.filename,
                                                         'page_number': el.metadata.page_number,
                                                         'type': 'Table'}))

                        captions.append(elements[i-1].text)

                    elif (elements[i+1].text not in captions) and ((elements[i+1].category == 'FigureCaption') or (elements[i+1].text[:5] == 'Table')):
                        tables.append(Document(page_content=elements[i+1].text + '\n' + el.text,
                                               metadata={'filename': el.metadata.filename,
                                                         'page_number': el.metadata.page_number,
                                                         'type': 'Table'}))

                        captions.append(elements[i+1].text)

                    else:
                        tables.append(Document(page_content=el.text,
                                               metadata={'filename': el.metadata.filename,
                                                         'page_number': el.metadata.page_number,
                                                         'type': 'Table'}))
                except IndexError:
                    tables.append(Document(page_content=el.text,
                                           metadata={'filename': el.metadata.filename,
                                                     'page_number': el.metadata.page_number,
                                                     'type': 'Table'}))

            elif (el.category == 'NarrativeText') or (el.category == 'Title'):
                if el.metadata.page_number not in text_by_page:
                    text_by_page[el.metadata.page_number] = Document(page_content=el.text,
                                                                     metadata={'filename': el.metadata.filename,
                                                                               'page_number': el.metadata.page_number,
                                                                               'type': 'Text'})
                else:
                    text_by_page[el.metadata.page_number].page_content += (' ' + el.text)

        return tables, list(text_by_page.values())


class CustomRetriever(BaseRetriever):

    vector_retriever: VectorStoreRetriever
    external_docs: List

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        vector_retriever_docs = self.vector_retriever.get_relevant_documents(query)
        return vector_retriever_docs + self.external_docs


class CustomDummyRetriever(BaseRetriever):

    external_docs: List

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        return self.external_docs
