from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


class TDMPrompt:

    def __init__(self, system_prompt, model_type):
        if model_type == 'llama-2-chat-70b':
            initial_template = "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n{question}\n\nRetrieved document contents:\n\n{context}.\n\n Output json: [/INST]"
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["context", "question"])

        elif model_type == 'llama-3-instruct-70b':
            initial_template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
                                system_prompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
                                "{question}\n\nRetrieved document contents:\n\n{context}.\n\n Output json: <|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["context", "question"])

        elif model_type == 'gpt4-turbo-128k':
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate.from_template("{question}\n\nRetrieved document contents:\n\n{context}.\n\n Output json:\n\n")
            self.prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        elif model_type == 'Mixtral-8x7B-Instruct-v0.1':
            initial_template = "<s>[INST] " + system_prompt + "\n\n{question}\n\nRetrieved document contents:\n\n{context}.\n\n Output json: [/INST]"
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["context", "question"])

        else:
            raise ValueError('Unrecognized model type')


class NormalizationPrompt:

    def __init__(self, system_prompt, model_type):
        if model_type == 'llama-2-chat-70b':
            initial_template = "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\nItem list: {items}\n\nInput: {input}\n\nAnswer: [/INST]"
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["items", "input"])

        elif model_type == 'llama-3-instruct-70b':
            initial_template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
                                system_prompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
                                "Item list: {items}\n\nInput: {input}\n\nAnswer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>")
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["items", "input"])

        elif model_type == 'gpt4-turbo-128k':
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message_prompt = HumanMessagePromptTemplate.from_template("\n\nItem list: {items}\n\nInput: {input}\n\nAnswer: ")
            self.prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        elif model_type == 'Mixtral-8x7B-Instruct-v0.1':
            initial_template = "<s>[INST] " + system_prompt + "\n\nItem list: {items}\n\nInput: {input}\n\nAnswer: [/INST]"
            self.prompt_template = PromptTemplate(template=initial_template, input_variables=["items", "input"])

        else:
            raise ValueError('Unrecognized model type')
