from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import PromptTemplate

from operator import itemgetter

import glob

files = glob.glob('Resources/*.pdf')

MODEL = 'llama3'

ollama_model = Ollama(model=MODEL)
ollama_embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

prompt_template = '''
You are a research assistant that is tasked with reviewing scientific papers. You are given different research questions that you have to answer by reading the papers. You will be given one paper at a time. You have to read the paper, create a summary of the paper, including its main contributions and drawbacks, and then answer the questions based on the content of the paper. Do not add information not present in the paper.

Context: {context}

Question: {question}
'''

for file in files:
    file_name = file.split("\\")[-1]
    result_file = f'Results/{file_name}_result.txt'
    print(f'Processing: {file_name}\n\n')

    loader = PyPDFLoader(file)

    print("Loading PDF...")
    pages = loader.load_and_split()

    print("Processing prompt...")
    prompt = PromptTemplate.from_template(prompt_template)

    print("Creating a vector store and retriever...")
    vectore_store = DocArrayInMemorySearch.from_documents(pages, embedding = ollama_embeddings)

    retriever = vectore_store.as_retriever()

    print("Creating a chain...")
    chain = (
    {
        'context': itemgetter('question') | retriever,
        'question': itemgetter('question')
    }
    | prompt
    | ollama_model
    | parser
    )

    questions = [
    'What is the contribution of the paper?',
    'What are the implications for programming education?',
    'What data has been used in the paper?',
    'What are the advantages and disadvantages of the proposed approach?'
    ]

    with open(result_file, 'a') as f:
            
        f.write('-------------------\n')
        f.write(f'File: {file_name}\n')

        for question in questions:
            f.write(f'Question: {question}\n')
            f.write(f"Answer: {chain.invoke({'question': question})}\n")
            f.write('-------------------\n')

