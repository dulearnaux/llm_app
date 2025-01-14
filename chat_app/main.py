import os

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

if __name__ == '__main__':
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    # flatten list
    docs_flat = [i for lst in docs for i in lst]

    # split docs into chunks to make retrieval more efficient
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0)

    doc_chunks = text_splitter.split_documents(docs_flat)
    # base_url = st.text_input('base_url', 'http://127.0.0.0:11434')
    base_url = 'http://127.0.0.0:11434'

    embeddings = {
        'openai': OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
        'llama3.1:8b': OllamaEmbeddings(model="llama3.1:8b"),
        'llama3.2:1b': OllamaEmbeddings(model="llama3.2:1b")
    }

    emb = st.sidebar.radio(
        "Choose your embedding.",
        ["openai", "llama3.1:8b", "llama3.2:1b"], index=0,
        captions=['uses online api (text-embedding-ada-002), loads the quickest'
                  ', but costs $$$',
                  'local 8b, works with RAG, but slow initial load',
                  'local 1b, doesnt use RAG but loads quicker'])

    # create embeddings for the docs. One set of embeddings for each chunk.
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_chunks,
        embedding=embeddings[emb])
    # Use the vectorstore as a retriever
    retriever = vectorstore.as_retriever(k=4)

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
            Use the following documents to answer the question. If the
            information is not in the documents, then use your general 
            knowledge to answer the question. Let me know which you used.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
            """,
        input_variables=["question", "documents"]
    )
    model = st.sidebar.radio(
        "Choose your language model.",
        ["llama3.1:8b", "llama3.2:1b"], index=1,
        captions=['local 8b, works with RAG, but slow initial load',
                  'local 1b, doesnt use RAG but loads quicker'])
    llm = ChatOllama(
        model=model, temperature=0, stop=["<|eot_id|>"], base_url=base_url)

    rag_chain = prompt | llm | StrOutputParser()

    class RAGApp:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain

        def run(self, question):
            documents = self.retriever.invoke(question)
            doc_texts = "\\n".join([doc.page_content for doc in documents])
            answer = self.rag_chain.invoke(
                {"question": question, "documents": doc_texts})
            return answer

    st.header('LLM RAG Demo')
    st.write('Demo of an LLM RAG using Ollama and Llama models.\n')
    tmp = '\n \n'.join(urls)
    st.sidebar.subheader('RAG docs are:')
    st.sidebar.write(f'{tmp}')

    rag_app = RAGApp(retriever, rag_chain)
    q = st.text_area('Write your query here', 'what is prompt engineering?')
    a = rag_app.run(q)
    st.write(a)
