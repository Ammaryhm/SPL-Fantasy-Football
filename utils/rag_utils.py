import os
import logging
import streamlit as st  # Used for @st.cache_resource, st.error, st.warning, st.stop
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

logger = logging.getLogger(__name__)


@st.cache_resource
def setup_rag_components(rag_prompt_template_str: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    csv_file_paths = [
        "data/spl_mock_data.csv",
        "data/spl_win_status.csv"
    ]

    docs = []

    for csv_file_path in csv_file_paths:
        try:
            new_docs = CSVLoader(file_path=csv_file_path).load()

            if not new_docs:
                logger.warning(f"No documents loaded from {csv_file_path}. Please check your CSV file and path.")
            docs.extend(new_docs)

        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file_path}: {e}")
            st.error(f"Error loading data for the Virtual Assistant: {e}. Please ensure '{csv_file_path}' exists and is accessible.")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=350, chunk_overlap=50
    )

    chunked_documents = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunked_documents)} chunks.")

    if not chunked_documents:
        st.warning("No document chunks available after splitting. Cannot create vector store.")
        st.stop()

    faiss_vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
    )

    retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template_str)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG components setup complete.")
    logger.info(f"Loaded {len(docs)} docs")
    logger.info(f"First doc: {docs[0] if docs else 'No docs loaded'}")
    return rag_chain


def retrieve_and_log(retriever, query):
    retrieved = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved docs: {retrieved}")
    return retrieved
