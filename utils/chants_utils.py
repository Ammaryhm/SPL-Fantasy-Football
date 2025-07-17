import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

 

logger = logging.getLogger(__name__)


@st.cache_resource
def setup_chant_chain(chant_prompt_template_str: str):
    logger.info("Setting up Chant Generation LLM chain...")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.8)
    chant_prompt = ChatPromptTemplate.from_template(chant_prompt_template_str)

    chant_chain = (

        {"user_input": lambda x: x}
        | chant_prompt
        | llm
        | StrOutputParser()
    )
    logger.info("Chant Generation LLM chain setup complete.")

    return chant_chain

 
