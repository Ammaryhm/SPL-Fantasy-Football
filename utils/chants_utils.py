import logging
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Initialize module-level logger
logger = logging.getLogger(__name__)

@st.cache_resource
def setup_chant_chain(chant_prompt_template_str: str) -> LLMChain:
    """Sets up and caches an LLMChain for chant generation using OpenAI."""
    logger.info("Setting up Chant Generation LLM chain...")

    # Instantiate the LLM with the correct parameter name (model_name)
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.8
    )

    # Build the prompt template and chain
    chant_prompt = ChatPromptTemplate.from_template(chant_prompt_template_str)
    chant_chain = LLMChain(llm=llm, prompt=chant_prompt)

    return chant_chain
