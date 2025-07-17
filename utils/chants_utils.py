import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_resource
def setup_chant_chain(chant_prompt_template_str):
    """
    Set up the chant generation chain with proper error handling
    """
    try:
        logger.info("Setting up Chant Generation LLM chain...")
        
        # Updated ChatOpenAI initialization - use 'model' instead of 'model_name'
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Changed from model_name to model
            temperature=0.8,
            # Ensure we're using the correct API key from environment
            openai_api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["team_or_theme"],
            template=chant_prompt_template_str
        )
        
        # Create the chain
        chain = prompt_template | llm | StrOutputParser()
        
        logger.info("Chant Generation LLM chain setup complete!")
        return chain
        
    except Exception as e:
        logger.error(f"Error setting up chant chain: {str(e)}")
        st.error(f"Failed to setup chant generation: {str(e)}")
        return None

def generate_chant(chain, team_or_theme):
    """
    Generate a chant using the provided chain
    """
    try:
        if chain is None:
            return "Error: Chain not properly initialized"
            
        result = chain.invoke({"team_or_theme": team_or_theme})
        return result
        
    except Exception as e:
        logger.error(f"Error generating chant: {str(e)}")
        return f"Error generating chant: {str(e)}"
