import random
import os
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

def get_openai_api_key():
    """
    Get OpenAI API key from Streamlit secrets or environment variables
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        # Fallback to environment variable (for local development)
        return os.getenv("OPENAI_API_KEY")

# Get API key using the helper function
OPENAI_API_KEY = get_openai_api_key()

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment variables or Streamlit secrets.")
    raise ValueError("OPENAI_API_KEY not set in environment variables or Streamlit secrets.")

FACTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'trivia.txt')

 
class TriviaQuestion(BaseModel):
    question: str = Field(description="The trivia question generated from the fact.")
    options: list[str] = Field(description="A list of exactly four answer options, including one correct answer and three plausible distractors.")
    correct_answer_index: int = Field(description="The zero-based index of the correct answer in the options list (0, 1, 2, or 3).")

 
@st.cache_resource
def get_llm():
    """
    Get the LLM instance with proper caching
    """
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY
    )

def get_parser():
    """
    Get the JSON parser instance
    """
    return JsonOutputParser(pydantic_object=TriviaQuestion)

def get_trivia_prompt_template():
    """
    Get the trivia prompt template
    """
    parser = get_parser()
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant specialized in creating engaging multiple-choice trivia questions about the Saudi Pro League. Your task is to generate one question, four distinct and plausible answer options (one correct, three distractors), and specify the index of the correct answer, all based on a given fact. Ensure the correct answer is accurately derived from the fact."),
            ("human", "Generate a trivia question, four options, and the correct answer index (0-3) based on the following fact. The output must be valid JSON following the schema:\n```json\n{format_instructions}\n```\n\nFact: {fact}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

@st.cache_resource
def get_trivia_generation_chain():
    """
    Get the trivia generation chain with caching
    """
    llm = get_llm()
    parser = get_parser()
    trivia_prompt_template = get_trivia_prompt_template()
    return trivia_prompt_template | llm | parser

def load_facts(filepath=FACTS_FILE):
    facts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    facts.append(line)
    except FileNotFoundError:
        logger.error(f"Facts file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error loading facts from {filepath}: {e}")
        return []
    return facts

def generate_trivia_question_from_fact():
    facts = load_facts()
    if not facts:
        logger.warning("No facts loaded to generate trivia questions.")
        return None

    selected_fact = random.choice(facts)
    logger.info(f"Generating question from fact: '{selected_fact}'")

    try:
        # Get the chain
        trivia_generation_chain = get_trivia_generation_chain()
        
        # Invoke the Langchain chain to generate the question
        generated_question_data = trivia_generation_chain.invoke({"fact": selected_fact})

        # Validate the generated data against the Pydantic model
        validated_question = TriviaQuestion(**generated_question_data)

        # Ensure exactly 4 options are returned and index is valid
        if len(validated_question.options) != 4:
            logger.warning(f"Generated question has {len(validated_question.options)} options, expected 4. Skipping: {validated_question.question}")
            return None

        if not (0 <= validated_question.correct_answer_index < 4):
            logger.warning(f"Generated question has invalid correct_answer_index: {validated_question.correct_answer_index}. Skipping: {validated_question.question}")
            return None

        return {
            "question": validated_question.question,
            "options": validated_question.options,
            "correct_answer_index": validated_question.correct_answer_index
        }

    except ValidationError as e:
        logger.error(f"Pydantic validation error for generated question: {e}")
        logger.error(f"Raw generated data: {generated_question_data}")
        return None

    except Exception as e:
        logger.error(f"Error generating trivia question with LLM from fact '{selected_fact}': {e}")
        return None


def validate_answer(selected_option_index, correct_answer_index):
    """
    Validates if the selected answer is correct.
    """
    return selected_option_index == correct_answer_index

if __name__ == '__main__':
    # Simple test for trivia_util.py with LLM generation
    print("Attempting to generate a trivia question using GPT-4o...")
    q = generate_trivia_question_from_fact()

    if q:
        print(f"\nGenerated Question: {q['question']}")
        for i, option in enumerate(q['options']):
            print(f"{i}. {option}")
        print(f"Correct answer index: {q['correct_answer_index']}")

        # Simulate correct and incorrect answers
        correct_idx = q['correct_answer_index']

        # Simulate user selection
        user_choice_correct = correct_idx
        user_choice_incorrect = (correct_idx + 1) % 4 # Pick a wrong one

        print(f"Validating user choice {user_choice_correct} (should be correct): {validate_answer(user_choice_correct, correct_idx)}")
        print(f"Validating user choice {user_choice_incorrect} (should be incorrect): {validate_answer(user_choice_incorrect, correct_idx)}")
    else:
        print("Failed to generate a question. Check logs for errors (e.g., API key, fact file, LLM output format).")
