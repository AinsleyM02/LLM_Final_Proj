"""
Script for running the pipeline.

This should:
- Load the data
- Clean the data
- Extract the text from the data and vectorize
- Set up a local vector DB if it doesn't already exist (else connect to it)
- Load the vectorized data into the vector DB (if it doesn't already exist)
- Run the LLM on the data
- Output the results
"""

# Standard imports
import argparse
import shutil
import requests
import sys
from typing import Dict
from pathlib import Path

# Internal imports
from const import PATH_TO_DATA
from data_handler import DataHandler
from chroma import ChromaDB

# Create an argument parser
parser = argparse.ArgumentParser(
    description="Run pipeline with command-line parameters."
)
parser.add_argument(
    "--clean_data",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Whether to setup data (default: True)",
)
parser.add_argument(
    "--vectorize_data",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Whether to vectorize data (default: True)",
)
args = parser.parse_args()

# See if None
if args is None:
    clean_data = True
    vectorize_data = True

# Use the argument
if args.clean_data:
    clean_data = True
else:
    clean_data = False

if args.vectorize_data:
    vectorize_data = True
else:
    vectorize_data = False


def run_LLM(clean_data: bool = True, vectorize_data: bool = True):
    """
    Function that runs the LLM.
    """
    datahandler = __traverse_data_pipeline(
        Path(PATH_TO_DATA), clean_data=clean_data, vectorize_data=vectorize_data
    )

    # Set up the local vector DB and add data to it
    vector_db = __set_up_local_vector_db(datahandler)

    # To verify this works search and print results
    # context = vector_db.search("Teach me about medullary thyroid cancer")
    # print("Context: ", context)

    # Get LLM ready and run it
    __set_up_and_run_LLM(vector_db)


def __traverse_data_pipeline(
    data_path: Path = Path(PATH_TO_DATA),
    clean_data: bool = True,
    vectorize_data: bool = True,
) -> DataHandler:
    """
    Function that traverses the data pipeline.

    Parameters:
    - data_path: str, path to the data

    Returns:
    - data_handler: DataHandler, the data handler object
    """
    # Load the data
    data_handler = DataHandler(data_path=data_path)

    # If we need to clean the data and save it then let's do that
    if clean_data:
        data_handler.load_data()
        data_handler.clean_data()

    # If we need to vectorize the data then let's do that
    if vectorize_data:
        data_handler.vectorize_data()
    else:
        data_handler.load_vectorized_data()

    return data_handler


def __set_up_local_vector_db(datahandler: DataHandler) -> ChromaDB:
    """
    Function that sets up a local vector DB if it doesn't already exist.
    """
    # Set up the local vector DB and add data to it
    vector_db = ChromaDB()
    vector_db.add_data(datahandler)

    return vector_db

def __get_llm():
    """
    Returns a function that sends a prompt to Ollama's local API using the specified model.
    It checks for the presence of the 'ollama' CLI to ensure the backend is available.
    """

    # Check if Ollama is installed
    if shutil.which("ollama") is None:
        print("Ollama is not installed. Please install it from https://ollama.com/download")
        sys.exit(1)

    def llm(prompt: str, model="mistral:instruct", host="http://localhost:11434"):
        try:
            response = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()

        except requests.exceptions.RequestException as e:
            print(f"Error contacting Ollama at {host}: {e}")
            return "[LLM Error: Could not get a response]"

    return llm

# Function to set up and run LLM with vector DB
def __set_up_and_run_LLM(vector_db):
    """
    Function that sets up the LLM.
    Note vector_db is the instance of ChromaDB that has been set up and is passed in.
    This function will handle the LLM setup and querying.
    """
    # We can discuss this once we have the vector DB set up.
    # Prob worth creating a class? that loads the LLM or downloads it if not already present.
    # Then it internally can handle querying the vector DB for context and running the LLM on the input.
    # This can be a while loop that they enter 'q' to quit.
    
    # Load the LLM with the function created previousl
    llm = __get_llm()
    #set system prompt
    SYSTEM_PROMPT = """You are a medical student. You use reputable retrieved documents to answer questions. Use sources provided to answer the provided question. 
    Provide clear and concise answers in the format citing the sources you used. If you don't know the answer, say "I don't know".
    """
    response = None  # define this up front
    # while the user has not entered 'q' to quit, keep asking for input
    while True:
        query = input("Enter your question (or type 'q' to quit): ").strip()
        if query.lower() == 'q':
            print("Exiting...")
            break
        # calls the instance of the vector DB to get the context for the question
        context = vector_db.search(query)
        prompt = f"{SYSTEM_PROMPT}\n\nSources:\n{context}\n\nQuestion: {query}"

        response = llm(prompt)
        print(response)

    return response



run_LLM(clean_data=clean_data, vectorize_data=vectorize_data)
