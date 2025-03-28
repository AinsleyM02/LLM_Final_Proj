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
from typing import Dict
from pathlib import Path

# Internal imports
from const import PATH_TO_DATA
from data_handler import DataHandler


def run_LLM():
    """
    Function that runs the LLM.
    """
    title_vector_dict = __traverse_data_pipeline(Path(PATH_TO_DATA))

    # Set up the local vector DB and add data to it
    vector_db = __set_up_local_vector_db(title_vector_dict)

    # Get LLM ready and run it
    __set_up_LLM(context=vector_db)
    __run_LLM()  # Ideally, we take input and output in the terminal.


def __traverse_data_pipeline(data_path: Path = Path(PATH_TO_DATA)) -> Dict[str, str]:
    """
    Function that traverses the data pipeline.

    Parameters:
    - data_path: str, path to the data

    Returns:
    - title_vector_dict: dict, dictionary of titles and vectors of content
    """
    # Load the data
    data_handler = DataHandler(data_path=data_path)
    data_handler.load_data()

    # Clean the data
    data_handler.clean_data()

    # Extract the text from the data and vectorize
    return data_handler.g()


def __set_up_local_vector_db():
    """
    Function that sets up a local vector DB if it doesn't already exist.
    """
    pass


def __set_up_LLM():
    """
    Function that sets up the LLM.
    """
    pass


def __run_LLM():
    """
    Function that runs the LLM.
    """
    pass
