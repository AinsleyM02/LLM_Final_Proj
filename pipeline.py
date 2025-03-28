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
from typing import Dict
from pathlib import Path

# Internal imports
from const import PATH_TO_DATA
from data_handler import DataHandler

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
    title_vector_dict = __traverse_data_pipeline(
        Path(PATH_TO_DATA), clean_data=clean_data, vectorize_data=vectorize_data
    )

    # Set up the local vector DB and add data to it
    vector_db = __set_up_local_vector_db(title_vector_dict)

    # Get LLM ready and run it
    __set_up_LLM(context=vector_db)
    __run_LLM()  # Ideally, we take input and output in the terminal.


def __traverse_data_pipeline(
    data_path: Path = Path(PATH_TO_DATA),
    clean_data: bool = True,
    vectorize_data: bool = True,
) -> Dict[str, str]:
    """
    Function that traverses the data pipeline.

    Parameters:
    - data_path: str, path to the data

    Returns:
    - title_vector_dict: dict, dictionary of titles and vectors of content
    """
    # Load the data
    data_handler = DataHandler(data_path=data_path)

    # If we need to clean the data and save it then let's do that
    if clean_data:
        data_handler.load_data()
        data_handler.clean_data()

    # If we need to vectorize the data then let's do that
    if vectorize_data:
        title_vector_dict = data_handler.vectorize_data()
    else:
        title_vector_dict = data_handler.load_vectorized_data()

    # Now we can vectorize the data
    title_vector_dict = data_handler.vectorize_data()

    return title_vector_dict


def __set_up_local_vector_db(title_vector_dict: Dict[str, str]):
    """
    Function that sets up a local vector DB if it doesn't already exist.
    """
    print("Setting up local vector DB...")
    print(title_vector_dict)
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


run_LLM(clean_data=clean_data)
