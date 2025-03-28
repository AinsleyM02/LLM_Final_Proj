"""
Class that manages data for the RAG LLM. It prepares the data for the model, which should be populated in the data folder.

The data that will be accepeted in the following formats:
- PDF
- DOCX
- TXT
- NXML (specifically from statpearls)

Other formats will be ignored. Once cleaned the data will be extracted to a unified format (PDF) the vectorized to be loaded into a vector DB.
"""

# Standard imports
import os
from typing import Tuple
from pathlib import Path

# Internal imports
from utils import embed_text
from const import PATH_TO_DATA, PATH_TO_CLEANED_DATA

# External imports
import textract
import lxml.etree as et


class DataHandler:
    """
    Class that handles the data for the RAG LLM.

    Attributes:
    - data_path: str, path to the data folder
    - data: list, list of paths to the data files
    - file_types: list, list of accepted file types

    Methods:
    - load_data: loads the data from the data folder
    - clean_data: cleans the data
    """

    def __init__(
        self,
        data_path: Path = Path(PATH_TO_DATA),
        clean_data_path: Path = Path(PATH_TO_CLEANED_DATA),
    ) -> None:
        self.data_path = data_path
        self.clean_data_path = clean_data_path
        self.data = []
        self.file_types = ["pdf", "docx", "txt", "pptx", "nxml"]
        self.data_dict = {}  # Dictionary of titles and extracted content
        self.vectorized_data = {}  # Dictionary of titles and vectorized content

    def load_data(self) -> None:
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.split(".")[-1] in self.file_types:
                    self.data.append(os.path.join(root, file))
                # If .gitkeep file, ignore
                elif file == ".gitkeep":
                    continue
                # Else delete the file
                else:
                    print(f"File {file} is not in the accepted file types. Deleting...")
                    os.remove(os.path.join(root, file))

    def clean_data(self) -> None:
        for file in self.data:
            if file.split(".")[-1] in ["pdf", "PDF", "docx", "DOCX", "pptx", "PPTX"]:
                title, text = self.__clean_textract(file)
            elif file.split(".")[-1] == "nxml":
                title, text = self.__clean_nxml(file)

            if title is None or text is None:
                print(f"Error cleaning {file}. Skipping...")
                continue
            self.data_dict[title] = text

            # Sanitize the title by replacing path separators and invalid filename characters
            invalid_chars = '<>:"/\\|?*'
            safe_title = "".join(c for c in title if c not in invalid_chars)
            safe_title = safe_title.strip()  # Remove leading/trailing whitespace

            # Save the cleaned data
            with open(
                f"{self.clean_data_path / Path(safe_title + '.txt')}",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)

    def vectorize_data(self) -> None:
        """
        Function that vectorizes the data.
        """
        for title, text in self.data_dict.items():
            self.vectorized_data[title] = embed_text(text)

    def __clean_textract(self, file) -> Tuple[str, str]:
        """
        Function that cleans a PDF, DOCX, or PPTX file using textract.

        Parameters:
        - file: str, path to the file

        Returns:
        - title: str, title of the file
        - text: str, extracted text from the file
        """
        if file.split(".")[-1] not in ["pdf", "PDF", "docx", "DOCX", "pptx", "PPTX"]:
            raise ValueError(
                f"File {file} is not a PDF, DOCX, or PPTX file. Cannot clean using textract."
            )
        text = textract.process(file).decode("utf-8")
        title = file.split("/")[-1].split(".")[0]
        return title, text

    def __clean_nxml(self, file):
        """
        Cleans an nxml file. Specifically built to work with statpearls nxml files.

        Parameters:
        - file: str, path to the file

        Returns:
        - title: str, title of the file
        - text: str, extracted text from the file
        """
        print(f"Cleaning {file}...")
        # Ensure right file type
        if file.split(".")[-1] != "nxml":
            raise ValueError(f"File {file} is not an nxml file. Cannot clean as nxml.")

        # Set up file extraction
        tree = et.parse(file)
        root = tree.getroot()

        # Extract the title using title-group tag
        try:
            title = root.find(".//title-group/title").text
        except AttributeError:
            print(f"Error extracting title from {file}. Skipping...")
            return None, None

        print(f"Topic {title}...")

        # Extract tags with the sec-type as long as it does not have the value of "Continuing Education Activity"
        text = ""
        for sec in root.findall(".//sec"):
            sec_type = sec.get("sec-type")  # Access sec-type as an attribute of <sec>

            if sec_type is not None and sec_type != "Continuing Education Activity":
                for element in sec:  # Iterate over child elements in order
                    if element.tag == "title" and element.text:
                        text += element.text + "\n"
                    elif element.tag == "p" and element.text:
                        text += element.text + "\n"

        if text == "":
            print(f"Error extracting text from {file}. Skipping...")
            return None, None

        return f"StatPearls Chapter: {title}", text
