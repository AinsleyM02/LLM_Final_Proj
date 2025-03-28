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
import fitz
from pptx import Presentation
from docx import Document
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
            if file.split(".")[-1] in ["pdf", "PDF"]:
                title, text = self.__clean_pdf(file)
            elif file.split(".")[-1] in ["txt", "TXT"]:
                title, text = self.__clean_txt(file)
            elif file.split(".")[-1] in ["docx", "DOCX"]:
                title, text = self.__clean_docx(file)
            elif file.split(".")[-1] == "nxml":
                title, text = self.__clean_nxml(file)
            else:
                print(
                    f"File {file} is not in the accepted file types. Should have been deleted... Skipping..."
                )
                continue

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

    def __clean_pdf(self, file: str) -> Tuple[str, str]:
        """
        Cleans a PDF file.

        Parameters:
        - file: str, path to the file

        Returns:
        - title: str, title of the file
        - text: str, extracted text from the file
        """
        print(f"Cleaning {file}...")
        # Ensure right file type
        if file.split(".")[-1] not in ["pdf", "PDF"]:
            raise ValueError(f"File {file} is not a PDF file. Cannot clean as PDF.")
        # Open the PDF file
        pdf = fitz.open(file)
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        title = file.split("/")[-1].split(".")[0]
        return title, text

    def __clean_txt(self, file: str) -> Tuple[str, str]:
        """
        Cleans a TXT file.

        Parameters:
        - file: str, path to the file

        Returns:
        - title: str, title of the file
        - text: str, extracted text from the file
        """
        print(f"Cleaning {file}...")
        # Ensure right file type
        if file.split(".")[-1] not in ["txt", "TXT"]:
            raise ValueError(f"File {file} is not a TXT file. Cannot clean as TXT.")
        # Open the TXT file
        with open(file, "r") as f:
            text = f.read()
        title = file.split("/")[-1].split(".")[0]
        return title, text

    def __clean_docx(self, file: str) -> Tuple[str, str]:
        """
        Cleans a DOCX file.

        Parameters:
        - file: str, path to the file

        Returns:
        - title: str, title of the file
        - text: str, extracted text from the file
        """
        print(f"Cleaning {file}...")
        # Ensure right file type
        if file.split(".")[-1] not in ["docx", "DOCX"]:
            raise ValueError(f"File {file} is not a DOCX file. Cannot clean as DOCX.")
        # Open the DOCX file
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text
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
