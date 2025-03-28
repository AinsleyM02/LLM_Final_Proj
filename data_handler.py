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
    - clean_pdf: cleans a PDF file
    - clean_docx: cleans a DOCX file
    - clean_txt: cleans a TXT file
    """

    def __init__(
        self,
        data_path: Path = Path(PATH_TO_DATA),
        clean_data_path: Path = Path(PATH_TO_CLEANED_DATA),
    ) -> None:
        self.data_path = data_path
        self.clean_data_path = clean_data_path
        self.data = []
        self.file_types = ["pdf", "docx", "txt", "nxml"]
        self.data_dict = {}  # Dictionary of titles and extracted content
        self.vectorized_data = {}  # Dictionary of titles and vectorized content

    def load_data(self) -> None:
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.split(".")[-1] in self.file_types:
                    self.data.append(os.path.join(root, file))
                # Else delete the file
                else:
                    os.remove(os.path.join(root, file))

    def clean_data(self) -> None:
        for file in self.data:
            if file.split(".")[-1] == "pdf":
                title, text = self.__clean_pdf(file)
            elif file.split(".")[-1] == "docx":
                title, text = self.__clean_docx(file)
            elif file.split(".")[-1] == "txt":
                title, text = self.__clean_txt(file)
            elif file.split(".")[-1] == "nxml":
                title, text = self.__clean_nxml(file)
            self.data_dict[title] = text

            # Save the cleaned data
            with open(f"{self.clean_data_path + title}.txt", "w") as f:
                f.write(text)

    def vectorize_data(self) -> None:
        """
        Function that vectorizes the data.
        """
        for title, text in self.data_dict.items():
            self.vectorized_data[title] = embed_text(text)

    def __clean_pdf(self, file) -> Tuple[str, str]:
        """
        Function that cleans a PDF file.

        Parameters:
        - file: str, path to the PDF file

        Returns:
        - title: str, title of the PDF
        - text: str, extracted text from the PDF
        """

    def __clean_docx(self, file):
        pass

    def __clean_txt(self, file):
        pass

    def __clean_nxml(self, file):
        pass
