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
import re
import string

# Internal imports
from const import PATH_TO_DATA


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

    def __init__(self, data_path=PATH_TO_DATA) -> None:
        self.data_path = data_path
        self.data = []
        self.file_types = ["pdf", "docx", "txt", "nxml"]

    def load_data(self):
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.split(".")[-1] in self.file_types:
                    self.data.append(os.path.join(root, file))
                # Else delete the file
                else:
                    os.remove(os.path.join(root, file))

    def clean_data(self):
        for file in self.data:
            if file.split(".")[-1] == "pdf":
                self.clean_pdf(file)
            elif file.split(".")[-1] == "docx":
                self.clean_docx(file)
            elif file.split(".")[-1] == "txt":
                self.clean_txt(file)

    def clean_pdf(self, file):
        pass

    def clean_docx(self, file):
        pass

    def clean_txt(self, file):
        pass

    def clean_nxml(self, file):
        pass
