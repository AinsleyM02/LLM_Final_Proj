"""
File that contains a class for:
- Setting up a local vector DB using Chroma
- Adding data to it
- Searching for data in it
"""

# Standard library imports
import gzip
import uuid
import base64
from tqdm import tqdm
from typing import Dict

# Third party imports
import chromadb
import numpy as np

# Local application imports
from utils import embed_text_no_chunk
from const import PATH_TO_VECTOR_DB
from config import DEFAULT_RESULTS_PER_SEARCH
from data_handler import DataHandler


class ChromaDB:
    """
    Class for setting up a local vector DB using Chroma.
    """

    def __init__(
        self,
        persist_directory: str = PATH_TO_VECTOR_DB,
        collection_name: str = "medical_school",
    ):
        """
        Initialize the ChromaDB class and set up the database if not already present.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        # Create a collection
        self.collection = chroma_client.get_or_create_collection(
            name=self.collection_name
        )

        # Check if the collection already exists
        if self.collection.count() == 0:
            print(f"Collection '{self.collection_name}' created.")
            self.db_populated = False
        else:
            print(f"Collection '{self.collection_name}' already exists.")
            self.db_populated = True

    def __compress_text(self, text: str) -> str:
        """Compress text using gzip and encode it with base64 for safe storage."""
        return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("utf-8")

    def __decompress_text(self, encoded: str) -> str:
        return gzip.decompress(base64.b64decode(encoded.encode("utf-8"))).decode(
            "utf-8"
        )

    def add_data(self, handler: DataHandler) -> None:
        """
        Add data to the ChromaDB collection.

        Parameters:
            handler (DataHandler): Instance of DataHandler class to load and process data.
        """
        # Check if the database is already populated
        if self.db_populated:
            print("Database already populated. Skipping data addition.")
            return

        # Load vectorized data
        # Iterate over the data and add it to the collection
        for section_name, emb_and_text in tqdm(handler.load_vectorized_data()):
            embeddings = np.array(emb_and_text["embeddings"])
            texts = emb_and_text["texts"]

            # Compress all texts
            compressed_texts = [self.__compress_text(t) for t in texts]

            if len(embeddings) < 5461:
                self.collection.add(
                    ids=[str(uuid.uuid4()) for _ in range(len(embeddings))],
                    embeddings=embeddings,
                    metadatas=[
                        {"text": compressed_texts[i], "title": section_name}
                        for i in range(len(compressed_texts))
                    ],
                )
            else:
                chunk_size = 5461
                for i in range(0, len(embeddings), chunk_size):
                    chunk = embeddings[i : i + chunk_size]
                    chunk_texts = compressed_texts[i : i + chunk_size]
                    self.collection.add(
                        ids=[str(uuid.uuid4()) for _ in range(len(chunk))],
                        embeddings=chunk,
                        metadatas=[
                            {"text": chunk_texts[j], "title": section_name}
                            for j in range(len(chunk))
                        ],
                    )

    def search(
        self, search_str: str, n_results: int = DEFAULT_RESULTS_PER_SEARCH
    ) -> Dict[str, str]:
        """
        Search for a string in the ChromaDB collection.

        Parameters:
            search_str (str): The string to search for.
            n_results (int): The number of results to return.

        Returns:
            Dict[str, str]: A dictionary with IDs and their corresponding texts.
        """
        # Ensure search string is not empty and is a string
        if not search_str or not isinstance(search_str, str):
            raise ValueError("Search string must be a non-empty string.")

        results = self.collection.query(
            query_embeddings=[embed_text_no_chunk(search_str)],
            n_results=n_results,
            include=["distances", "metadatas"],
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        decompressed_results = {
            f"{id}_{metadata["title"]}": self.__decompress_text(metadata["text"])
            for id, metadata in zip(ids, metadatas)
        }

        # Remove duplicates from the results
        unique_results = {}
        for id, text in decompressed_results.items():
            if id not in unique_results:
                unique_results[id] = text

        return unique_results
