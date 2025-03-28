"""
File that contains utility functions for the project.
These are not specific to a certain operation and used in multiple places.

Namely:
- embedding text: used to embed text from queries as well as data
- decoding text: used to decode embeddings to text
"""

# Internal imports
from const import EMBEDDING_MODEL

# External imports
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> np.array:
    """
    Function that embeds text.

    Parameters:
    - text: str, text to embed

    Returns:
    - np.array, embedding of the text
    """
    return embedding_model.encode(text)


def decode_text(embedding: np.array) -> str:
    """
    Function that decodes an embedding.

    Parameters:
    - embedding: np.array, embedding to decode

    Returns:
    - str, decoded text
    """
    return embedding_model.decode(embedding)
