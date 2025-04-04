"""
File that contains utility functions for the project.
These are not specific to a certain operation and used in multiple places.

Namely:
- embedding text: used to embed text from queries as well as data
- decoding text: used to decode embeddings to text
"""

# Standard imports
import re

# Internal imports
from const import EMBEDDING_MODEL

# External imports
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def simple_sentence_splitter(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def embed_text(text: str, max_chunk_size: int = 256) -> np.array:
    """
    Function that embeds text by chunking if necessary.

    Parameters:
    - text: str, text to embed
    - max_chunk_size: int, maximum number of tokens per chunk

    Returns:
    - np.array, stacked embeddings of all chunks
    """
    sentences = simple_sentence_splitter(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk).split()) > max_chunk_size:
            # Remove last sentence if it exceeds limit
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    return np.vstack(embeddings)


def decode_text(embedding: np.array) -> str:
    """
    Function that decodes an embedding.

    Parameters:
    - embedding: np.array, embedding to decode

    Returns:
    - str, decoded text
    """
    return embedding_model.decode(embedding)
