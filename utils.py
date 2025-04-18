"""
File that contains utility functions for the project.
These are not specific to a certain operation and used in multiple places.

Namely:
- embedding text: used to embed text from queries as well as data
- decoding text: used to decode embeddings to text
"""

# Standard imports
import re
from typing import Tuple

# Internal imports
from config import EMBEDDING_MODEL

# External imports
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# Using this and word length for speed's sake
def sentence_splitter(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def embed_text_no_chunk(text: str) -> np.array:
    """
    Function that embeds text without chunking.

    Parameters:
    - text: str, text to embed

    Returns:
    - np.array, embedding of the text
    """
    # We can use the embedding model to encode the text
    return embedding_model.encode(text)


def embed_text(text: str, max_chunk_size: int = 256) -> Tuple[np.ndarray, list[str]]:
    """
    Function that embeds text by chunking if necessary.

    Parameters:
    - text: str, text to embed
    - max_chunk_size: int, maximum number of tokens per chunk

    Returns:
    - Tuple: (np.array of embeddings, list of corresponding chunks)
    """
    sentences = sentence_splitter(text)
    chunks = []  # A list of all chunks
    current_chunk = []  # A list of sentences in the current chunk

    # Go through the sentences and add them to the current chunk
    for sentence in sentences:
        current_chunk.append(sentence)
        # Until the current chunk is too big, then we remove the last sentence and append the current chunk to the list of chunks
        if len(" ".join(current_chunk).split()) > max_chunk_size:
            # Remove last sentence if it exceeds limit
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Now start a new chunk with the last sentence

    # Add the last chunk if it exists
    # If the last chunk is empty, then we don't need to add it to the list of chunks
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Now we have a list of chunks, let's embed them
    # We can use the embedding model to encode the chunks
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    # Now we have a list of embeddings, let's stack them
    # We can use np.vstack to stack the embeddings
    return np.vstack(embeddings), chunks  # Return the embeddings and the chunks
