#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hashlib

from dotenv import load_dotenv

load_dotenv

from langchain_openai import OpenAIEmbeddings

from typing import List

# Acessing the various API KEYS
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('schh')

def read_doc(directory: str) -> list[str]:
    '''Function to read the PDFs from a directory.

    Args:
        directory (str): The path of the directory where the PDFs are stored.

    Returns:
        list[str]: A list of text in the PDFs.
    '''
    # Initialize a PyPDFDirectoryLoader object with the given directory
    file_loader = PyPDFDirectoryLoader(directory)
    
    # Load PDF documents from the directory
    documents = file_loader.load()
    
    # Extract only the page content from each document
    page_contents = [doc.page_content for doc in documents]
    
    return page_contents

def chunk_text_for_list(docs: list[str], max_chunk_size: int = 1000) -> list[list[str]]:
    """
    Break down each text in a list of texts into chunks of a maximum size, attempting to preserve whole paragraphs.

    :param docs: The list of texts to be chunked.
    :param max_chunk_size: Maximum size of each chunk in characters.
    :return: List of lists containing text chunks for each document.
    """

    def chunk_text(text: str, max_chunk_size: int) -> List[str]:
        # Ensure each text ends with a double newline to correctly split paragraphs
        if not text.endswith("\n\n"):
            text += "\n\n"
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        # Iterate over paragraphs and assemble chunks
        for paragraph in paragraphs:
            # Check if adding the current paragraph exceeds the maximum chunk size
            if (
                len(current_chunk) + len(paragraph) + 2 > max_chunk_size
                and current_chunk
            ):
                # If so, add the current chunk to the list and start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Add the current paragraph to the current chunk
            current_chunk += paragraph.strip() + "\n\n"
        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Apply the chunk_text function to each document in the list
    return [chunk_text(doc, max_chunk_size) for doc in docs]


def generate_embeddings(documents: list[any]) -> list[list[float]]:
    """
    Generate embeddings for a list of documents.

    Args:
        documents (list[any]): A list of document objects, each containing a 'page_content' attribute.

    Returns:
        list[list[float]]: A list containig a list of embeddings corresponding to the documents.
    """
    embedded = [EMBEDDINGS.embed_documents(doc) for doc in documents]
    return embedded

def generate_short_id(content: str) -> str:
    """
    Generate a short ID based on the content using SHA-256 hash.

    Args:
    - content (str): The content for which the ID is generated.

    Returns:
    - short_id (str): The generated short ID.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def combine_vector_and_text(
    documents: list[any], doc_embeddings: list[list[float]]
) -> list[dict[str, any]]:
    """
    Process a list of documents along with their embeddings.

    Args:
    - documents (List[Any]): A list of documents (strings or other types).
    - doc_embeddings (List[List[float]]): A list of embeddings corresponding to the documents.

    Returns:
    - data_with_metadata (List[Dict[str, Any]]): A list of dictionaries, each containing an ID, embedding values, and metadata.
    """
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        # Create a data item dictionary
        data_item = {
            "id": doc_id,
            "values": embedding[0],
            "metadata": {"text": doc_text},  # Include the text as metadata
        }

        # Append the data item to the list
        data_with_metadata.append(data_item)

    return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    """
    Upsert data with metadata into a Pinecone index.

    Args:
    - data_with_metadata (List[Dict[str, Any]]): A list of dictionaries, each containing data with metadata.

    Returns:
    - None
    """
    index.upsert(vectors=data_with_metadata)

full_document = read_doc('data/pdfs')
chunked_document = chunk_text_for_list(docs=full_document)
chunked_document_embeddings = generate_embeddings(documents=chunked_document)

# Let's see the dimension of our embedding model so we can set it up later in pinecone
print (len(chunked_document_embeddings))

data_with_meta_data = combine_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings) 
upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)
