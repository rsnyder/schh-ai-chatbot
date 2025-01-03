#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hashlib
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import RecursiveUrlLoader

from pinecone import Pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('schh')

def read_pdfs(directory: str) -> list[str]:
  file_loader = PyPDFDirectoryLoader(directory) # Initialize a PyPDFDirectoryLoader object
  documents = file_loader.load() # Load PDF documents from the directory
  return  [doc.page_content for doc in documents] # Extract only the page content from each document
  
def chunk_text_for_list(docs: list[str], max_chunk_size: int = 1000) -> list[list[str]]:
  """
  Break down each text in a list of texts into chunks of a maximum size, attempting to preserve whole paragraphs.
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
  """Generate embeddings for a list of documents."""
  return [EMBEDDINGS.embed_documents(doc) for doc in documents]

def generate_short_id(content: str) -> str:
  """Generate a short ID based on the content using SHA-256 hash."""
  hash_obj = hashlib.sha256()
  hash_obj.update(content.encode('utf-8'))
  return hash_obj.hexdigest()

def combine_vector_and_text(documents: list[any], doc_embeddings: list[list[float]]) -> list[dict[str, any]]:
  
  data_with_metadata = []

  for doc_text, embedding in zip(documents, doc_embeddings):
    # Convert doc_text to string if it's not already a string
    if not isinstance(doc_text, str):
      doc_text = str(doc_text)

    doc_id = generate_short_id(doc_text) # Generate a unique ID based on the text content

    # Create a data item dictionary
    data_item = {
      'id': doc_id,
      'values': embedding[0],
      'metadata': {'text': doc_text},  # Include the text as metadata
    }
    data_with_metadata.append(data_item)  # Append the data item to the list

  return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
  index.upsert(vectors=data_with_metadata)

def load_pdfs():
  print ('Loading PDF files')
  docs = read_pdfs('knowledge-base/pdfs')
  chunked = chunk_text_for_list(docs=docs)
  doc_embeddings = generate_embeddings(documents=chunked)
  data_with_metadata = combine_vector_and_text(documents=chunked, doc_embeddings=doc_embeddings) 
  upsert_data_to_pinecone(data_with_metadata=data_with_metadata)

def load_markdown():
  print ('Loading markdown files')
  # loader = DirectoryLoader('knowledge-base/markdown', glob='**/*.md', show_progress=True)
  loader = DirectoryLoader('knowledge-base/markdown', glob='**/*.md', show_progress=True, loader_cls=UnstructuredMarkdownLoader)
  docs = loader.load()
  chunked = chunk_text_for_list([doc.page_content for doc in docs])
  doc_embeddings = generate_embeddings(documents=chunked)
  data_with_metadata = combine_vector_and_text(documents=chunked, doc_embeddings=doc_embeddings) 
  upsert_data_to_pinecone(data_with_metadata=data_with_metadata)

def load_website(root_url: str):
  loader = RecursiveUrlLoader(
      root_url,
      max_depth=1,
      # use_async=False,
      # extractor=None,
      # metadata_extractor=None,
      # exclude_dirs=(),
      # timeout=10,
      # check_response_status=True,
      # continue_on_failure=True,
      # prevent_outside=True,
      # base_url=None,
      # ...
  )
  docs = loader.load()
  for doc in docs:
    print(doc.page_content)
    print('\n')
  '''
  chunked = chunk_text_for_list([doc.page_content for doc in docs])
  print(len(chunked))
  doc_embeddings = generate_embeddings(documents=chunked)
  data_with_metadata = combine_vector_and_text(documents=chunked, doc_embeddings=doc_embeddings) 
  upsert_data_to_pinecone(data_with_metadata=data_with_metadata)
  '''

if __name__ == '__main__':
  # load_website('https://www.suncityhiltonhead.org/')
  load_pdfs()
  load_markdown()
