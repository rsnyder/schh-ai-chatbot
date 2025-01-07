#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse, json, os, sys
import hashlib
from typing import List

from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
  
def generate_short_id(content: str) -> str:
  """Generate a short ID based on the content using SHA-256 hash."""
  hash_obj = hashlib.sha256()
  hash_obj.update(content.encode('utf-8'))
  return hash_obj.hexdigest()

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]], index_name, **kwargs) -> None:
  print (f'Upsert data to Pinecone: {len(data_with_metadata)} index_name={index_name}')
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  index.upsert(vectors=data_with_metadata)

def chunk_markdown(path):
  print (f'Chunking Markdown file: {path}')
  markdown = open(path).read()
  md_header_splits = MarkdownHeaderTextSplitter(
    headers_to_split_on = [ ('#', 'Header 1'), ('##', 'Header 2') ], 
    strip_headers=False
  ).split_text(markdown)
  # Char-level splits
  docs = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
  ).split_documents(md_header_splits)
  return docs

def chunk_pdf(path):
  print (f'Chunking PDF file: {path}')
  loader = PyPDFLoader(path)
  pages = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
  docs = text_splitter.split_documents(pages)
  return docs

def load(path, dryrun=False, verbose=False, **kwargs):
  logger.info(f'Loading {path} into Pinecone, dryrun={dryrun}')
  file_type = path.split('.')[-1]
  docs = []
  
  if 'md' == file_type:
    docs = chunk_markdown(path)
  elif 'pdf' == file_type:
    docs = chunk_pdf(path)

  if docs:
    if verbose:
      print(f'\nDocs: {len(docs)}' + '\n\n---\n')
      for doc in docs:
        print(doc.page_content + '\n\n---\n')
    else:
      print(f'Docs={len(docs)}')
    
    print('Generating embeddings')
    doc_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])
    
    data_with_metadata = []
    for doc, embedding in zip(docs, doc_embeddings):
      if file_type == 'md':
        id = ': '.join([doc.metadata[key] for key in sorted(doc.metadata)]).replace(' ', '_')
      elif file_type == 'pdf':
        id = generate_short_id(doc.page_content)
      data_item = {
          'id': id,
          'values': embedding,
          'metadata': doc.metadata | {'text': doc.page_content},  # add text as metadata
      }
      data_with_metadata.append(data_item)  # Append the data item to the list
    
    if not dryrun:
      upsert_data_to_pinecone(data_with_metadata=data_with_metadata, **kwargs)

if __name__ == '__main__':
  BASEDIR = os.path.abspath(os.path.dirname(__file__))

  parser = argparse.ArgumentParser(description='SCHH Knowledge Base Loader')  
  parser.add_argument('--dryrun', default=False, action='store_true', help='Don\'t load data into Pinecone')
  parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
  parser.add_argument('--content', default=BASEDIR, help='Knowledge base root directory')
  parser.add_argument('--index_name', default='schh', help='Pinecone index name')
  parser.add_argument('path', help='Path to a file or directory to load')

  args = vars(parser.parse_args())
  print(json.dumps(args))
  
  path = sys.argv[1]
  if os.path.isdir(args['path']):
    for root, dirs, files in os.walk(args['path']):
      for file in files:
        name, extension = os.path.splitext(file)
        if (extension == '.pdf' and f'{name}.md' in files):
          continue # Skip PDF files that have corresponding markdown files
        args['path'] = os.path.join(root, file)
        load(**args)
  elif os.path.isfile(args['path']):
    load(**args)