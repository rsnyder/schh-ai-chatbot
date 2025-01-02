#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import json, os
import asyncio
from typing import AsyncIterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

from langchain_openai import OpenAIEmbeddings  
from langchain.callbacks import AsyncIteratorCallbackHandler
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA 
from langchain.schema import HumanMessage
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

from pydantic import BaseModel

import uvicorn

app = FastAPI(title='OpenAI API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'] )

class Message(BaseModel):
    content: str

openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

async def send_message(content: str) -> AsyncIterable[str]:
  
  model_name = 'text-embedding-ada-002'  
  index_name = 'schh'
  text_field = 'text'

  pc = Pinecone(api_key=pinecone_api_key)
  index = pc.Index(index_name)
  embeddings = OpenAIEmbeddings( model=model_name, openai_api_key=openai_api_key )
  vectorstore = PineconeVectorStore( index, embeddings, text_field )  

  callback = AsyncIteratorCallbackHandler()
  llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-4o-mini',
    temperature=0.0,
    streaming=True,
    verbose=True,
    callbacks=[callback]
  )

  task = asyncio.create_task (
    RetrievalQA.from_chain_type( llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever() ).invoke(content)
    #llm.agenerate(messages=[[HumanMessage(content=content)]])
  )

  try:
    async for token in callback.aiter():
      yield token
  except Exception as e:
    print(f'Caught exception: {e}')
  finally:
    callback.done.set()

  await task

@app.get('/')
def docs():
  return RedirectResponse(url='/docs')

@app.post('/chat')
async def stream_chat(message: Message):
  generator = send_message(message.content)
  return StreamingResponse(generator, media_type="text/event-stream")

@app.post('/stream')
async def stream_qa(message: Message):
  """Endpoint for streaming question answering."""
  query = message.content
  model_name = 'text-embedding-ada-002'  
  index_name = 'schh'
  text_field = 'text'

  pc = Pinecone(api_key=pinecone_api_key)
  index = pc.Index(index_name)
  embeddings = OpenAIEmbeddings( model=model_name, openai_api_key=openai_api_key )
  vectorstore = PineconeVectorStore( index, embeddings, text_field )  

  llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-4o-mini',
    temperature=0.0,
    streaming=True
  )
  
  # Create a callback handler for streaming
  callback_handler = AsyncIteratorCallbackHandler()

  # Create the RetrievalQA chain
  qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff', 
    retriever=vectorstore.as_retriever(),
    callbacks=[callback_handler]
  )

  # Run the chain asynchronously
  async def generate_response():
    await qa_chain.ainvoke({'query': query})
    yield 'data: [DONE]\n\n'

  # Return the streaming response
  return StreamingResponse(
    generate_response(), 
    media_type='text/event-stream'
  )
    
if __name__ == '__main__':
  port = int(os.environ.get('PORT', '8088'))
  logger.info(f'Starting server on port {port}')
  uvicorn.run(app, host='0.0.0.0', port=port)