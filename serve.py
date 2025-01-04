#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### Define options ####

llm_model = 'gpt-4o'
embeddings_model = 'text-embedding-ada-002'


#### Define LLM ####

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=llm_model)


#### Construct retriever and get vector store ####

import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

index_name = 'schh'
text_field = 'text'

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings( model=embeddings_model, openai_api_key=os.getenv('OPENAPI_API_KEY') )
vectorstore = PineconeVectorStore( index, embeddings, text_field )  
retriever = vectorstore.as_retriever()


#### Contextualize question ####
 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

contextualize_q_system_prompt = '''Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.'''

contextualize_q_prompt = ChatPromptTemplate.from_messages(
  [
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
  ]
)
history_aware_retriever = create_history_aware_retriever(
  llm, retriever, contextualize_q_prompt
)


#### Answer question ####

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

system_prompt = '''You are an expert assistant tasked with answering questions using the provided context. \
Prioritize the context to generate your response. If the context is insufficient to fully address the question, \
use your general knowledge cautiously. Avoid making up information or guessing. \
\
Guidelines: \
1. Use the context as the primary source for your answers. \
2. If additional information is needed and not provided in the context, \rely on your general knowledge. \
3. All references to Sun City refer to Sun City Hilton Head, unless otherwise specified. \
4. All output is returned as Markdown formatted text. \

{context}'''

qa_prompt = ChatPromptTemplate.from_messages(
  [
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
  ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


#### Statefully manage chat history ####

import json
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
  rag_chain,
  get_session_history,
  input_messages_key='input',
  history_messages_key='chat_history',
  output_messages_key='answer',
).with_config(tags=['main_chain'])

def print_response(resp):
  as_dict = {
    'input': resp['input'],
    'chat_history': [doc.model_dump() for doc in resp['chat_history']],
    'context': [doc.model_dump() for doc in resp['context']],
    'answer': resp['answer']
  }
  print(json.dumps(as_dict, indent=2) + '\n')

def print_session_data(sessionid):
  print(json.dumps(store[sessionid].model_dump(), indent=2))


#### Output streamer ####

from langchain_core.messages import AIMessageChunk

async def generate_chat_events(message, session_id):
  
  def serialize_aimessagechunk(chunk):
    if isinstance(chunk, AIMessageChunk):
      return chunk.content
    else:
      raise TypeError(f'Object of type {type(chunk).__name__} is not correctly formatted for serialization')
  
  try:
    async for event in conversational_rag_chain.astream_events(message, version='v1', config={'configurable': {'session_id': session_id}} ):
      # print(event['tags'], event['event'], event.get('data',{}).get('chunk'))
      # Only get the answer
      sources_tags = ['seq:step:3', 'main_chain']
      if all(value in event['tags'] for value in sources_tags) and event['event'] == 'on_chat_model_stream':
        chunk_content = serialize_aimessagechunk(event['data']['chunk'])
        if len(chunk_content) != 0:
          yield chunk_content
          
  except Exception as e:
    print('error'+ str(e))
    
#### FastAPI server ####

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import secrets

app = FastAPI()
class CacheControlStaticFiles(StaticFiles):
  def file_response(self, *args, **kwargs) -> Response:
    response = super().file_response(*args, **kwargs)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

app.mount("/static", CacheControlStaticFiles(directory="static"), name="static")
app.mount('/static', StaticFiles(directory='static'), name='static')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get('/')
async def root():
  html = open('index.html', 'r').read()

  html = html.replace('MODEL_NAME', llm_model)
  return FileResponse('index.html')

@app.get('/manifest.json') # For PWA
async def pwa_manifest():
  return FileResponse('manifest.json')

@app.get('/chat/{prompt}')
async def chat(prompt: str, sessionid: Optional[str] = None, stream: Optional[bool] = False):
  
  sessionid = sessionid or secrets.token_hex(4) # Generates 4 bytes, resulting in an 8-character hex string

  if stream:
    return StreamingResponse(generate_chat_events({'input': prompt, 'chat_history': []}, sessionid), media_type='text/event-stream')
  else:
    resp = conversational_rag_chain.invoke({'input': prompt}, config={'configurable': {'session_id': sessionid}}, )
    return Response(status_code=200, content=resp['answer'], media_type='text/plain')

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)