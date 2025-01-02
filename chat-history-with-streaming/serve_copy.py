#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bs4
import json
from typing import Optional
import secrets

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessageChunk

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

### Construct retriever ###
loader = WebBaseLoader(
  web_paths=('https://lilianweng.github.io/posts/2023-06-23-agent/',),
  bs_kwargs=dict(
    parse_only=bs4.SoupStrainer(class_=('post-content', 'post-title', 'post-header'))
  )
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = '''Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.'''
contextualize_q_prompt = ChatPromptTemplate.from_messages(
  [
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}')
  ]
)
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

contextualize_q_chain = (contextualize_q_prompt | llm | StrOutputParser()).with_config(tags=['contextualize_q_chain'])

qa_system_prompt = '''You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}'''
qa_prompt = ChatPromptTemplate.from_messages(
  [
    ('system', qa_system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
  ]
)

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
  history_aware_retriever, 
  question_answer_chain
).with_config(tags=['main_chain'])

'''
rag_chain = (
  RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
  | qa_prompt
  | llm
).with_config(tags=['main_chain'])
'''

conversational_rag_chain = RunnableWithMessageHistory(
  rag_chain,
  get_session_history,
  input_messages_key='input',
  history_messages_key='chat_history',
  output_messages_key='answer'
)

async def generate_chat_events(message, session_id):
  
  def serialize_aimessagechunk(chunk):
    '''
    Custom serializer for AIMessageChunk objects.
    Convert the AIMessageChunk object to a serializable format.
    '''
    if isinstance(chunk, AIMessageChunk):
      return chunk.content
    else:
      raise TypeError(f'Object of type {type(chunk).__name__} is not correctly formatted for serialization')
  
  try:
    async for event in conversational_rag_chain.astream_events(message, version='v1', config={'configurable': {'session_id': session_id}} ):
      print(event['tags'], event['event'], event.get('data',{}).get('chunk'))
      # Only get the answer
      sources_tags = ['seq:step:3', 'main_chain']
      if all(value in event['tags'] for value in sources_tags) and event['event'] == 'on_chat_model_stream':
        chunk_content = serialize_aimessagechunk(event['data']['chunk'])
        if len(chunk_content) != 0:
          # data_dict = {'data': chunk_content}
          # data_json = json.dumps(data_dict)
          # yield f'data: {data_json}\n\n'
          yield chunk_content
          
  except Exception as e:
    print('error'+ str(e))

def print_response(resp):
  as_dict = {
    'input': resp['input'],
    'chat_history': [doc.model_dump() for doc in resp['chat_history']],
    'context': [doc.model_dump() for doc in resp['context']],
    'answer': resp['answer']
  }
  print(json.dumps(as_dict, indent=2) + '\n')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get('/')
async def root():
  return FileResponse('index.html')

@app.get('/chat/{prompt}')
async def chat(prompt: str, sessionid: Optional[str] = None, stream: Optional[bool] = False):
  sessionid = sessionid or secrets.token_hex(4) # Generates 4 bytes, resulting in an 8-character hex string
  print(f'Prompt: {prompt} Session ID: {sessionid} Stream: {stream}')
  
  if stream:
    return StreamingResponse(generate_chat_events({'input': prompt, 'chat_history': []}, sessionid), media_type='text/event-stream')
  else:
    resp = conversational_rag_chain.invoke({'input': prompt}, config={'configurable': {'session_id': sessionid}}, )
    # print(json.dumps(store[sessionid].model_dump(), indent=2))
    # print_response(resp)
    return Response(status_code=200, content=resp['answer'], media_type='text/plain')
  
if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)