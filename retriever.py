# server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import constants
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List, Dict

os.environ["ACTIVELOOP_TOKEN"] = constants.ACTIVELOOP_APIKEY
os.environ["OPENAI_API_KEY"] = constants.OPENAI_APIKEY
username = os.environ["USERNAME"] = constants.USERNAME

model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)
embeddings = OpenAIEmbeddings(disallowed_special=())
dbl = DeepLake(dataset_path=f"hub://{username}/bwebsite",read_only=True, embedding=embeddings)

def mainRetriever(model):
    retriever = dbl.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 200
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    return retriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str
    chat_history: List[str] = []

def get_chat_history(inputs) -> str:
    res = []
    for message in inputs:
        res.append(f"{message['role'].capitalize()}: {message['content']}")
    return "\n".join(res)

@app.post('/message')
async def handle_message(message: Message):
    user_query = message.message
    chat_history = message.chat_history
    retriever = mainRetriever(model)
    print(user_query)
    print(chat_history)
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever,memory=ConversationBufferMemory(chat_history))
    result = qa({"question" : user_query, "chat_history": chat_history})

    return {'response': result['answer']}