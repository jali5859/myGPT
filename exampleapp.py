import os
import constants
import torch
import torch.nn.functional as F
import ssl
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PythonLoader
from sentence_transformers import SentenceTransformer


ssl._create_default_https_context = ssl._create_unverified_context

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Custom class for MiniLM embeddings
class MiniLMEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, texts):
        return self.model.encode(texts)
    
    def embed_documents(self, documents):
        # Assuming documents is a list of text strings
        return self.encode(documents)

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#directory
directory = '/Users/justinali/pythonapp/etl'

#Loading directory containing text files
def load_docs(directory):
  loader = DirectoryLoader(directory,glob="**/*.py",loader_cls=PythonLoader)
  documents = loader.load()
  return documents

#splitting the loaded text files
def split_docs(documents,chunk_size=1000,chunk_overlap=0):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

documents = load_docs(directory)
len(documents)
docs = split_docs(documents)

#save embeddings on disk from db
persist_directory = 'db/repos'
embedding = MiniLMEmbeddings()
vectordb = Chroma.from_documents(
  documents=docs,
  embedding=embedding,
  persist_directory=persist_directory
)

retriever = vectordb.as_retriever(search_kwargs={"k": 1})
retriever.search_type
newdocs=retriever.get_relevant_documents("backend files")

len(newdocs)

# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    # print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        
# full example
query = "what does the app.py do?"
llm_response = qa_chain(query)
process_llm_response(llm_response)