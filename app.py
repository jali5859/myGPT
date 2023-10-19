import os
import constants
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ["ACTIVELOOP_TOKEN"] = constants.ACTIVELOOP_APIKEY
os.environ["OPENAI_API_KEY"] = constants.OPENAI_APIKEY

embeddings = OpenAIEmbeddings(disallowed_special=())
model = ChatOpenAI(model='gpt-4')

root_dir = './bwebsite'
docs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass
print(len(docs))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(docs)

username = "jali5859"
db = DeepLake(dataset_path=f"hub://{username}/bwebsite", embedding=embeddings)
db.add_documents(texts)

dbl = DeepLake(dataset_path=f"hub://{username}/bwebsite",read_only=True, embedding=embeddings)

retriever = dbl.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

questions = ["What componenets does this application have?","What sub componenets does the CustomHeader component have?"]

chat_history = []

for question in questions:
    result = qa({"question" : question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"-> **Answer**: {result['answer']} \n")
