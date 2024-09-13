print("Starting......")
import os
from langchain import PromptTemplate
import pickle
import pandas as pd
import pickle as pk
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from AuthAzure import SetEnv
from langchain.document_loaders import PyPDFDirectoryLoader,PyPDFLoader,DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
os.environ["http_proxy"] = "http://proxy-us.intel.com:911"
os.environ["https_proxy"] = "http://proxy-us.intel.com:912"
SetEnv()
path = "data/pdfs/"
# loader = PyPDFDirectoryLoader(path)
text_loader_kwargs = {"autodetect_encoding": True}
# loader = DirectoryLoader(
#     path,
#     glob="**/*.pdf",
#     loader_cls=PyPDFLoader,
#     recursive=True,
#     show_progress=True,
# )
# documents = loader.load()
# print("Total Documents",len(documents))
# with open("data/temp/temp_documents.pkl","wb") as e:
#     pk.dump(documents,e)
with open("data/temp/temp_documents.pkl","rb") as e:
    documents = pk.load(e)

text_splitter = CharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50, separator="\n"
)
docs = text_splitter.split_documents(documents)
print("Number of docs after splitting",len(docs))

embedding = OpenAIEmbeddings(deployment="ADA-002", chunk_size=1)
print(type(docs))
vectorstore = FAISS.from_documents(docs[:1], embedding)
for i, doc in enumerate(docs):
    if i % 1000 == 0:
        print("Getting new token",i)
        SetEnv()
        embedding = OpenAIEmbeddings(deployment="ADA-002", chunk_size=1)
    if i%50 == 0:
        print(i)
    faiss_index_i = FAISS.from_documents([doc], embedding)
    vectorstore.merge_from(faiss_index_i)

print("encoding done...\nSaving Vector index")
vectorstore.save_local("data/vector_store/pdfs_openai_ada")
print("Index saved")