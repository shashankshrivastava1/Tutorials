import os
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message

print("importing complete")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
import os

os.environ['http_proxy'] = 'http://proxy-us.intel.com:911'
os.environ['https_proxy'] = 'http://proxy-us.intel.com:912'

if __name__ == "__main__":
    print("question answering document")
    # files = os.listdir("pdfs/")
    path = "C:/Users/shashan3/OneDrive - Intel Corporation/Documents/CCG_Jarvis/SyncDictGeneration/data/documents/pdfs"
    # # path = "C:/Users/shashan3/OneDrive - Intel Corporation/Documents/Training/LangChainCourse/RAG/pdfs_civil"
    st.header("Intel Synonym Finder \n\
    # helping in finding synonyms with Openai LLMs")
    prompt = st.text_input("Prompt", placeholder="Please enter you query")
    #
    # loader = PyPDFDirectoryLoader(path)
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=30, separator="\n"
    # )
    # docs = text_splitter.split_documents(documents)
    #
    embedding = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(docs,embedding)
    # vectorstore.save_local("local_faiss")
    new_vectorestore = FAISS.load_local("local_faiss", embedding)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
                                     retriever=new_vectorestore.as_retriever(), return_source_documents=True)
    # query = ("Please tell me some synonyms words for s0ix? \

    # what would be synonyms for edp")
    print(prompt)
    if prompt:
        with st.spinner("Generating response..."):
            generated_response = qa({"query": prompt})
            doc = generated_response["source_documents"]
            print(generated_response["result"])

            # metadeta ="-----\n"
            # for i in generated_response["source_documents"]:
            #     doc = i.metadata["source"].split("\\")[-1]
            #     pageNumber = i.metadata["page"]
            #     print(doc,pageNumber)
            #     metadeta+= "Document:"+doc +" PageNumber:"+str(pageNumber)+"\n"
            #     # print("Document:",doc)
            #     # print("Page Number:",pageNumber,"\n---\n")
            # print(metadeta)

