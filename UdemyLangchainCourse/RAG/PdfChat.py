print("Starting......")
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader,TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from AuthAzure import SetEnv
from langchain.llms import OpenAI
import streamlit as st
from streamlit_chat import message

print("importing complete")
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
import os

os.environ["http_proxy"] = "http://proxy-us.intel.com:911"
os.environ["https_proxy"] = "http://proxy-us.intel.com:912"
@st.cache_data
def load_data_st(faiss_dump,_embedding):
    print("Loading from local....")
    return FAISS.load_local(faiss_dump, embedding)
@st.cache_data
def set_env(input_variable):
    print("Setting Env Variable")
    SetEnv()

if __name__ == "__main__":
    print("question answering document")
    set_env("1")
    faiss_dump = "data/vector_store/pdfs_openai_ada"
    print("embedding model initialized... ")
    embedding = OpenAIEmbeddings(deployment="ADA-002", chunk_size=1)
    new_vectorestore = load_data_st(faiss_dump,embedding)
    az_model = AzureChatOpenAI(deployment_name="GPT4", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=az_model,
        chain_type="stuff",
        retriever=new_vectorestore.as_retriever(),
        return_source_documents=True,
    )

    st.header("Intel Synonym Finder \n\
    # helping in finding synonyms with Openai LLMs")
    if (
            "chat_answers_history" not in st.session_state
            and "user_prompt_history" not in st.session_state
            and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    prompt = st.text_input("Prompt",placeholder="Please enter you query")
    print(prompt)
    if prompt:
        with st.spinner("Generating response..."):
            generated_response = qa({"query":prompt})
            doc = generated_response["source_documents"]
            print(generated_response["result"])
            for i in generated_response["source_documents"]:
                doc = i.metadata["source"]#.split("\\")[-1]
                # pageNumber = i.metadata["page"]
                print(doc,i.metadata["page"])
            st.session_state.chat_history.append((prompt, generated_response["result"]))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(generated_response["result"])

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(
                user_query,
                is_user=True,
            )
            message(generated_response)
